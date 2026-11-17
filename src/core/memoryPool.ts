/** Shared, host-injected admission policy for participating scene members. */

import { finiteAtLeast, integerAtLeast } from "./numeric";
import {
  createMemoryResources,
  type MemoryOwner,
  type MemoryDomain,
  type MemoryResourceStats,
} from "./memoryResources";

const MiB = 1024 * 1024;

/** Fallback byte budget when the environment reports nothing about memory. */
export const DEFAULT_MEMORY_BUDGET_BYTES = 512 * MiB;

/**
 * Estimate a byte budget for resident tile data on this device.
 *
 * This conservative policy hint uses approximate reported RAM, never free
 * VRAM. Hosts can configure the allowance independently of this heuristic.
 */
export const defaultMemoryBudgetBytes = (): number => {
  const nav = (globalThis as { navigator?: { deviceMemory?: unknown } })
    .navigator;
  const deviceGb = nav?.deviceMemory;
  if (
    typeof deviceGb === "number" &&
    Number.isFinite(deviceGb) &&
    deviceGb > 0
  ) {
    return Math.min(
      Math.max((deviceGb * 1024 * MiB) / 8, 256 * MiB),
      1024 * MiB,
    );
  }
  return DEFAULT_MEMORY_BUDGET_BYTES;
};

export type MemoryPoolMember = {
  /** Current demand-aware byte grant. Admission still checks shared ownership. */
  budgetBytes(): number;
  update(demand: MemoryDemand): void;
  /** Leave active scheduling. Resource owners retain their independent lifetime. */
  release(): void;
};

export type MemoryDemand = {
  readonly minimumBytes: number;
  readonly desiredBytes: number;
  readonly residentBytes: number;
};

export type MemoryPoolOptions = {
  /** Managed renderer working-set admission ceiling. */
  totalBytes?: number;
  /** Retained CPU buffers and reserved realization arrays. Default twice totalBytes. */
  cpuBytes?: number;
  /** Concurrent work's estimated overlapping allocations. Default twice totalBytes. */
  workBytes?: number;
  /** Extra renderer capacity usable only for a complete replacement transition. */
  transitionBytes?: number;
  /** Shared fetch/decode operations, including queued worker inputs. Default 2. */
  maxConcurrentOperations?: number;
};

export type MemoryPool = {
  /**
   * Join the pool. `onChange` fires whenever this member's share moves
   * because another member joined or left, but never during this `register`
   * call itself.
   */
  register(onChange?: () => void): MemoryPoolMember;
  createOwner(options?: {
    readonly reclaim?: (domain: MemoryDomain) => void;
  }): MemoryOwner;
  /** Update limits. Existing allocations remain charged while members coarsen. */
  configure(options: MemoryPoolOptions): void;
  memberCount(): number;
  stats(): MemoryResourceStats & {
    readonly memberCount: number;
    readonly shareBytes: number;
  };
};

export const createMemoryPool = (
  options: MemoryPoolOptions = {},
): MemoryPool => {
  const normalize = (value: MemoryPoolOptions) => {
    const totalBytes = Math.floor(
      finiteAtLeast(
        "totalBytes",
        value.totalBytes ?? defaultMemoryBudgetBytes(),
        0,
      ),
    );
    return {
      totalBytes,
      cpuBytes: Math.floor(
        finiteAtLeast("cpuBytes", value.cpuBytes ?? totalBytes * 2, 0),
      ),
      workBytes: Math.floor(
        finiteAtLeast("workBytes", value.workBytes ?? totalBytes * 2, 0),
      ),
      transitionBytes: Math.floor(
        finiteAtLeast(
          "transitionBytes",
          value.transitionBytes ?? Math.floor(totalBytes / 4),
          0,
        ),
      ),
      maxConcurrentOperations: integerAtLeast(
        "maxConcurrentOperations",
        value.maxConcurrentOperations ?? 2,
        1,
      ),
    };
  };
  let limits = normalize(options);
  type Entry = { onChange?: () => void; demand: MemoryDemand; grant: number };
  const members = new Set<Entry>();

  const notify = (except?: object): void => {
    // Snapshot deliberately: an `onChange` may release ANOTHER member, and a
    // Set drops a not-yet-visited entry deleted mid-iteration. Every member
    // registered when the change happened is entitled to hear about it.
    // oxlint-disable-next-line unicorn/no-useless-spread
    for (const member of [...members]) {
      if (member !== except) member.onChange?.();
    }
  };

  const allocate = (except?: Entry): void => {
    const entries = [...members];
    const previous = entries.map((entry) => entry.grant);
    let available = limits.totalBytes;
    for (const entry of entries) entry.grant = 0;
    // Stable insertion order preserves an existing feasible coverage frontier
    // when simultaneous minimum representations cannot all fit.
    for (const entry of entries) {
      const minimum = Math.min(
        entry.demand.minimumBytes,
        entry.demand.desiredBytes,
      );
      if (minimum > available) continue;
      entry.grant = minimum;
      available -= minimum;
    }
    let pending = entries.filter(
      (entry) =>
        entry.grant < entry.demand.desiredBytes &&
        entry.grant >=
          Math.min(entry.demand.minimumBytes, entry.demand.desiredBytes),
    );
    while (available > 0 && pending.length > 0) {
      const share = Math.max(1, Math.floor(available / pending.length));
      for (const entry of pending) {
        const extra = Math.min(
          share,
          available,
          entry.demand.desiredBytes - entry.grant,
        );
        entry.grant += extra;
        available -= extra;
      }
      pending = pending.filter(
        (entry) => entry.grant < entry.demand.desiredBytes,
      );
    }
    entries.forEach((entry, index) => {
      if (entry !== except && entry.grant !== previous[index])
        entry.onChange?.();
    });
  };
  let notificationQueued = false;
  const resourceChanged = (): void => {
    if (notificationQueued) return;
    notificationQueued = true;
    queueMicrotask(() => {
      notificationQueued = false;
      notify();
    });
  };
  const resources = createMemoryResources(limits, resourceChanged);

  return {
    register(onChange) {
      const entry: Entry = {
        onChange,
        demand: {
          minimumBytes: 0,
          desiredBytes: Number.POSITIVE_INFINITY,
          residentBytes: 0,
        },
        grant: 0,
      };
      members.add(entry);
      allocate(entry);
      return {
        budgetBytes: () => (members.has(entry) ? entry.grant : 0),
        update(demand) {
          if (!members.has(entry)) return;
          const valid = (value: number): number =>
            Number.isFinite(value) && value >= 0 ? Math.ceil(value) : 0;
          const next = {
            minimumBytes: valid(demand.minimumBytes),
            desiredBytes:
              demand.desiredBytes === Number.POSITIVE_INFINITY
                ? demand.desiredBytes
                : valid(demand.desiredBytes),
            residentBytes: valid(demand.residentBytes),
          };
          if (
            entry.demand.minimumBytes === next.minimumBytes &&
            entry.demand.desiredBytes === next.desiredBytes &&
            entry.demand.residentBytes === next.residentBytes
          )
            return;
          entry.demand = next;
          allocate();
        },
        release: () => {
          if (members.delete(entry)) allocate();
        },
      };
    },
    createOwner: resources.createOwner,
    configure(next) {
      limits = normalize({ ...limits, ...next });
      resources.configure(limits);
      allocate();
    },
    memberCount: () => members.size,
    stats: () => ({
      ...resources.stats(),
      memberCount: members.size,
      shareBytes: members.size
        ? Math.floor(limits.totalBytes / members.size)
        : limits.totalBytes,
    }),
  };
};
