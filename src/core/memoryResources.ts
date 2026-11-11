import { finiteAtLeast, integerAtLeast } from "./numeric";
import { safeCall } from "./observers";

export type MemoryDomain = "cpu" | "renderer" | "work";

/** A configured accounting refusal, independent of allocation or decode failure. */
export class ResourcePolicyError extends Error {
  readonly domain: MemoryDomain;
  readonly requestedBytes: number;
  readonly limitBytes: number;
  readonly retryable: boolean;

  constructor(
    domain: MemoryDomain,
    requestedBytes: number,
    limitBytes: number,
    retryable = false,
  ) {
    super(
      `${domain} memory requires ${requestedBytes} bytes; configured limit is ${limitBytes} bytes`,
    );
    this.name = "ResourcePolicyError";
    this.domain = domain;
    this.requestedBytes = requestedBytes;
    this.limitBytes = limitBytes;
    this.retryable = retryable;
  }
}

export const resourcePolicyError = (
  error: unknown,
): ResourcePolicyError | null => {
  let current = error;
  const seen = new Set<unknown>();
  while (current instanceof Error && !seen.has(current)) {
    if (current instanceof ResourcePolicyError) return current;
    seen.add(current);
    current = current.cause;
  }
  return null;
};

export type MemoryLease = {
  share(): MemoryLease;
  /** Replace realized CPU ownership and the still-unrealized CPU reservation atomically. */
  setCpu(buffers: Iterable<ArrayBuffer>, reservedBytes?: number): void;
  /** Only reduces a renderer reservation, for retiring part of a replacement group. */
  shrinkRenderer(bytes: number): void;
  release(): void;
};

export type MemoryWork = {
  /** Grow or shrink the estimate for all overlapping allocations in this operation. */
  resize(bytes: number): void;
  bytes(): number;
  /** Transfer finished output into retained ownership and relinquish the work slot. */
  commit(buffers: Iterable<ArrayBuffer>, estimatedBytes?: number): MemoryLease;
  /** Call only once the physical operation has settled, including after cancellation. */
  release(): void;
};

export type MemoryOwner = {
  retain(buffers: Iterable<ArrayBuffer>, estimatedBytes?: number): MemoryLease;
  reserveRenderer(
    bytes: number,
    options?: {
      cpuBytes?: number;
      replacing?: readonly MemoryLease[];
    },
  ): MemoryLease;
  reserveRendererGroup(
    amounts: readonly {
      readonly rendererBytes: number;
      readonly cpuBytes: number;
    }[],
    replacing?: readonly MemoryLease[],
  ): MemoryLease[];
  work(signal?: AbortSignal, estimatedBytes?: number): Promise<MemoryWork>;
  /** Stop accepting work. Existing leases and physical work keep their charges. */
  close(): void;
};

export type MemoryResourceOptions = {
  readonly totalBytes: number;
  readonly cpuBytes: number;
  readonly workBytes: number;
  readonly transitionBytes: number;
  readonly maxConcurrentOperations: number;
};

export type MemoryResourceStats = MemoryResourceOptions & {
  readonly cpuRetainedBytes: number;
  readonly cpuReservedBytes: number;
  readonly rendererBytes: number;
  readonly workReservedBytes: number;
  readonly activeOperations: number;
  readonly queuedOperations: number;
  readonly owners: number;
  readonly revision: number;
  readonly availabilityRevision: number;
};

type OwnerEntry = {
  closed: boolean;
  readonly leases: Set<LeaseEntry>;
  readonly operations: Set<WorkEntry>;
  readonly reclaim?: (domain: MemoryDomain) => void;
};

type LeaseEntry = {
  readonly owner: OwnerEntry;
  buffers: Set<ArrayBuffer>;
  cpuReserved: number;
  renderer: number;
  released: boolean;
  references: number;
  readonly replaces: readonly LeaseEntry[];
};

type WorkEntry = {
  readonly owner: OwnerEntry;
  bytes: number;
  released: boolean;
};
type WaitingWork = {
  readonly owner: OwnerEntry;
  readonly signal?: AbortSignal;
  readonly bytes: number;
  readonly resolve: (work: MemoryWork) => void;
  readonly reject: (error: unknown) => void;
  readonly abort: () => void;
};

const abortError = (): DOMException =>
  new DOMException("Memory operation cancelled", "AbortError");
const bytesValue = (name: string, value: number): number =>
  Math.ceil(finiteAtLeast(name, value, 0));

/** Shared ownership ledger. Scheduling membership never owns these allocations. */
export const createMemoryResources = (
  initial: MemoryResourceOptions,
  changed: () => void,
) => {
  let limits = initial;
  let revision = 0;
  let availabilityRevision = 0;
  let cpuRetained = 0;
  let cpuReserved = 0;
  let renderer = 0;
  let workReserved = 0;
  let operations = 0;
  let reclaiming = false;
  let pumping = false;
  const owners = new Set<OwnerEntry>();
  const buffers = new Map<ArrayBuffer, { references: number; bytes: number }>();
  const leases = new WeakMap<MemoryLease, LeaseEntry>();
  const claimedReplacements = new Map<LeaseEntry, number>();
  const waiting: WaitingWork[] = [];

  const retireOwner = (owner: OwnerEntry): void => {
    if (owner.closed && owner.leases.size === 0 && owner.operations.size === 0)
      owners.delete(owner);
  };
  const reclaim = (domain: MemoryDomain): void => {
    if (reclaiming) return;
    reclaiming = true;
    try {
      // Reclaimers may close owners and mutate the registry.
      // oxlint-disable-next-line unicorn/no-useless-spread
      for (const owner of [...owners]) safeCall(() => owner.reclaim?.(domain));
    } finally {
      reclaiming = false;
    }
  };
  const replaceBuffers = (entry: LeaseEntry, next: Set<ArrayBuffer>): void => {
    for (const buffer of entry.buffers) {
      if (next.has(buffer)) continue;
      const held = buffers.get(buffer)!;
      held.references -= 1;
      if (held.references === 0) {
        buffers.delete(buffer);
        cpuRetained -= held.bytes;
      }
    }
    for (const buffer of next) {
      if (entry.buffers.has(buffer)) continue;
      const held = buffers.get(buffer);
      if (held) held.references += 1;
      else {
        buffers.set(buffer, { references: 1, bytes: buffer.byteLength });
        cpuRetained += buffer.byteLength;
      }
    }
    entry.buffers = next;
  };

  let pump: () => void;
  const makeLease = (
    owner: OwnerEntry,
    cpuBytes: number,
    rendererBytes: number,
    replaces: readonly LeaseEntry[] = [],
  ): MemoryLease => {
    const entry: LeaseEntry = {
      owner,
      buffers: new Set(),
      cpuReserved: cpuBytes,
      renderer: rendererBytes,
      released: false,
      references: 0,
      replaces,
    };
    owner.leases.add(entry);
    cpuReserved += cpuBytes;
    renderer += rendererBytes;
    for (const old of replaces)
      claimedReplacements.set(old, (claimedReplacements.get(old) ?? 0) + 1);
    const handle = (): MemoryLease => {
      entry.references += 1;
      let released = false;
      const api: MemoryLease = {
        share() {
          if (released || entry.released)
            throw new Error("allocation already released");
          return handle();
        },
        setCpu(values, reservedBytes = 0) {
          if (released || entry.released) return;
          const reserved = bytesValue("reserved CPU bytes", reservedBytes);
          replaceBuffers(entry, new Set(values));
          cpuReserved += reserved - entry.cpuReserved;
          entry.cpuReserved = reserved;
          changed();
        },
        shrinkRenderer(bytes) {
          if (released || entry.released) return;
          const next = bytesValue("renderer bytes", bytes);
          if (next > entry.renderer)
            throw new RangeError("renderer reservations may only shrink");
          renderer -= entry.renderer - next;
          entry.renderer = next;
          availabilityRevision += 1;
          changed();
        },
        release() {
          if (released || entry.released) return;
          released = true;
          entry.references -= 1;
          if (entry.references > 0) return;
          entry.released = true;
          replaceBuffers(entry, new Set());
          cpuReserved -= entry.cpuReserved;
          renderer -= entry.renderer;
          for (const old of entry.replaces) {
            const remaining = claimedReplacements.get(old)! - 1;
            if (remaining === 0) claimedReplacements.delete(old);
            else claimedReplacements.set(old, remaining);
          }
          owner.leases.delete(entry);
          retireOwner(owner);
          availabilityRevision += 1;
          changed();
          pump();
        },
      };
      leases.set(api, entry);
      return api;
    };
    return handle();
  };

  const checkCpu = (extra: number | (() => number)): void => {
    const additional = () => (typeof extra === "number" ? extra : extra());
    if (cpuRetained + cpuReserved + additional() > limits.cpuBytes)
      reclaim("cpu");
    const added = additional();
    if (cpuRetained + cpuReserved + added > limits.cpuBytes) {
      throw new ResourcePolicyError(
        "cpu",
        cpuRetained + cpuReserved + added,
        limits.cpuBytes,
        added <= limits.cpuBytes,
      );
    }
  };

  const retain = (
    owner: OwnerEntry,
    values: Iterable<ArrayBuffer>,
    estimate = 0,
  ): MemoryLease => {
    const unique = new Set(values);
    const extra = bytesValue("estimated retained CPU bytes", estimate);
    checkCpu(() => {
      let added = extra;
      for (const buffer of unique)
        if (!buffers.has(buffer)) added += buffer.byteLength;
      return added;
    });
    const lease = makeLease(owner, extra, 0);
    lease.setCpu(unique, extra);
    return lease;
  };

  const startWork = (owner: OwnerEntry, bytes: number): MemoryWork => {
    const entry: WorkEntry = { owner, bytes, released: false };
    owner.operations.add(entry);
    operations += 1;
    workReserved += bytes;
    const release = (): void => {
      if (entry.released) return;
      entry.released = true;
      operations -= 1;
      workReserved -= entry.bytes;
      owner.operations.delete(entry);
      availabilityRevision += 1;
      retireOwner(owner);
      changed();
      pump();
    };
    return {
      resize(value) {
        if (entry.released) throw new Error("memory work already settled");
        const next = bytesValue("work bytes", value);
        const requested = workReserved - entry.bytes + next;
        if (requested > limits.workBytes)
          throw new ResourcePolicyError(
            "work",
            requested,
            limits.workBytes,
            next <= limits.workBytes,
          );
        workReserved = requested;
        entry.bytes = next;
        changed();
      },
      bytes: () => entry.bytes,
      commit(values, estimate = 0) {
        if (entry.released) throw new Error("memory work already settled");
        const lease = retain(owner, values, estimate);
        release();
        return lease;
      },
      release,
    };
  };

  const reserveRendererGroup = (
    owner: OwnerEntry,
    amounts: readonly {
      readonly rendererBytes: number;
      readonly cpuBytes: number;
    }[],
    replacements: readonly MemoryLease[] = [],
  ): MemoryLease[] => {
    const normalized = amounts.map((amount) => ({
      rendererBytes: bytesValue("renderer bytes", amount.rendererBytes),
      cpuBytes: bytesValue("reserved CPU bytes", amount.cpuBytes),
    }));
    const amount = normalized.reduce(
      (sum, item) => sum + item.rendererBytes,
      0,
    );
    const cpu = normalized.reduce((sum, item) => sum + item.cpuBytes, 0);
    const replacing = [
      ...new Set(replacements.map((lease) => leases.get(lease))),
    ];
    if (
      replacing.some(
        (entry) =>
          !entry ||
          entry.owner !== owner ||
          entry.released ||
          claimedReplacements.has(entry),
      )
    ) {
      throw new Error("replacement must own unclaimed live allocations");
    }
    const old = replacing as LeaseEntry[];
    const credit = () =>
      old.reduce(
        (sum, entry) => sum + (entry.released ? 0 : entry.renderer),
        0,
      );
    const ceiling =
      limits.totalBytes + (old.length > 0 ? limits.transitionBytes : 0);
    if (
      renderer + amount > ceiling ||
      renderer + amount - credit() > limits.totalBytes
    )
      reclaim("renderer");
    const finalBytes = renderer + amount - credit();
    if (finalBytes > limits.totalBytes)
      throw new ResourcePolicyError(
        "renderer",
        finalBytes,
        limits.totalBytes,
        amount <= limits.totalBytes,
      );
    if (renderer + amount > ceiling)
      throw new ResourcePolicyError(
        "renderer",
        renderer + amount,
        ceiling,
        amount <= limits.totalBytes,
      );
    checkCpu(cpu);
    const result = normalized.map((item) =>
      makeLease(owner, item.cpuBytes, item.rendererBytes, old),
    );
    changed();
    return result;
  };

  pump = (): void => {
    if (pumping || reclaiming) return;
    pumping = true;
    try {
      while (
        waiting.length > 0 &&
        operations < limits.maxConcurrentOperations
      ) {
        const next = waiting[0]!;
        if (next.owner.closed || next.signal?.aborted) {
          waiting.shift();
          next.signal?.removeEventListener("abort", next.abort);
          next.reject(abortError());
          continue;
        }
        if (next.bytes > limits.workBytes) {
          waiting.shift();
          next.signal?.removeEventListener("abort", next.abort);
          next.reject(
            new ResourcePolicyError("work", next.bytes, limits.workBytes),
          );
          continue;
        }
        if (workReserved + next.bytes > limits.workBytes) break;
        waiting.shift();
        next.signal?.removeEventListener("abort", next.abort);
        next.resolve(startWork(next.owner, next.bytes));
      }
    } finally {
      pumping = false;
    }
    changed();
  };

  return {
    createOwner(
      options: { readonly reclaim?: (domain: MemoryDomain) => void } = {},
    ): MemoryOwner {
      const owner: OwnerEntry = {
        closed: false,
        leases: new Set(),
        operations: new Set(),
        reclaim: options.reclaim,
      };
      owners.add(owner);
      const assertOpen = (): void => {
        if (owner.closed) throw abortError();
      };
      return {
        retain(values, estimate) {
          assertOpen();
          return retain(owner, values, estimate);
        },
        reserveRenderer(value, options = {}) {
          assertOpen();
          return reserveRendererGroup(
            owner,
            [{ rendererBytes: value, cpuBytes: options.cpuBytes ?? 0 }],
            options.replacing,
          )[0]!;
        },
        reserveRendererGroup(amounts, replacing) {
          assertOpen();
          return reserveRendererGroup(owner, amounts, replacing);
        },
        work(signal, estimate = 0) {
          assertOpen();
          if (signal?.aborted) return Promise.reject(abortError());
          const bytes = bytesValue("estimated work bytes", estimate);
          return new Promise((resolve, reject) => {
            const item: WaitingWork = {
              owner,
              signal,
              bytes,
              resolve,
              reject,
              abort: () => {
                const index = waiting.indexOf(item);
                if (index < 0) return;
                waiting.splice(index, 1);
                reject(abortError());
                pump();
              },
            };
            waiting.push(item);
            signal?.addEventListener("abort", item.abort, { once: true });
            pump();
          });
        },
        close() {
          if (owner.closed) return;
          owner.closed = true;
          // Removing one queued operation mutates the waiting list.
          // oxlint-disable-next-line unicorn/no-useless-spread
          for (const item of [...waiting])
            if (item.owner === owner) item.abort();
          retireOwner(owner);
          changed();
        },
      };
    },
    configure(options: MemoryResourceOptions) {
      for (const [name, value] of Object.entries(options)) {
        if (name === "maxConcurrentOperations") integerAtLeast(name, value, 1);
        else bytesValue(name, value);
      }
      limits = options;
      revision += 1;
      availabilityRevision += 1;
      reclaim("renderer");
      reclaim("cpu");
      pump();
    },
    stats(): MemoryResourceStats {
      return {
        ...limits,
        cpuRetainedBytes: cpuRetained,
        cpuReservedBytes: cpuReserved,
        rendererBytes: renderer,
        workReservedBytes: workReserved,
        activeOperations: operations,
        queuedOperations: waiting.length,
        owners: owners.size,
        revision,
        availabilityRevision,
      };
    },
  };
};
