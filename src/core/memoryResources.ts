import { finiteAtLeast, integerAtLeast } from "./numeric";
import { safeCall } from "./observers";
import type { RequestAdmission } from "./requestAdmission";

export type MemoryDomain = "cpu" | "renderer" | "work";

export type MemoryRefusalEstimate = {
  readonly reason: "configured-limit" | "contention" | "transition-unavailable";
  readonly finalBytes: number;
  readonly peakBytes: number;
  readonly wake: "policy" | "availability" | "presentation";
};

/** A configured accounting refusal, independent of allocation or decode failure. */
export class ResourcePolicyError extends Error {
  readonly domain: MemoryDomain;
  readonly requestedBytes: number;
  readonly limitBytes: number;
  readonly retryable: boolean;
  readonly operationBytes?: number;
  readonly estimate: MemoryRefusalEstimate;

  constructor(
    domain: MemoryDomain,
    requestedBytes: number,
    limitBytes: number,
    retryable = false,
    operationBytes?: number,
    estimate?: MemoryRefusalEstimate,
  ) {
    super(
      `${domain} memory requires ${requestedBytes} bytes; configured limit is ${limitBytes} bytes`,
    );
    this.name = "ResourcePolicyError";
    this.domain = domain;
    this.requestedBytes = requestedBytes;
    this.limitBytes = limitBytes;
    this.retryable = retryable;
    this.operationBytes = operationBytes;
    this.estimate = estimate ?? {
      reason: retryable ? "contention" : "configured-limit",
      finalBytes: requestedBytes,
      peakBytes: requestedBytes,
      wake: retryable ? "availability" : "policy",
    };
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
  /** Acquire another reference while any holder keeps this allocation alive. */
  reacquireWhileHeld(onLastRelease: () => void): () => MemoryLease | undefined;
  /** Replace realized CPU ownership and the still-unrealized CPU reservation atomically. */
  setCpu(buffers: Iterable<ArrayBuffer>, reservedBytes?: number): void;
  /** Only reduces a renderer reservation, for retiring part of a replacement group. */
  shrinkRenderer(bytes: number): void;
  release(): void;
};

/** Index a completed immutable payload for exactly as long as consumers own it. */
export const reuseMemoryResult = <T extends { memory?: MemoryLease }>(
  value: T,
  onLastRelease: () => void,
): (() => T | undefined) | undefined => {
  const acquire = value.memory?.reacquireWhileHeld(onLastRelease);
  return (
    acquire &&
    (() => {
      const memory = acquire();
      return memory && { ...value, memory };
    })
  );
};

export type MemoryWork = {
  /** Grow or shrink the estimate for all overlapping allocations in this operation. */
  resize(bytes: number): void;
  bytes(): number;
  /** Transfer finished output into retained ownership and relinquish reserved work bytes. */
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
  /**
   * False when a live owned allocation's replacement credit is already claimed.
   * Foreign, released, or unknown allocations throw.
   */
  replacementAvailable(lease: MemoryLease): boolean;
  reserveRendererGroup(
    amounts: readonly {
      readonly rendererBytes: number;
      readonly cpuBytes: number;
    }[],
    replacing?: readonly MemoryLease[],
  ): MemoryLease[];
  readonly admission: RequestAdmission;
  /** Scope result reuse to owners backed by this allocation ledger. */
  scopeForSharing(source: object): object;
  availability(): number;
  canRetry(error: ResourcePolicyError): boolean;
  canReserveWork(bytes: number): boolean;
  /** Reserve immediately or throw a policy refusal. The request service handles deferral. */
  work(signal?: AbortSignal, estimatedBytes?: number): MemoryWork;
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
  /** Physical stages running under the shared page admission limit. */
  readonly activeOperations: number;
  readonly queuedOperations: number;
  readonly owners: number;
  readonly revision: number;
  readonly availabilityRevision: number;
};

type OwnerEntry = {
  closed: boolean;
  readonly leases: Set<LeaseEntry>;
  readonly reservations: Set<WorkEntry>;
  readonly reclaim?: (domain: MemoryDomain) => void;
};

type LeaseEntry = {
  readonly owner: OwnerEntry;
  buffers: Set<ArrayBuffer>;
  cpuReserved: number;
  renderer: number;
  released: boolean;
  references: number;
  onLastRelease?: Set<() => void>;
  readonly replaces: readonly LeaseEntry[];
};

type WorkEntry = {
  readonly owner: OwnerEntry;
  bytes: number;
  released: boolean;
};
const abortError = (): DOMException =>
  new DOMException("Memory operation cancelled", "AbortError");
const bytesValue = (name: string, value: number): number =>
  Math.ceil(finiteAtLeast(name, value, 0));

/** Shared ownership ledger. Scheduling membership never owns these allocations. */
export const createMemoryResources = (
  initial: MemoryResourceOptions,
  changed: () => void,
  admission: RequestAdmission,
) => {
  let limits = initial;
  let revision = 0;
  let availabilityRevision = 0;
  let cpuRetained = 0;
  let cpuReserved = 0;
  let renderer = 0;
  let workReserved = 0;
  let reclaiming = false;
  const owners = new Set<OwnerEntry>();
  const buffers = new Map<ArrayBuffer, { references: number; bytes: number }>();
  const leases = new WeakMap<MemoryLease, LeaseEntry>();
  const claimedReplacements = new Map<LeaseEntry, number>();
  const sharingScopes = new WeakMap<object, object>();
  const scopeForSharing = (source: object): object => {
    let scope = sharingScopes.get(source);
    if (!scope) {
      scope = {};
      sharingScopes.set(source, scope);
    }
    return scope;
  };

  const retireOwner = (owner: OwnerEntry): void => {
    if (
      owner.closed &&
      owner.leases.size === 0 &&
      owner.reservations.size === 0
    )
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
        reacquireWhileHeld(onLastRelease) {
          if (released || entry.released)
            throw new Error("allocation already released");
          (entry.onLastRelease ??= new Set()).add(onLastRelease);
          return () => (entry.released ? undefined : handle());
        },
        setCpu(values, reservedBytes = 0) {
          if (released || entry.released) return;
          const reserved = bytesValue("reserved CPU bytes", reservedBytes);
          const before = cpuRetained + cpuReserved;
          replaceBuffers(entry, new Set(values));
          cpuReserved += reserved - entry.cpuReserved;
          entry.cpuReserved = reserved;
          if (cpuRetained + cpuReserved < before) availabilityRevision += 1;
          changed();
        },
        shrinkRenderer(bytes) {
          if (released || entry.released) return;
          const next = bytesValue("renderer bytes", bytes);
          if (next > entry.renderer)
            throw new RangeError("renderer reservations may only shrink");
          if (entry.renderer === next) return;
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
          const previousBytes = cpuRetained + cpuReserved + renderer;
          let releasedCredit = false;
          replaceBuffers(entry, new Set());
          cpuReserved -= entry.cpuReserved;
          renderer -= entry.renderer;
          for (const old of entry.replaces) {
            const remaining = claimedReplacements.get(old)! - 1;
            if (remaining === 0) {
              claimedReplacements.delete(old);
              releasedCredit ||= !old.released;
            } else claimedReplacements.set(old, remaining);
          }
          owner.leases.delete(entry);
          retireOwner(owner);
          const listeners = entry.onLastRelease;
          entry.onLastRelease = undefined;
          listeners?.forEach(safeCall);
          if (
            releasedCredit ||
            cpuRetained + cpuReserved + renderer < previousBytes
          )
            availabilityRevision += 1;
          changed();
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
        added,
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
    owner.reservations.add(entry);
    workReserved += bytes;
    changed();
    const release = (): void => {
      if (entry.released) return;
      entry.released = true;
      workReserved -= entry.bytes;
      owner.reservations.delete(entry);
      if (entry.bytes > 0) availabilityRevision += 1;
      retireOwner(owner);
      changed();
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
            next,
          );
        if (next < entry.bytes) availabilityRevision += 1;
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
        undefined,
        {
          reason:
            amount <= limits.totalBytes ? "contention" : "configured-limit",
          finalBytes,
          peakBytes: renderer + amount,
          wake: amount <= limits.totalBytes ? "availability" : "policy",
        },
      );
    if (renderer + amount > ceiling) {
      const canFitWithCredit = credit() + amount <= ceiling;
      throw new ResourcePolicyError(
        "renderer",
        renderer + amount,
        ceiling,
        canFitWithCredit,
        undefined,
        {
          reason: "transition-unavailable",
          finalBytes,
          peakBytes: renderer + amount,
          wake: canFitWithCredit ? "availability" : "policy",
        },
      );
    }
    checkCpu(cpu);
    const result = normalized.map((item) =>
      makeLease(owner, item.cpuBytes, item.rendererBytes, old),
    );
    changed();
    return result;
  };

  return {
    createOwner(
      options: { readonly reclaim?: (domain: MemoryDomain) => void } = {},
    ): MemoryOwner {
      const owner: OwnerEntry = {
        closed: false,
        leases: new Set(),
        reservations: new Set(),
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
        replacementAvailable(lease) {
          assertOpen();
          const entry = leases.get(lease);
          if (!entry || entry.owner !== owner || entry.released)
            throw new Error("replacement must own unclaimed live allocations");
          return !claimedReplacements.has(entry);
        },
        reserveRendererGroup(amounts, replacing) {
          assertOpen();
          return reserveRendererGroup(owner, amounts, replacing);
        },
        admission,
        scopeForSharing,
        availability: () => availabilityRevision,
        canRetry(error) {
          if (error.operationBytes === undefined) return true;
          if (error.domain === "work")
            return workReserved + error.operationBytes <= limits.workBytes;
          if (error.domain === "cpu")
            return (
              cpuRetained + cpuReserved + error.operationBytes <=
              limits.cpuBytes
            );
          return true;
        },
        canReserveWork: (bytes) => workReserved + bytes <= limits.workBytes,
        work(signal, estimate = 0) {
          assertOpen();
          signal?.throwIfAborted();
          const bytes = bytesValue("estimated work bytes", estimate);
          if (workReserved + bytes > limits.workBytes)
            throw new ResourcePolicyError(
              "work",
              workReserved + bytes,
              limits.workBytes,
              bytes <= limits.workBytes,
              bytes,
            );
          return startWork(owner, bytes);
        },
        close() {
          if (owner.closed) return;
          owner.closed = true;
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
      changed();
    },
    availability: () => availabilityRevision,
    policyRevision: () => revision,
    stats(): MemoryResourceStats {
      return {
        ...limits,
        maxConcurrentOperations: admission.capacity(),
        cpuRetainedBytes: cpuRetained,
        cpuReservedBytes: cpuReserved,
        rendererBytes: renderer,
        workReservedBytes: workReserved,
        activeOperations: admission.activeOperations(),
        queuedOperations: admission.queuedOperations(),
        owners: owners.size,
        revision,
        availabilityRevision,
      };
    },
  };
};
