import {
  resourcePolicyError,
  type ResourcePolicyError,
} from "./memoryResources";
import { safeCall } from "./observers";
import { integerAtLeast } from "./numeric";
import {
  isResting,
  recordFailure,
  retryDelayMs,
  type FailureRecord,
  type RetryPolicy,
} from "./retryPolicy";

export type RequestLane = "network" | "compute" | "sort";

export type RequestShare<T> = {
  readonly scope: object;
  readonly key: string;
  /** Reacquire the immutable result while its existing allocation is held. */
  readonly reuse: (
    value: T,
    onLastRelease: () => void,
  ) => (() => T | undefined) | undefined;
};

export type RequestStage<T> = {
  readonly lane: RequestLane;
  /** Optional identity of an initial stage that acquires resources only when started. */
  readonly share?: RequestShare<T>;
  /** Called before admission. The executor signals when readiness changes. */
  readonly ready?: () => boolean;
  readonly start: (
    signal: AbortSignal,
  ) => RequestResult<T> | Promise<RequestResult<T>>;
  /** Release stage input after running or if demand withdraws while it waits. */
  readonly release?: () => void;
};

type RequestResult<T> =
  | { readonly value: T }
  | { readonly next: RequestStage<T> };

export type RequestClock = {
  now(): number;
  setTimeout(callback: () => void, delayMs: number): unknown;
  clearTimeout(handle: unknown): void;
};

export type RequestStatus =
  | "queued"
  | "active"
  | "retrying"
  | "resting"
  | "blocked"
  | "failed";
export type RequestState = {
  readonly status: RequestStatus;
  readonly attempt: number;
};

export type LoaderCounts = {
  readonly queued: number;
  readonly physical: number;
  readonly retrying: number;
  readonly reading: number;
  readonly wantedReading: number;
};

type RequestOwner = {
  /** Replace all demanded keys in the current priority order. */
  want(keys: readonly string[]): void;
  abandon(): void;
  reset(): void;
  dispose(): void;
  setConcurrency(limit: number): void;
  retryBlocked(): void;
  state(key: string): RequestState | undefined;
  physicalCount(): number;
  resting(key: string): boolean;
  restingCount(): number;
  counts(): LoaderCounts;
};

type OwnerOptions<T> = {
  readonly group?: object;
  readonly priority?: number;
  readonly concurrency: number;
  readonly cancelGraceMs?: number;
  readonly adoptCancelled?: boolean;
  readonly keepUnwantedResults?: boolean;
  readonly operation: (key: string) => RequestStage<T>;
  readonly availability?: () => number;
  readonly canRetry?: (error: ResourcePolicyError) => boolean;
  readonly retry?: RetryPolicy;
  readonly retryLimit?: number;
  readonly retryBackoffMs?: (attempt: number) => number;
  readonly shouldRetry?: (error: unknown) => boolean;
  readonly onRetryError?: (key: string, error: unknown) => unknown;
  readonly clock?: RequestClock;
  readonly onLoaded: (key: string, value: T, attempt: number) => void;
  readonly onDiscarded?: (value: T) => void;
  readonly onFailed: (key: string, error: unknown) => void;
  readonly onBlocked?: (key: string, refusal: ResourcePolicyError) => void;
  readonly onChange?: () => void;
};

type OneShotOptions<T> = {
  readonly group?: object;
  readonly availability?: () => number;
  readonly canRetry?: (error: ResourcePolicyError) => boolean;
  readonly signal: AbortSignal;
  readonly stage: RequestStage<T>;
  readonly onDiscarded?: (value: T) => void;
};

type Grant = {
  readonly group: object;
  readonly priority: number;
  start(): void;
};

type PendingOwner = {
  /** Explicit member identity for lending network capacity across groups. */
  readonly group?: object;
  offer(lane: RequestLane): Grant | undefined;
  queued(): number;
};

type Operation<T> = {
  readonly key: string;
  readonly controller: AbortController;
  stage: RequestStage<T>;
  status: RequestStatus;
  attempt: number;
  wanted: boolean;
  firstStage: boolean;
  refusal?: ResourcePolicyError;
  availability?: number;
  clearTimer?: () => void;
  clearGrace?: () => void;
  shared?: SharedOperation<T>;
};

type OperationOutcome<T> = { result: RequestResult<T> } | { error: unknown };

type SharedClaim<T> = {
  readonly operation: Operation<T>;
  readonly settle: (outcome: OperationOutcome<T>) => void;
  readonly change: () => void;
};

type SharedOperation<T> = {
  readonly controller: AbortController;
  readonly claims: Map<Operation<T>, SharedClaim<T>>;
  readonly remove: () => void;
  readonly publish: (value: T) => (() => T | undefined) | undefined;
  started: boolean;
  active: boolean;
  stage?: RequestStage<T>;
};

type SharedResult<T> = { readonly reuse: () => T | undefined };

const systemClock: RequestClock = {
  now: () => Date.now(),
  setTimeout: (callback, delayMs) => setTimeout(callback, delayMs),
  clearTimeout: (handle) =>
    clearTimeout(handle as ReturnType<typeof setTimeout>),
};
const isAbortError = (error: unknown): boolean =>
  error instanceof Error && error.name === "AbortError";

type WorkerKind = "copc" | "decode" | "sort";
type IdleWorker = { readonly idle: () => boolean; readonly retire: () => void };

const persistentLimits: Record<WorkerKind, number> = {
  copc: 4,
  decode: 4,
  sort: 2,
};

export type RequestAdmission = {
  createOwner<T>(options: OwnerOptions<T>): RequestOwner;
  run<T>(options: OneShotOptions<T>): Promise<T>;
  wake(): void;
  configure(parallel: number): void;
  /** Network and compute capacity. Sorting has an independent grant. */
  capacity(): number;
  sortCapacity(): number;
  activeOperations(): number;
  activeSortOperations(): number;
  /** Prefer a decode or a member with no fetch in flight for the last background grant. */
  favorIndependentFetches(): () => void;
  queuedOperations(): number;
  canAcquireWorker(kind: WorkerKind, reclaim?: boolean): boolean;
  workerCapacityPinned(kind: WorkerKind): boolean;
  acquireWorker(
    kind: WorkerKind,
    reclaimable?: IdleWorker,
    reclaim?: boolean,
  ): (() => void) | undefined;
};

/** One start authority for all owners on the page. Executors report readiness and settlement. */
export const createRequestAdmission = (parallel = 2): RequestAdmission => {
  let limit = integerAtLeast("parallel", parallel, 1);
  const sortLimit = 1;
  let background = 0;
  let networking = 0;
  const networkByGroup = new WeakMap<object, number>();
  let sorting = 0;
  let independentFetchUsers = 0;
  const owners = new Set<PendingOwner>();
  const sharedScopes = new WeakMap<object, Map<string, unknown>>();
  const workers = {
    copc: new Set<{ reclaimable?: IdleWorker }>(),
    decode: new Set<{ reclaimable?: IdleWorker }>(),
    sort: new Set<{ reclaimable?: IdleWorker }>(),
  };
  const backgroundLanes = ["compute", "network"] as const;
  const computeLane = ["compute"] as const;
  const networkLane = ["network"] as const;
  const sortLane = ["sort"] as const;
  const lastBackgroundGrant = new WeakMap<object, number>();
  const lastSortGrant = new WeakMap<object, number>();
  let serial = 0;
  let pumping = false;
  let wakeQueued = false;

  const charge = (lane: RequestLane, group: object, delta: number): void => {
    if (lane === "sort") sorting += delta;
    else background += delta;
    if (lane === "network") {
      networking += delta;
      networkByGroup.set(group, (networkByGroup.get(group) ?? 0) + delta);
    }
  };

  const offer = (
    lanes: readonly RequestLane[],
    served: WeakMap<object, number>,
    newNetworkGroupOnly = false,
  ): Grant | undefined => {
    let next: Grant | undefined;
    let nextAge = Infinity;
    // A ready decode can free buffered inputs. For equally ranked work in
    // the same group, finish it before starting another fetch.
    for (const lane of lanes)
      for (const owner of owners) {
        if (
          newNetworkGroupOnly &&
          (!owner.group || networkByGroup.get(owner.group))
        )
          continue;
        const candidate = owner.offer(lane);
        if (!candidate) continue;
        const age = served.get(candidate.group) ?? 0;
        if (
          !next ||
          age < nextAge ||
          (age === nextAge && candidate.priority < next.priority)
        ) {
          next = candidate;
          nextAge = age;
        }
      }
    return next;
  };

  const offerBackground = (): Grant | undefined => {
    if (independentFetchUsers === 0 || limit === 1 || networking < limit - 1)
      return offer(backgroundLanes, lastBackgroundGrant);
    // A ready decode can free buffered input without waiting for another
    // response. Keep the last grant open to independent members so several
    // slow requests from one tileset cannot block another member's first tile.
    return (
      offer(computeLane, lastBackgroundGrant) ??
      offer(networkLane, lastBackgroundGrant, true)
    );
  };

  const pump = (): void => {
    if (pumping) {
      wake();
      return;
    }
    pumping = true;
    try {
      while (true) {
        const sort =
          sorting < sortLimit ? offer(sortLane, lastSortGrant) : undefined;
        const next =
          sort ?? (background < limit ? offerBackground() : undefined);
        if (!next) break;
        (sort ? lastSortGrant : lastBackgroundGrant).set(next.group, ++serial);
        next.start();
      }
    } finally {
      pumping = false;
    }
  };

  const wake = (): void => {
    if (wakeQueued) return;
    wakeQueued = true;
    queueMicrotask(() => {
      wakeQueued = false;
      pump();
    });
  };

  const idleWorker = (kind: WorkerKind): IdleWorker | undefined => {
    for (const { reclaimable } of workers[kind])
      if (reclaimable?.idle()) return reclaimable;
    return undefined;
  };

  const createOwner = <T>(options: OwnerOptions<T>): RequestOwner => {
    const clock = options.clock ?? systemClock;
    const group = options.group ?? {};
    const priority = options.priority ?? 0;
    const grace = options.cancelGraceMs ?? 0;
    let concurrency = options.concurrency;
    let disposed = false;
    let records = new Map<string, Operation<T>>();
    const running = new Map<string, Operation<T>>();
    const failures = new Map<string, FailureRecord>();
    const onChange = (): void => safeCall(() => options.onChange?.());
    const resting = (key: string): boolean =>
      !!options.retry &&
      isResting(failures.get(key), options.retry, clock.now());

    const detach = (op: Operation<T>): void => {
      const shared = op.shared;
      if (!shared) return;
      op.shared = undefined;
      shared.claims.delete(op);
      if (shared.claims.size > 0) return;
      shared.remove();
      if (shared.active) shared.controller.abort();
      else shared.stage?.release?.();
    };

    const remove = (op: Operation<T>): void => {
      if (records.get(op.key) === op) records.delete(op.key);
      op.clearTimer?.();
      op.clearTimer = undefined;
      if (op.shared) detach(op);
      else if (op.status === "queued") op.stage.release?.();
    };

    const retry = (op: Operation<T>, delay: number, rest: boolean): void => {
      if (!Number.isFinite(delay) || delay < 0)
        throw new RangeError("retry delay must be finite and non-negative");
      op.status = rest ? "resting" : "retrying";
      let cancelled = false;
      const handle = clock.setTimeout(() => {
        if (cancelled || records.get(op.key) !== op || !op.wanted) return;
        op.clearTimer = undefined;
        op.status = "queued";
        join(op);
        onChange();
        pump();
      }, delay);
      op.clearTimer = () => {
        cancelled = true;
        safeCall(() => clock.clearTimeout(handle));
      };
    };

    const refreshRegistration = (): void => {
      if (records.size || running.size) owners.add(owner);
      else owners.delete(owner);
    };

    const fail = (op: Operation<T>, error: unknown): void => {
      const { key } = op;
      const refusal = resourcePolicyError(error);
      if (!op.wanted) {
        // Keep the point source's cooldown even when a camera move removes demand.
        if (options.retry && !refusal && !isAbortError(error)) {
          failures.set(
            key,
            recordFailure(failures.get(key), options.retry, clock.now()),
          );
          safeCall(() => options.onFailed(key, error));
        }
        remove(op);
        return;
      }
      op.stage = options.operation(key);
      op.firstStage = true;
      if (refusal) {
        op.status = "blocked";
        op.refusal = refusal;
        op.availability = options.availability?.();
        if (options.onBlocked)
          safeCall(() => options.onBlocked?.(key, refusal));
        else if (!refusal.retryable || op.availability === undefined)
          safeCall(() => options.onFailed(key, refusal));
        return;
      }
      if (isAbortError(error) && op.controller.signal.aborted) {
        records.set(key, {
          ...op,
          controller: new AbortController(),
          status: "queued",
          attempt: Math.max(0, op.attempt - 1),
        });
        return;
      }
      try {
        if (options.retry) {
          const failure = recordFailure(
            failures.get(key),
            options.retry,
            clock.now(),
          );
          failures.set(key, failure);
          retry(
            op,
            retryDelayMs(failure, options.retry, clock.now()),
            resting(key),
          );
          safeCall(() => options.onFailed(key, error));
          return;
        }
        if (
          op.attempt < (options.retryLimit ?? 1) &&
          (options.shouldRetry?.(error) ?? true)
        ) {
          retry(
            op,
            (options.retryBackoffMs ?? ((attempt) => 250 * 2 ** (attempt - 1)))(
              op.attempt,
            ),
            false,
          );
          return;
        }
      } catch (cause) {
        error = options.onRetryError?.(key, cause) ?? cause;
      }
      op.status = "failed";
      safeCall(() => options.onFailed(key, error));
    };

    const settle = (op: Operation<T>, outcome: OperationOutcome<T>): void => {
      op.clearGrace?.();
      op.clearGrace = undefined;
      const current = records.get(op.key) === op && !disposed;
      if ("result" in outcome) {
        const { result } = outcome;
        if ("next" in result) {
          if (current && op.wanted && !op.controller.signal.aborted) {
            op.stage = result.next;
            op.firstStage = false;
            op.status = "queued";
          } else {
            result.next.release?.();
            if (current && op.wanted) fail(op, op.controller.signal.reason);
            else if (current) remove(op);
          }
        } else if (current && (op.wanted || options.keepUnwantedResults)) {
          records.delete(op.key);
          failures.delete(op.key);
          safeCall(() => options.onLoaded(op.key, result.value, op.attempt));
        } else {
          if (current) records.delete(op.key);
          options.onDiscarded?.(result.value);
        }
      } else if (current) fail(op, outcome.error);
      refreshRegistration();
      onChange();
    };

    // Only a claimant that still needs physical work can make an offer.
    const join = (op: Operation<T>): boolean => {
      const share = op.stage.share;
      if (!share || op.shared) return false;
      let scope = sharedScopes.get(share.scope);
      if (!scope) {
        scope = new Map();
        sharedScopes.set(share.scope, scope);
      }
      // Each executor owns its scope and the result type of keys in it.
      const cached = scope.get(share.key) as
        | SharedOperation<T>
        | SharedResult<T>
        | undefined;
      let shared: SharedOperation<T> | undefined;
      if (cached && "reuse" in cached) {
        const value = cached.reuse();
        if (value !== undefined) {
          op.status = "active";
          queueMicrotask(() => settle(op, { result: { value } }));
          return true;
        }
      } else shared = cached;
      if (!shared) {
        const group = scope;
        shared = {
          controller: new AbortController(),
          claims: new Map(),
          remove: () => group.delete(share.key),
          publish(value) {
            const reuse = share.reuse(value, () => {
              if (group.get(share.key) === result) group.delete(share.key);
            });
            if (!reuse) {
              group.delete(share.key);
              return undefined;
            }
            const result = { reuse };
            group.set(share.key, result);
            return reuse;
          },
          started: false,
          active: false,
        };
        scope.set(share.key, shared);
      }
      if (op.status === "blocked") op.attempt = 0;
      op.shared = shared;
      op.status = shared.active ? "active" : "queued";
      if (shared.started) op.attempt += 1;
      shared.claims.set(op, {
        operation: op,
        settle: (outcome) => settle(op, outcome),
        change: onChange,
      });
      return shared.active;
    };

    const settleShared = (
      shared: SharedOperation<T>,
      outcome: OperationOutcome<T>,
    ): void => {
      shared.active = false;
      const claims = [...shared.claims.values()];
      const result = "result" in outcome ? outcome.result : undefined;
      if (result && "next" in result) {
        if (claims.length === 0) result.next.release?.();
        else {
          shared.stage = result.next;
          for (const { operation } of claims) operation.status = "queued";
          for (const { change } of claims) change();
        }
        return;
      }
      shared.claims.clear();
      shared.stage = undefined;
      for (const { operation } of claims) operation.shared = undefined;
      if (!result) {
        if (claims.length > 0) shared.remove();
        for (const claim of claims) claim.settle(outcome);
      } else {
        const { value } = result;
        const reuse = claims.length > 0 ? shared.publish(value) : undefined;
        try {
          for (const claim of claims)
            claim.settle({ result: { value: reuse?.() ?? value } });
        } finally {
          options.onDiscarded?.(value);
        }
      }
    };

    const finish = (
      op: Operation<T>,
      stage: RequestStage<T>,
      shared: SharedOperation<T> | undefined,
      outcome: OperationOutcome<T>,
    ): void => {
      running.delete(op.key);
      charge(stage.lane, group, -1);
      stage.release?.();
      if (shared) {
        const executorStillClaimed = shared.claims.has(op);
        settleShared(shared, outcome);
        refreshRegistration();
        if (!executorStillClaimed) onChange();
      } else settle(op, outcome);
      pump();
    };

    const start = (op: Operation<T>): void => {
      const shared = op.shared;
      const stage = shared?.stage ?? op.stage;
      const signal = (shared?.controller ?? op.controller).signal;
      running.set(op.key, op);
      charge(stage.lane, group, 1);
      if (shared) {
        shared.active = true;
        const claims = [...shared.claims.values()];
        for (const { operation } of claims) {
          operation.status = "active";
          if (!shared.started) operation.attempt += 1;
        }
        shared.started = true;
        for (const { change } of claims) change();
      } else {
        if (op.status === "blocked") op.attempt = 0;
        if (op.firstStage) op.attempt += 1;
        op.status = "active";
        onChange();
      }
      try {
        signal.throwIfAborted();
        Promise.resolve(stage.start(signal)).then(
          (result) => finish(op, stage, shared, { result }),
          (error: unknown) => finish(op, stage, shared, { error }),
        );
      } catch (error) {
        finish(op, stage, shared, { error });
      }
    };

    const owner: PendingOwner = {
      group: options.group,
      offer(lane) {
        if (running.size >= concurrency) return undefined;
        for (const op of records.values()) {
          if (
            !op.wanted ||
            running.has(op.key) ||
            (op.status !== "queued" &&
              !(
                op.status === "blocked" &&
                op.refusal?.retryable &&
                op.availability !== undefined &&
                op.availability !== options.availability?.() &&
                options.canRetry?.(op.refusal) !== false
              ))
          )
            continue;
          if (join(op)) continue;
          const stage = op.shared?.stage ?? op.stage;
          if (stage.lane !== lane || stage.ready?.() === false) continue;
          return { group, priority, start: () => start(op) };
        }
        return undefined;
      },
      queued() {
        let count = 0;
        for (const op of records.values())
          if (
            op.wanted &&
            (op.status === "queued" ||
              (op.status === "blocked" && op.refusal?.retryable)) &&
            (!op.shared || op.shared.claims.keys().next().value === op)
          )
            count++;
        return count;
      },
    };

    const withdraw = (op: Operation<T>, immediate: boolean): void => {
      op.wanted = false;
      if (op.status !== "active") {
        remove(op);
      } else if (immediate || grace === 0) {
        op.clearGrace?.();
        op.clearGrace = undefined;
        if (op.shared) remove(op);
        else {
          op.controller.abort();
          if (!options.adoptCancelled) records.delete(op.key);
        }
      } else if (!op.clearGrace && grace !== Infinity) {
        const timer = setTimeout(() => {
          op.clearGrace = undefined;
          if (!op.wanted) {
            if (op.shared) remove(op);
            else {
              op.controller.abort();
              if (!options.adoptCancelled && records.get(op.key) === op)
                records.delete(op.key);
            }
          }
        }, grace);
        op.clearGrace = () => clearTimeout(timer);
      }
    };

    const setDemand = (keys: readonly string[], immediate: boolean): void => {
      if (disposed) return;
      const wanted = new Set(keys);
      for (const op of records.values())
        if (!wanted.has(op.key)) withdraw(op, immediate);
      const ordered = new Map<string, Operation<T>>();
      for (const key of wanted) {
        let op = records.get(key);
        if (op) {
          op.clearGrace?.();
          op.clearGrace = undefined;
          op.wanted = true;
        } else {
          op = {
            key,
            controller: new AbortController(),
            stage: options.operation(key),
            status: "queued",
            attempt: failures.get(key)?.count ?? 0,
            wanted: true,
            firstStage: true,
          };
        }
        ordered.set(key, op);
      }
      for (const [key, op] of records) if (!op.wanted) ordered.set(key, op);
      records = ordered;
      for (const op of records.values()) {
        const failure = failures.get(op.key);
        if (!op.wanted || op.status !== "queued") continue;
        if (failure && options.retry) {
          const now = clock.now();
          const rest = resting(op.key);
          const delay =
            retryDelayMs(failure, options.retry, now) -
            (rest ? 0 : now - failure.lastMs);
          if (delay > 0) {
            retry(op, delay, rest);
            continue;
          }
        }
        join(op);
      }
      refreshRegistration();
      pump();
    };

    const reset = (): void => {
      for (const op of records.values()) withdraw(op, true);
      records.clear();
      failures.clear();
      refreshRegistration();
    };

    return {
      want: (keys) => setDemand(keys, false),
      abandon: () => setDemand([], true),
      reset,
      dispose() {
        disposed = true;
        reset();
      },
      setConcurrency(limit) {
        concurrency = limit;
        pump();
      },
      retryBlocked() {
        for (const op of records.values())
          if (op.status === "blocked") {
            // Temporary contention wakes from the availability revision that refused it.
            if (
              op.refusal?.retryable &&
              op.availability !== undefined &&
              options.canRetry?.(op.refusal) === false
            )
              continue;
            op.status = "queued";
            op.attempt = 0;
            join(op);
          }
        pump();
      },
      state(key) {
        const op = records.get(key);
        return op && { status: op.status, attempt: op.attempt };
      },
      physicalCount: () => running.size,
      resting,
      restingCount: () => [...failures.keys()].filter(resting).length,
      counts() {
        let reading = 0;
        let wantedReading = 0;
        let queued = 0;
        let retrying = 0;
        for (const op of records.values()) {
          if (op.status === "active") {
            reading += 1;
            if (op.wanted) wantedReading += 1;
          } else if (op.wanted) {
            if (op.status === "queued") queued += 1;
            if (op.status === "retrying") retrying += 1;
          }
        }
        return {
          queued,
          physical: running.size,
          retrying,
          reading,
          wantedReading,
        };
      },
    };
  };

  return {
    createOwner,
    run<T>(options: OneShotOptions<T>): Promise<T> {
      const { signal, stage, onDiscarded } = options;
      signal.throwIfAborted();
      return new Promise((resolve, reject) => {
        const end = (): void => {
          signal.removeEventListener("abort", abort);
          owner.dispose();
        };
        const abort = (): void => {
          end();
          reject(signal.reason);
        };
        const owner = createOwner({
          concurrency: 1,
          group: options.group,
          availability: options.availability,
          canRetry: options.canRetry,
          operation: () => stage,
          onDiscarded,
          onLoaded(_key, value) {
            end();
            resolve(value);
          },
          onFailed(_key, error) {
            end();
            reject(error);
          },
        });
        signal.addEventListener("abort", abort, { once: true });
        owner.want([""]);
      });
    },
    wake,
    configure(parallel) {
      limit = integerAtLeast("parallel", parallel, 1);
      wake();
    },
    capacity: () => limit,
    sortCapacity: () => sortLimit,
    activeOperations: () => background + sorting,
    activeSortOperations: () => sorting,
    favorIndependentFetches() {
      independentFetchUsers += 1;
      let held = true;
      return () => {
        if (!held) return;
        held = false;
        independentFetchUsers -= 1;
        wake();
      };
    },
    queuedOperations: () =>
      [...owners].reduce((sum, owner) => sum + owner.queued(), 0),
    canAcquireWorker: (kind, reclaim = true) =>
      workers[kind].size < persistentLimits[kind] ||
      (reclaim && !!idleWorker(kind)),
    workerCapacityPinned: (kind) =>
      workers[kind].size >= persistentLimits[kind] &&
      [...workers[kind]].every(({ reclaimable }) => !reclaimable),
    acquireWorker(kind, reclaimable, reclaim = true) {
      const owned = workers[kind];
      if (reclaim && owned.size >= persistentLimits[kind])
        idleWorker(kind)?.retire();
      if (owned.size >= persistentLimits[kind]) return undefined;
      const entry = { reclaimable };
      owned.add(entry);
      return () => {
        if (owned.delete(entry)) wake();
      };
    },
  };
};

let pageAdmission: RequestAdmission | undefined;
export const getPageRequestAdmission = (): RequestAdmission =>
  (pageAdmission ??= createRequestAdmission());
