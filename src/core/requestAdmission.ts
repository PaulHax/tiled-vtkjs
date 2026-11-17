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

export type RequestStage<T> = {
  readonly lane: RequestLane;
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
};

const systemClock: RequestClock = {
  now: () => Date.now(),
  setTimeout: (callback, delayMs) => setTimeout(callback, delayMs),
  clearTimeout: (handle) =>
    clearTimeout(handle as ReturnType<typeof setTimeout>),
};
const isAbortError = (error: unknown): boolean =>
  error instanceof Error && error.name === "AbortError";

const limitsFor = (parallel: number): Record<RequestLane, number> => ({
  network: parallel,
  compute: parallel,
  sort: 1,
});

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
  let limits = limitsFor(integerAtLeast("parallel", parallel, 1));
  const active: Record<RequestLane, number> = {
    network: 0,
    compute: 0,
    sort: 0,
  };
  const owners = new Set<PendingOwner>();
  const workers = {
    copc: new Set<{ reclaimable?: IdleWorker }>(),
    decode: new Set<{ reclaimable?: IdleWorker }>(),
    sort: new Set<{ reclaimable?: IdleWorker }>(),
  };
  const lastGrant = {
    network: new WeakMap<object, number>(),
    compute: new WeakMap<object, number>(),
    sort: new WeakMap<object, number>(),
  };
  let serial = 0;
  let pumping = false;
  let wakeQueued = false;

  const available = (lane: RequestLane): boolean => active[lane] < limits[lane];
  const charge = (lane: RequestLane, delta: number): void => {
    active[lane] += delta;
  };

  const pump = (): void => {
    if (pumping) {
      wake();
      return;
    }
    pumping = true;
    try {
      for (const lane of ["sort", "compute", "network"] as const) {
        const served = lastGrant[lane];
        while (available(lane)) {
          let next: Grant | undefined;
          let nextAge = Infinity;
          for (const owner of owners) {
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
          if (!next) break;
          served.set(next.group, ++serial);
          next.start();
        }
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
    const ownerActive: Record<RequestLane, number> = {
      network: 0,
      compute: 0,
      sort: 0,
    };
    const failures = new Map<string, FailureRecord>();
    const onChange = (): void => safeCall(() => options.onChange?.());
    const resting = (key: string): boolean =>
      !!options.retry &&
      isResting(failures.get(key), options.retry, clock.now());

    const remove = (op: Operation<T>): void => {
      if (records.get(op.key) === op) records.delete(op.key);
      op.clearTimer?.();
      op.clearTimer = undefined;
      if (op.status === "queued") op.stage.release?.();
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

    const finish = (
      op: Operation<T>,
      outcome: { result: RequestResult<T> } | { error: unknown },
    ): void => {
      const lane = op.stage.lane;
      running.delete(op.key);
      ownerActive[lane] -= 1;
      charge(lane, -1);
      op.clearGrace?.();
      op.clearGrace = undefined;
      op.stage.release?.();
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
      pump();
    };

    const start = (op: Operation<T>): void => {
      const lane = op.stage.lane;
      if (op.status === "blocked") op.attempt = 0;
      if (op.firstStage) op.attempt += 1;
      op.status = "active";
      running.set(op.key, op);
      ownerActive[lane] += 1;
      charge(lane, 1);
      onChange();
      try {
        op.controller.signal.throwIfAborted();
        Promise.resolve(op.stage.start(op.controller.signal)).then(
          (result) => finish(op, { result }),
          (error: unknown) => finish(op, { error }),
        );
      } catch (error) {
        finish(op, { error });
      }
    };

    const owner: PendingOwner = {
      offer(lane) {
        if (ownerActive[lane] >= concurrency) return undefined;
        for (const op of records.values()) {
          if (
            op.stage.lane !== lane ||
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
          if (op.stage.ready?.() === false) continue;
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
              (op.status === "blocked" && op.refusal?.retryable))
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
        op.controller.abort();
        if (!options.adoptCancelled) records.delete(op.key);
      } else if (!op.clearGrace && grace !== Infinity) {
        const timer = setTimeout(() => {
          op.clearGrace = undefined;
          if (!op.wanted) {
            op.controller.abort();
            if (!options.adoptCancelled && records.get(op.key) === op)
              records.delete(op.key);
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
        if (!op.wanted || op.status !== "queued" || !failure || !options.retry)
          continue;
        const now = clock.now();
        const rest = resting(op.key);
        const delay =
          retryDelayMs(failure, options.retry, now) -
          (rest ? 0 : now - failure.lastMs);
        if (delay > 0) retry(op, delay, rest);
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
      limits = limitsFor(integerAtLeast("parallel", parallel, 1));
      wake();
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
