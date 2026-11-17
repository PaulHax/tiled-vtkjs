/** Renderer-neutral tile content selection and decoded retention. */

import { finiteAtLeast, integerAtLeast } from "../core/numeric";
import { safeCall } from "../core/observers";
import { TileDecodeError } from "./decode/types";
import {
  createRequestAdmission,
  type RequestClock,
  type RequestStage,
} from "../core/requestAdmission";
import {
  resourcePolicyError,
  type MemoryOwner,
  type MemoryLease,
  type MemoryWork,
  type ResourcePolicyError,
} from "../core/memoryResources";
import {
  readMemoryResponse,
  type BufferedResponse,
} from "../core/memoryResponse";

const MAX_EVICTION_HISTORY = 4_096;

export type TileContentRequest = {
  readonly id: string;
  readonly url: string;
};

export type ContentQueueFetchResponse = BufferedResponse & {
  readonly ok: boolean;
  readonly status: number;
  readonly statusText: string;
  arrayBuffer(): Promise<ArrayBuffer>;
};

export type ContentQueueFetch = (
  url: string,
  init: { readonly signal: AbortSignal },
) => Promise<ContentQueueFetchResponse>;

export type ContentDecodeContext = {
  readonly signal: AbortSignal;
  readonly revision: string;
  readonly configGeneration: number;
  readonly work?: MemoryWork;
};

export type ContentQueueClock = RequestClock;

export type ContentQueueEntryStatus =
  | "queued"
  | "fetching"
  | "retrying"
  | "ready"
  | "blocked"
  | "failed";

export type ContentQueueEntrySnapshot = TileContentRequest & {
  readonly status: ContentQueueEntryStatus;
  readonly attempt: number;
};

export type ContentQueueSnapshot = {
  readonly revision: string;
  readonly configGeneration: number;
  readonly maxConcurrency: number;
  readonly maxDecodedBytes: number;
  readonly selected: number;
  readonly active: number;
  readonly queued: number;
  readonly retrying: number;
  readonly ready: number;
  readonly failed: number;
  readonly blocked: number;
  readonly cached: number;
  readonly decodedBytes: number;
  readonly cacheHits: number;
  readonly cacheMisses: number;
  readonly cacheEvictions: number;
  readonly cacheRevisits: number;
  readonly workPending: boolean;
  readonly disposed: boolean;
  readonly entries: readonly ContentQueueEntrySnapshot[];
};

export type ContentQueueOptions<T> = {
  readonly revision: string;
  readonly configGeneration: number;
  readonly maxConcurrency: number;
  readonly maxDecodedBytes: number;
  readonly fetch?: ContentQueueFetch;
  readonly memory?: MemoryOwner;
  readonly decodeReady?: () => boolean;
  readonly priority?: number;
  readonly decodedBuffers?: (content: T) => Iterable<ArrayBuffer>;
  readonly decode: (
    bytes: ArrayBuffer,
    request: TileContentRequest,
    context: ContentDecodeContext,
  ) => Promise<T> | T;
  readonly decodedByteLength: (
    content: T,
    request: TileContentRequest,
  ) => number;
  readonly maxAttempts?: number;
  readonly retryBackoffMs?: (failedAttempt: number) => number;
  readonly clock?: ContentQueueClock;
  readonly onContent?: (request: TileContentRequest, content: T) => void;
  readonly onError?: (request: TileContentRequest, error: Error) => void;
  readonly onPolicyRefusal?: (
    request: TileContentRequest,
    error: ResourcePolicyError,
  ) => void;
  readonly onEvict?: (request: TileContentRequest, content: T) => void;
  readonly onStateChange?: (snapshot: ContentQueueSnapshot) => void;
};

export type ContentQueueConfiguration = {
  readonly revision?: string;
  readonly configGeneration?: number;
  readonly maxConcurrency?: number;
  readonly maxDecodedBytes?: number;
};

export type ContentQueue<T> = {
  setSelection(
    requests: readonly TileContentRequest[],
    options?: { readonly requeueUncached?: ReadonlySet<string> },
  ): void;
  configure(configuration: ContentQueueConfiguration): void;
  /** Look up decoded content and mark it most recently used. */
  get(id: string): T | undefined;
  /** Every cached decode, in least-recently-used order. */
  contents(): ReadonlyMap<string, T>;
  snapshot(): ContentQueueSnapshot;
  trimCache(bytes: number): void;
  retryBlocked(availabilityOnly?: boolean): void;
  dispose(): void;
};

export class ContentQueueError extends Error {
  readonly id: string;
  readonly url: string;

  constructor(
    name: string,
    request: TileContentRequest,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${request.id}: ${message}`, options);
    this.name = name;
    this.id = request.id;
    this.url = request.url;
  }
}

export class ContentQueueFetchError extends ContentQueueError {
  readonly status?: number;

  constructor(
    request: TileContentRequest,
    message: string,
    status?: number,
    options?: ErrorOptions,
  ) {
    super("ContentQueueFetchError", request, message, options);
    this.status = status;
  }
}

export class ContentQueueDecodeError extends ContentQueueError {
  constructor(request: TileContentRequest, options?: ErrorOptions) {
    super("ContentQueueDecodeError", request, "content decode failed", options);
  }
}

export class ContentQueueRetryError extends ContentQueueError {
  constructor(
    request: TileContentRequest,
    message: string,
    options?: ErrorOptions,
  ) {
    super("ContentQueueRetryError", request, message, options);
  }
}

type SelectedEntry = {
  readonly request: TileContentRequest;
  readyAttempt?: number;
};

type CacheEntry<T> = {
  readonly request: TileContentRequest;
  readonly content: T;
  readonly bytes: number;
  readonly memory?: MemoryLease;
};

const normalizeRequest = (request: TileContentRequest): TileContentRequest => {
  if (typeof request.id !== "string" || request.id.length === 0) {
    throw new TypeError("content request id must be a non-empty string");
  }
  if (typeof request.url !== "string" || request.url.length === 0) {
    throw new TypeError(
      `content request ${request.id} URL must be a non-empty string`,
    );
  }
  return Object.freeze({ id: request.id, url: request.url });
};

export const createContentQueue = <T>(
  options: ContentQueueOptions<T>,
): ContentQueue<T> => {
  let revision = options.revision;
  let configGeneration = integerAtLeast(
    "configGeneration",
    options.configGeneration,
    0,
  );
  let maxConcurrency = integerAtLeast(
    "maxConcurrency",
    options.maxConcurrency,
    1,
  );
  let maxDecodedBytes = finiteAtLeast(
    "maxDecodedBytes",
    options.maxDecodedBytes,
    0,
  );
  const maxAttempts = integerAtLeast(
    "maxAttempts",
    options.maxAttempts ?? 3,
    1,
  );
  const retryBackoffMs =
    options.retryBackoffMs ?? ((attempt) => 250 * 2 ** (attempt - 1));
  const fetcher =
    options.fetch ?? (globalThis.fetch as unknown as ContentQueueFetch);
  if (typeof revision !== "string" || revision.length === 0) {
    throw new TypeError("revision must be a non-empty string");
  }
  if (typeof fetcher !== "function") {
    throw new TypeError("fetch is unavailable");
  }

  let disposed = false;
  let selected = new Map<string, SelectedEntry>();
  const cache = new Map<string, CacheEntry<T>>();
  let decodedBytes = 0;
  let cacheHits = 0;
  let cacheMisses = 0;
  let cacheEvictions = 0;
  let cacheRevisits = 0;
  const previouslyEvicted = new Set<string>();

  const snapshot = (): ContentQueueSnapshot => {
    const entries: ContentQueueEntrySnapshot[] = [];
    const counts: Record<ContentQueueEntryStatus, number> = {
      queued: 0,
      retrying: 0,
      fetching: 0,
      ready: 0,
      failed: 0,
      blocked: 0,
    };
    for (const { request, readyAttempt } of selected.values()) {
      const state = owner.state(request.id);
      const status =
        readyAttempt !== undefined
          ? "ready"
          : state?.status === "active"
            ? "fetching"
            : state?.status === "resting"
              ? "retrying"
              : (state?.status ?? "queued");
      const attempt = readyAttempt ?? state?.attempt ?? 0;
      entries.push(Object.freeze({ ...request, status, attempt }));
      counts[status] += 1;
    }
    const { queued, retrying, fetching, ready, failed, blocked } = counts;
    return Object.freeze({
      revision,
      configGeneration,
      maxConcurrency,
      maxDecodedBytes,
      selected: selected.size,
      active: owner.physicalCount(),
      queued,
      retrying,
      ready,
      failed,
      blocked,
      cached: cache.size,
      decodedBytes,
      cacheHits,
      cacheMisses,
      cacheEvictions,
      cacheRevisits,
      workPending: queued + retrying + fetching > 0,
      disposed,
      entries: Object.freeze(entries),
    });
  };

  // Deliberately lazy: the snapshot is only built when someone is listening.
  const emit = (): void => {
    const onStateChange = options.onStateChange;
    if (onStateChange) safeCall(() => onStateChange(snapshot()));
  };

  const evict = (id: string): boolean => {
    const entry = cache.get(id);
    if (!entry) return false;
    cache.delete(id);
    entry.memory?.release();
    decodedBytes -= entry.bytes;
    cacheEvictions += 1;
    previouslyEvicted.add(id);
    while (previouslyEvicted.size > MAX_EVICTION_HISTORY) {
      const oldest = previouslyEvicted.values().next();
      if (oldest.done) break;
      previouslyEvicted.delete(oldest.value);
    }
    safeCall(() => options.onEvict?.(entry.request, entry.content));
    return true;
  };

  const enforceCacheBudget = (): void => {
    while (decodedBytes > maxDecodedBytes) {
      const oldest = cache.keys().next();
      if (oldest.done === true) break;
      evict(oldest.value);
    }
  };

  const putCache = (
    request: TileContentRequest,
    content: T,
    bytes: number,
    memory?: MemoryLease,
  ): void => {
    evict(request.id);
    // Oversize content is still delivered to the submission consumer, but is
    // never transiently charged to the CPU cache. Existing LRU entries remain
    // intact and the uncached value never produces an eviction notification.
    if (bytes > maxDecodedBytes) return;
    cache.set(request.id, { request, content, bytes, memory });
    decodedBytes += bytes;
    enforceCacheBudget();
  };

  const clearCache = (): void => {
    for (const id of cache.keys()) evict(id);
  };

  type Output = {
    readonly content: T;
    readonly bytes: number;
    readonly memory?: MemoryLease;
  };
  const admission =
    options.memory?.admission ?? createRequestAdmission(maxConcurrency);
  const owner = admission.createOwner<Output>({
    group: options.memory,
    priority: options.priority,
    availability: options.memory?.availability,
    canRetry: options.memory?.canRetry,
    concurrency: maxConcurrency,
    retryLimit: maxAttempts,
    retryBackoffMs,
    clock: options.clock,
    shouldRetry: (error) =>
      !(
        error instanceof ContentQueueFetchError &&
        error.status !== undefined &&
        error.status >= 400 &&
        error.status < 500 &&
        ![408, 429].includes(error.status)
      ) &&
      !(
        error instanceof ContentQueueDecodeError &&
        error.cause instanceof TileDecodeError &&
        error.cause.stage === "profile"
      ),
    onRetryError: (id, cause) =>
      new ContentQueueRetryError(
        selected.get(id)!.request,
        "retry policy or timer scheduling failed",
        { cause },
      ),
    operation(id): RequestStage<Output> {
      const { request } = selected.get(id)!;
      const operationRevision = revision;
      const operationGeneration = configGeneration;
      return {
        lane: "network",
        async start(signal) {
          const work = options.memory?.work(signal);
          let handedOff = false;
          try {
            let response: ContentQueueFetchResponse;
            try {
              response = await fetcher(request.url, { signal });
            } catch (cause) {
              throw new ContentQueueFetchError(
                request,
                "network request failed",
                undefined,
                { cause },
              );
            }
            if (!response.ok) {
              if (response.body) await response.body.cancel().catch(() => {});
              throw new ContentQueueFetchError(
                request,
                `HTTP ${response.status} ${response.statusText}`.trim(),
                response.status,
              );
            }
            let bytes: ArrayBuffer;
            try {
              bytes = work
                ? await readMemoryResponse(response, work, signal)
                : await response.arrayBuffer();
              signal.throwIfAborted();
              work?.resize(bytes.byteLength);
            } catch (cause) {
              throw new ContentQueueFetchError(
                request,
                "response body read failed",
                response.status,
                { cause },
              );
            }
            handedOff = true;
            return {
              next: {
                lane: "compute",
                ready: options.decodeReady,
                release: () => work?.release(),
                async start(signal) {
                  try {
                    const content = await options.decode(bytes, request, {
                      signal,
                      revision: operationRevision,
                      configGeneration: operationGeneration,
                      work,
                    });
                    const byteLength = options.decodedByteLength(
                      content,
                      request,
                    );
                    finiteAtLeast(
                      `decoded bytes for ${request.id}`,
                      byteLength,
                      0,
                    );
                    const buffers = options.decodedBuffers?.(content) ?? [];
                    return {
                      value: {
                        content,
                        bytes: byteLength,
                        memory: work?.commit(
                          buffers,
                          options.decodedBuffers ? 0 : byteLength,
                        ),
                      },
                    };
                  } catch (cause) {
                    throw new ContentQueueDecodeError(request, { cause });
                  }
                },
              },
            };
          } finally {
            if (!handedOff) work?.release();
          }
        },
      };
    },
    onLoaded(id, output, attempt) {
      const entry = selected.get(id)!;
      try {
        putCache(entry.request, output.content, output.bytes, output.memory);
        entry.readyAttempt = attempt;
        safeCall(() => options.onContent?.(entry.request, output.content));
      } finally {
        if (cache.get(id)?.memory !== output.memory) output.memory?.release();
      }
    },
    onDiscarded: (output) => output.memory?.release(),
    onFailed(id, error) {
      const item = selected.get(id)!.request;
      const refusal = resourcePolicyError(error);
      if (refusal) options.onPolicyRefusal?.(item, refusal);
      else
        options.onError?.(
          item,
          error instanceof Error
            ? error
            : new ContentQueueError(
                "ContentQueueError",
                item,
                "operation failed with a non-Error value",
              ),
        );
    },
    onChange: emit,
  });

  const requestMissing = (): void =>
    owner.want(
      [...selected.values()]
        .filter((entry) => entry.readyAttempt === undefined)
        .map(({ request }) => request.id),
    );

  return {
    trimCache(bytes) {
      for (const id of cache.keys()) {
        if (decodedBytes <= bytes) break;
        evict(id);
      }
      emit();
    },
    retryBlocked(availabilityOnly = false) {
      if (disposed) return;
      owner.retryBlocked(availabilityOnly);
      emit();
    },
    setSelection(requests, selectionOptions) {
      if (disposed) throw new Error("content queue is disposed");
      const normalized = new Map<string, TileContentRequest>();
      for (const raw of requests) {
        const item = normalizeRequest(raw);
        const duplicate = normalized.get(item.id);
        if (duplicate && duplicate.url !== item.url) {
          throw new Error(`conflicting content requests for ${item.id}`);
        }
        if (!duplicate) normalized.set(item.id, item);
      }

      // Validate retained IDs as a complete phase. No abort, timer cancel, or
      // cache mutation occurs unless the whole requested selection is valid.
      for (const [id, entry] of selected) {
        const next = normalized.get(id);
        if (next && next.url !== entry.request.url) {
          throw new Error(`conflicting content request URL for ${id}`);
        }
      }
      const nextSelection = new Map<string, SelectedEntry>();
      const hits: CacheEntry<T>[] = [];
      for (const item of normalized.values()) {
        const retained = selected.get(item.id);
        // Delivery is distinct from cache residency. A consumer that needs
        // an evicted payload again can explicitly reopen only that request,
        // without cancelling unrelated work or retrying cached/active jobs.
        if (
          retained &&
          !(
            retained.readyAttempt !== undefined &&
            !cache.has(item.id) &&
            selectionOptions?.requeueUncached?.has(item.id)
          )
        ) {
          nextSelection.set(item.id, retained);
          continue;
        }
        let cached = cache.get(item.id);
        if (cached && cached.request.url !== item.url) {
          evict(item.id);
          cached = undefined;
        }
        const entry: SelectedEntry = {
          request: item,
          readyAttempt: cached ? 0 : undefined,
        };
        nextSelection.set(item.id, entry);
        if (cached) {
          cacheHits += 1;
          cache.delete(item.id);
          cache.set(item.id, cached);
          hits.push(cached);
        } else {
          cacheMisses += 1;
          if (previouslyEvicted.delete(item.id)) cacheRevisits += 1;
        }
      }
      selected = nextSelection;
      requestMissing();
      for (const hit of hits)
        if (selected.has(hit.request.id) && cache.get(hit.request.id) === hit)
          safeCall(() => options.onContent?.(hit.request, hit.content));
      emit();
    },

    configure(configuration) {
      if (disposed) throw new Error("content queue is disposed");
      const nextRevision = configuration.revision ?? revision;
      const nextGeneration =
        configuration.configGeneration === undefined
          ? configGeneration
          : integerAtLeast(
              "configGeneration",
              configuration.configGeneration,
              0,
            );
      if (typeof nextRevision !== "string" || nextRevision.length === 0) {
        throw new TypeError("revision must be a non-empty string");
      }
      if (configuration.maxConcurrency !== undefined) {
        maxConcurrency = integerAtLeast(
          "maxConcurrency",
          configuration.maxConcurrency,
          1,
        );
      }
      if (configuration.maxDecodedBytes !== undefined) {
        maxDecodedBytes = finiteAtLeast(
          "maxDecodedBytes",
          configuration.maxDecodedBytes,
          0,
        );
      }

      const changedGeneration =
        nextRevision !== revision || nextGeneration !== configGeneration;
      if (changedGeneration) {
        owner.reset();
        clearCache();
        previouslyEvicted.clear();
        revision = nextRevision;
        configGeneration = nextGeneration;
        for (const entry of selected.values()) entry.readyAttempt = undefined;
      } else {
        enforceCacheBudget();
      }
      if (!options.memory) admission.configure(maxConcurrency);
      owner.setConcurrency(maxConcurrency);
      if (changedGeneration) requestMissing();
      emit();
    },

    get(id) {
      const entry = cache.get(id);
      if (!entry) return undefined;
      cache.delete(id);
      cache.set(id, entry);
      return entry.content;
    },

    contents: () =>
      new Map([...cache].map(([id, entry]) => [id, entry.content] as const)),

    snapshot,

    dispose() {
      if (disposed) return;
      disposed = true;
      owner.dispose();
      selected.clear();
      clearCache();
      previouslyEvicted.clear();
      emit();
    },
  };
};
