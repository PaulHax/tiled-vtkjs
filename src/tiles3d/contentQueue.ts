/** Renderer-neutral tile content selection and decoded retention. */

import { finiteAtLeast, integerAtLeast } from "../core/numeric";
import { safeCall } from "../core/observers";
import { TileDecodeError } from "./decode/types";
import {
  createRequestAdmission,
  type RequestClock,
  type RequestShare,
  type RequestStage,
} from "../core/requestAdmission";
import {
  reuseMemoryResult,
  type MemoryOwner,
  type MemoryLease,
  type MemoryWork,
  type ResourcePolicyError,
} from "../core/memoryResources";
import {
  readMemoryResponse,
  type BufferedResponse,
} from "../core/memoryResponse";

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

type ContentDecoder<T> = {
  readonly share?: Pick<RequestShare<Decoded<T>>, "scope" | "key">;
  readonly decode: (
    bytes: ArrayBuffer,
    context: ContentDecodeContext,
  ) => Promise<T> | T;
};

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
  readonly prepareDecode: (request: TileContentRequest) => ContentDecoder<T>;
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
  readonly onStateChange?: (snapshot: ContentQueueSnapshot) => void;
};

export type ContentQueueConfiguration = {
  readonly revision?: string;
  readonly configGeneration?: number;
  readonly maxConcurrency?: number;
  readonly maxDecodedBytes?: number;
};

export type ContentQueue<T> = {
  setSelection(requests: readonly TileContentRequest[]): void;
  state(id: string): ContentQueueEntrySnapshot | undefined;
  configure(configuration: ContentQueueConfiguration): void;
  /** Look up decoded content and mark it most recently used. */
  get(id: string): T | undefined;
  /** Live decodes held by current demand or by optional retention. */
  contents(): ReadonlyMap<string, T>;
  snapshot(): ContentQueueSnapshot;
  trimCache(bytes: number): void;
  retryBlocked(): void;
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

type Decoded<T> = {
  readonly content: T;
  readonly bytes: number;
  readonly memory?: MemoryLease;
};

type TileGeneration<T> = {
  readonly request: TileContentRequest;
  decoded?: Decoded<T> & { readonly attempt: number };
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
  let selected = new Map<string, TileGeneration<T>>();
  const cache = new Map<string, TileGeneration<T>>();
  let decodedBytes = 0;
  let cacheHits = 0;
  let cacheMisses = 0;
  let cacheEvictions = 0;
  const state = (id: string): ContentQueueEntrySnapshot | undefined => {
    const tile = selected.get(id);
    if (!tile) return undefined;
    const operation = owner.state(id);
    return {
      ...tile.request,
      status: tile.decoded
        ? "ready"
        : operation?.status === "active"
          ? "fetching"
          : operation?.status === "resting"
            ? "retrying"
            : (operation?.status ?? "queued"),
      attempt: tile.decoded?.attempt ?? operation?.attempt ?? 0,
    };
  };

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
    for (const id of selected.keys()) {
      const entry = state(id)!;
      entries.push(Object.freeze(entry));
      counts[entry.status] += 1;
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

  const releaseDecoded = (tile: TileGeneration<T>): void => {
    tile.decoded?.memory?.release();
    tile.decoded = undefined;
  };

  const evict = (id: string): void => {
    const tile = cache.get(id)!;
    const decoded = tile.decoded!;
    cache.delete(id);
    decodedBytes -= decoded.bytes;
    cacheEvictions += 1;
    if (selected.get(id) !== tile) releaseDecoded(tile);
  };

  const enforceCacheBudget = (budget = maxDecodedBytes): void => {
    for (const id of cache.keys()) {
      if (budget > 0 && decodedBytes <= budget) break;
      evict(id);
    }
  };

  const clearCache = (): void => {
    for (const id of cache.keys()) evict(id);
  };

  const admission =
    options.memory?.admission ?? createRequestAdmission(maxConcurrency);
  const owner = admission.createOwner<Decoded<T>>({
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
    operation(id): RequestStage<Decoded<T>> {
      const { request } = selected.get(id)!;
      const operationRevision = revision;
      const operationGeneration = configGeneration;
      const { decode, share } = options.prepareDecode(request);
      return {
        lane: "network",
        share: share && {
          ...share,
          scope: options.memory?.scopeForSharing(share.scope) ?? share.scope,
          reuse: reuseMemoryResult,
        },
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
                    const content = await decode(bytes, {
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
      const tile = selected.get(id)!;
      tile.decoded = { ...output, attempt };
      if (maxDecodedBytes > 0 && output.bytes <= maxDecodedBytes) {
        cache.set(id, tile);
        decodedBytes += output.bytes;
        enforceCacheBudget();
      }
      safeCall(() => options.onContent?.(tile.request, output.content));
    },
    onDiscarded: (output) => output.memory?.release(),
    onBlocked(id, refusal) {
      options.onPolicyRefusal?.(selected.get(id)!.request, refusal);
    },
    onFailed(id, error) {
      const item = selected.get(id)!.request;
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
        .filter((tile) => !tile.decoded)
        .map(({ request }) => request.id),
    );

  return {
    trimCache(bytes) {
      enforceCacheBudget(bytes);
      emit();
    },
    state,
    retryBlocked() {
      if (disposed) return;
      owner.retryBlocked();
      emit();
    },
    setSelection(requests) {
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
      const nextSelection = new Map<string, TileGeneration<T>>();
      const hits: TileGeneration<T>[] = [];
      for (const item of normalized.values()) {
        const retained = selected.get(item.id);
        if (retained) {
          nextSelection.set(item.id, retained);
          continue;
        }
        let cached = cache.get(item.id);
        if (cached && cached.request.url !== item.url) {
          evict(item.id);
          cached = undefined;
        }
        if (cached) {
          cacheHits += 1;
          cache.delete(item.id);
          cache.set(item.id, cached);
          hits.push(cached);
        } else {
          cacheMisses += 1;
        }
        nextSelection.set(item.id, cached ?? { request: item });
      }
      const previous = selected;
      selected = nextSelection;
      for (const [id, tile] of previous) {
        if (!selected.has(id) && !cache.has(id)) releaseDecoded(tile);
      }
      requestMissing();
      for (const hit of hits) {
        if (selected.get(hit.request.id) === hit && hit.decoded)
          safeCall(() =>
            options.onContent?.(hit.request, hit.decoded!.content),
          );
      }
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
        for (const tile of selected.values()) releaseDecoded(tile);
        revision = nextRevision;
        configGeneration = nextGeneration;
      } else {
        enforceCacheBudget();
      }
      if (!options.memory) admission.configure(maxConcurrency);
      owner.setConcurrency(maxConcurrency);
      if (changedGeneration) requestMissing();
      emit();
    },

    get(id) {
      const cached = cache.get(id);
      if (cached) {
        cache.delete(id);
        cache.set(id, cached);
      }
      return (selected.get(id) ?? cached)?.decoded?.content;
    },

    contents() {
      const live = new Map<string, T>();
      for (const [id, tile] of cache) live.set(id, tile.decoded!.content);
      for (const [id, tile] of selected) {
        if (tile.decoded) live.set(id, tile.decoded.content);
      }
      return live;
    },

    snapshot,

    dispose() {
      if (disposed) return;
      disposed = true;
      owner.dispose();
      clearCache();
      for (const tile of selected.values()) releaseDecoded(tile);
      selected.clear();
      emit();
    },
  };
};
