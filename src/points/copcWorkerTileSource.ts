import {
  getPageRequestAdmission,
  type RequestAdmission,
} from "../core/requestAdmission";
import type { CopcNodeEntry } from "./copcTileSource";
import type {
  CopcWorkerError,
  CopcWorkerRequest,
  CopcWorkerResponse,
} from "./copcWorkerProtocol";
import { ROOT_KEY, keyToString, type VoxelKey } from "./octree";
import type { LoadOptions, NodeInfo, TileData, TileSource } from "./tileSource";

export type CopcWorkerTileSourceOptions = {
  readonly signal?: AbortSignal;
  readonly admission?: RequestAdmission;
  /** Remote COPC URL or a local file/blob sent to the source workers. */
  readonly source: string | Blob;
  /** A fresh module worker, owned and terminated by the returned source. */
  readonly createWorker: () => Worker;
  /** URL of laz-perf.wasm. Defaults to `laz-perf.wasm` beside the document. */
  readonly lazPerfWasmUrl?: string;
  /**
   * Workers decoding tiles; the first also reads the hierarchy. Defaults to
   * one fewer than the logical cores, from one to three.
   */
  readonly workers?: number;
};

export class CopcWorkerCapacityError extends Error {
  constructor() {
    super(
      "All page COPC hierarchy worker slots are occupied. Close a COPC source before opening another.",
    );
    this.name = "CopcWorkerCapacityError";
  }
}

type PendingRequest = {
  readonly resolve: (response: CopcWorkerResponse) => void;
  readonly reject: (error: unknown) => void;
  readonly signal: AbortSignal | undefined;
  readonly onAbort: () => void;
  aborted: boolean;
};

/** One worker and the requests it has not answered yet. */
type Channel = {
  readonly send: (
    request: CopcWorkerRequest,
    signal?: AbortSignal,
  ) => Promise<CopcWorkerResponse>;
  readonly inFlight: () => number;
  /** Idempotent: fail everything in flight and stop the worker. */
  readonly close: (error: unknown) => void;
};

const abortReason = (signal: AbortSignal | undefined): unknown =>
  signal?.reason ?? new DOMException("The operation was aborted", "AbortError");

const remoteError = (input: CopcWorkerError): Error => {
  const error = new Error(input.message);
  error.name = input.name;
  return error;
};

const defaultWasmUrl = (): string => {
  if (typeof document === "undefined") return "laz-perf.wasm";
  return new URL("laz-perf.wasm", document.baseURI).href;
};

const defaultWorkerCount = (): number => {
  const cores = globalThis.navigator?.hardwareConcurrency ?? 2;
  return Math.max(1, Math.min(3, cores - 1));
};

/**
 * A channel settles an aborted request only after its worker reports that the
 * physical operation ended. `LodController` can therefore keep using promise
 * lifetime as its real concurrency ceiling even when an abort arrives during
 * synchronous WASM decoding.
 */
const openChannel = (
  worker: Worker,
  fail: (error: Error) => void,
  settled: () => void,
  release: () => void,
): Channel => {
  const pending = new Map<number, PendingRequest>();
  let closed = false;

  const detach = (request: PendingRequest): void =>
    request.signal?.removeEventListener("abort", request.onAbort);

  const onMessage = (event: MessageEvent<CopcWorkerResponse>): void => {
    const request = pending.get(event.data.id);
    if (request === undefined) return;
    pending.delete(event.data.id);
    detach(request);
    if (request.aborted) {
      request.reject(abortReason(request.signal));
    } else if (event.data.type === "error") {
      request.reject(remoteError(event.data.error));
    } else {
      request.resolve(event.data);
    }
    settled();
  };
  const onError = (event: ErrorEvent): void =>
    fail(new Error(event.message || "COPC worker failed"));
  const onMessageError = (): void =>
    fail(new Error("COPC worker returned an unreadable message"));
  worker.addEventListener("message", onMessage);
  worker.addEventListener("error", onError);
  worker.addEventListener("messageerror", onMessageError);

  return {
    send: (request, signal) => {
      if (closed) return Promise.reject(new Error("COPC source is disposed"));
      if (signal?.aborted) return Promise.reject(abortReason(signal));
      return new Promise((resolve, reject) => {
        const onAbort = (): void => {
          const current = pending.get(request.id);
          if (current === undefined || current.aborted) return;
          current.aborted = true;
          try {
            worker.postMessage({ type: "cancel", id: request.id });
          } catch {
            fail(new Error("Could not cancel COPC worker request"));
          }
        };
        pending.set(request.id, {
          resolve,
          reject,
          signal,
          onAbort,
          aborted: false,
        });
        signal?.addEventListener("abort", onAbort, { once: true });
        try {
          worker.postMessage(request);
        } catch {
          fail(new Error("Could not send COPC worker request"));
        }
      });
    },
    inFlight: () => pending.size,
    close: (error) => {
      if (closed) return;
      closed = true;
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      worker.removeEventListener("messageerror", onMessageError);
      for (const request of pending.values()) {
        detach(request);
        request.reject(error);
      }
      pending.clear();
      worker.terminate();
      release();
    },
  };
};

/**
 * Run COPC range reads, LAZ decoding, and point extraction in workers.
 *
 * The first worker opens the file and reads the hierarchy. The others are
 * decoders, opened from the header and colour shift the first one sampled;
 * each tile goes to the least busy worker, with its file location attached
 * for a decoder. Decoding is what bounds a COPC stream once its ranges are in
 * flight, so this multiplies how fast a view fills in. The hierarchy worker
 * failing fails the whole source, as it would with one worker; a decoder only
 * adds capacity, so one that fails is dropped and the tiles it held fail,
 * to be retried on the workers left.
 */
export const createCopcWorkerTileSource = async (
  options: CopcWorkerTileSourceOptions,
): Promise<TileSource> => {
  options.signal?.throwIfAborted();
  const admission = options.admission ?? getPageRequestAdmission();
  const lifetime = new AbortController();
  const workerCount = Math.max(
    1,
    Math.min(4, Math.floor(options.workers ?? defaultWorkerCount())),
  );
  const lazPerfWasmUrl = options.lazPerfWasmUrl ?? defaultWasmUrl();
  const channels = new Set<Channel>();
  let disposed = false;
  let nextId = 0;

  const teardown = (error: unknown): void => {
    if (disposed) return;
    disposed = true;
    lifetime.abort(error);
    for (const channel of channels) channel.close(error);
    channels.clear();
  };
  const abortOpen = (): void => teardown(abortReason(options.signal));
  options.signal?.addEventListener("abort", abortOpen, { once: true });
  let hierarchy!: Channel;
  let opened: CopcWorkerResponse;
  try {
    opened = await admission.run({
      signal: lifetime.signal,
      stage: {
        lane: "compute",
        ready: () =>
          admission.canAcquireWorker("copc") ||
          admission.workerCapacityPinned("copc"),
        async start(signal) {
          signal.throwIfAborted();
          const release = admission.acquireWorker("copc");
          if (!release) throw new CopcWorkerCapacityError();
          try {
            hierarchy = openChannel(
              options.createWorker(),
              teardown,
              admission.wake,
              release,
            );
            channels.add(hierarchy);
          } catch (error) {
            release();
            throw error;
          }
          return {
            value: await hierarchy.send({
              type: "open",
              id: ++nextId,
              source: options.source,
              lazPerfWasmUrl,
            }),
          };
        },
      },
    });
  } catch (error) {
    teardown(error);
    throw error;
  } finally {
    options.signal?.removeEventListener("abort", abortOpen);
  }
  options.signal?.throwIfAborted();
  if (opened.type !== "opened") {
    const error = new Error(
      `COPC worker returned ${opened.type} while opening`,
    );
    teardown(error);
    throw error;
  }
  const { metadata, state } = opened;

  /** Decoders that have finished opening; only these are given tiles. */
  const decoders: Channel[] = [];
  const drop = (decoder: Channel, error: unknown): void => {
    decoder.close(error);
    channels.delete(decoder);
    const index = decoders.indexOf(decoder);
    if (index >= 0) decoders.splice(index, 1);
  };
  let constructorFailed = false;
  const openDecoder = async (): Promise<boolean> => {
    if (!admission.canAcquireWorker("copc", false)) return false;
    return admission.run({
      signal: lifetime.signal,
      stage: {
        lane: "compute",
        async start(signal) {
          signal.throwIfAborted();
          if (constructorFailed) return { value: false };
          let decoder: Channel | undefined;
          const release = admission.acquireWorker(
            "copc",
            {
              idle: () => decoder !== undefined && decoder.inFlight() === 0,
              retire: () => {
                if (decoder) drop(decoder, new Error("COPC decoder retired"));
              },
            },
            false,
          );
          if (!release) return { value: false };
          try {
            decoder = openChannel(
              options.createWorker(),
              (error) => {
                if (decoder) drop(decoder, error);
              },
              admission.wake,
              release,
            );
            channels.add(decoder);
            await decoder.send({
              type: "open-decoder",
              id: ++nextId,
              source: options.source,
              lazPerfWasmUrl,
              state,
            });
            if (channels.has(decoder)) decoders.push(decoder);
            return { value: true };
          } catch (error) {
            if (decoder) drop(decoder, error);
            else {
              release();
              constructorFailed = true;
            }
            return { value: false };
          }
        },
      },
    });
  };
  // Extra codec workers are optional and yield to page limits and source lifetimes.
  for (let index = 1; index < workerCount; index += 1) {
    if (!admission.canAcquireWorker("copc", false)) break;
    void openDecoder().catch(() => {});
  }

  /** The file location of each node the hierarchy reads have described. */
  const entries = new Map<string, CopcNodeEntry>();
  const rootKey = keyToString(ROOT_KEY);

  const channelFor = (key: VoxelKey): Channel => {
    const keyString = keyToString(key);
    let channel = hierarchy;
    // The hierarchy worker holds the root sample and the only hierarchy state.
    if (keyString !== rootKey && entries.has(keyString)) {
      for (const decoder of decoders)
        if (decoder.inFlight() < channel.inFlight()) channel = decoder;
    }
    return channel;
  };

  const tileRequest = (
    key: VoxelKey,
  ): { readonly channel: Channel; readonly request: CopcWorkerRequest } => {
    const channel = channelFor(key);
    const entry = entries.get(keyToString(key));
    const id = ++nextId;
    return channel === hierarchy
      ? { channel, request: { type: "load-tile", id, key } }
      : { channel, request: { type: "load-tile", id, key, entry } };
  };

  return {
    admission,
    hierarchyLane: "compute",
    tileLane: "compute",
    canRead: (kind, key) => {
      const channel = kind === "hierarchy" ? hierarchy : channelFor(key);
      return disposed || channel.inFlight() === 0;
    },
    metadata: () => metadata,

    async nodes(key: VoxelKey, load?: LoadOptions): Promise<NodeInfo[]> {
      const response = await hierarchy.send(
        { type: "nodes", id: ++nextId, key },
        load?.signal,
      );
      if (response.type !== "nodes") {
        throw new Error(`COPC worker returned ${response.type} for hierarchy`);
      }
      for (const [keyString, entry] of response.entries) {
        entries.set(keyString, entry);
      }
      return response.nodes;
    },

    async loadTile(key: VoxelKey, load?: LoadOptions): Promise<TileData> {
      const entry = entries.get(keyToString(key));
      if (entry)
        load?.memory?.resize(
          Math.max(
            load.memory.bytes(),
            entry.pointCount * (state.copc.header.pointDataRecordLength + 30) +
              entry.pointDataLength * 2,
          ),
        );
      const { channel, request } = tileRequest(key);
      const response = await channel.send(request, load?.signal);
      if (response.type !== "tile") {
        throw new Error(`COPC worker returned ${response.type} for tile`);
      }
      return response.tile;
    },

    dispose() {
      teardown(new Error("COPC source is disposed"));
    },
  };
};
