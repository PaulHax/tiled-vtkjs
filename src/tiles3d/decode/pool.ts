import {
  getPageRequestAdmission,
  type RequestAdmission,
} from "../../core/requestAdmission";
import type {
  DecodeJob,
  DecodeTileRequest,
  DecodedTileContent,
  DecodeWorkerPoolHandle,
} from "./types";
import {
  ResourcePolicyError,
  type MemoryWork,
  type MemoryDomain,
} from "../../core/memoryResources";
import {
  TileDecodeError,
  TileUnsupportedExtensionError,
  TileUnsupportedGaussianProfileError,
  type TileDecodeStage,
} from "./types";

type DecodeWorkerRequestMessage = {
  kind: "decode";
  jobId: number;
  request: DecodeTileRequest;
  managedMemory?: boolean;
};

export type DecodeMemoryRequestMessage = {
  kind: "memory-request";
  jobId: number;
  requestId: number;
  bytes: number;
};
export type DecodeMemoryResponseMessage = {
  kind: "memory-response";
  jobId: number;
  requestId: number;
  error?: {
    domain: MemoryDomain;
    requestedBytes: number;
    limitBytes: number;
    retryable: boolean;
    operationBytes?: number;
  };
};

type DecodeWorkerCompleteMessage = {
  kind: "complete";
  jobId: number;
  result: DecodedTileContent;
};

type DecodeWorkerErrorMessage = {
  kind: "error";
  jobId: number;
  error: {
    readonly name: string;
    readonly message: string;
    readonly tileUri?: string;
    readonly stage?: TileDecodeStage;
    readonly reason?: string;
    readonly extension?: string;
    readonly domain?: MemoryDomain;
    readonly requestedBytes?: number;
    readonly limitBytes?: number;
    readonly retryable?: boolean;
    readonly operationBytes?: number;
    readonly property?: string;
    readonly value?: unknown;
  };
};

type DecodeWorkerResponseMessage =
  | DecodeWorkerCompleteMessage
  | DecodeWorkerErrorMessage;

export type DecodeWorkerLike = {
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage(
    message: DecodeWorkerRequestMessage | DecodeMemoryResponseMessage,
    transfer?: Transferable[],
  ): void;
  terminate(): void;
};

export type DecodeWorkerPoolOptions = {
  size?: number;
  workerFactory?: (index: number) => DecodeWorkerLike;
  workerUrl?: string | URL;
  admission?: RequestAdmission;
};

type JobRecord = {
  jobId: number;
  request: DecodeTileRequest;
  state: "active" | "settled";
  startedAt: number;
  resolve(value: DecodedTileContent): void;
  reject(reason: unknown): void;
  memory?: MemoryWork;
};

type WorkerSlot = {
  worker: DecodeWorkerLike;
  readonly release: () => void;
  active: JobRecord | null;
};

const BASIS_TIMING_SAMPLE_LIMIT = 64;

type MutableBasisTargetTiming = {
  count: number;
  totalMs: number;
  samplesMs: number[];
};

const abortError = (): DOMException =>
  new DOMException("Decode cancelled", "AbortError");

const defaultPoolSize = (): number => {
  const cores = globalThis.navigator?.hardwareConcurrency ?? 2;
  return Math.max(1, Math.min(4, cores - 1));
};

const defaultWorkerFactory =
  (workerUrl: string | URL) => (): DecodeWorkerLike =>
    new Worker(workerUrl, {
      // The artifact is deliberately classic because the Emscripten wrapper
      // dependencies are installed through importScripts.
      type: "classic",
      name: "tiles3d-decode",
    });

export class DecodeWorkerPoolDisposedError extends Error {
  constructor() {
    super("Decode worker pool has been disposed");
    this.name = "DecodeWorkerPoolDisposedError";
  }
}

export class DecodeWorkerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DecodeWorkerError";
  }
}

export class DecodeWorkerPool implements DecodeWorkerPoolHandle {
  readonly #size: number;
  readonly #factory: (index: number) => DecodeWorkerLike;
  readonly #admission: RequestAdmission;
  #slots: (WorkerSlot | undefined)[];
  #nextJobId = 1;
  #disposed = false;
  #completedJobs = 0;
  #failedJobs = 0;
  #cancelledJobs = 0;
  #workerElapsedMs = 0;
  #basisRuntimeInitializationMs = 0;
  #basisTranscodeMs = 0;
  #basisTextures = 0;
  #basisTargets = new Map<string, number>();
  #basisTargetTimings = new Map<string, MutableBasisTargetTiming>();

  constructor(options: DecodeWorkerPoolOptions = {}) {
    this.#size = options.size ?? defaultPoolSize();
    if (!Number.isInteger(this.#size) || this.#size <= 0) {
      throw new Error("Decode worker pool size must be a positive integer");
    }
    const workerUrl =
      options.workerUrl ??
      new URL(
        /* @vite-ignore */ "./tiles3dDecodeWorker.classic.js",
        import.meta.url,
      );
    this.#factory = options.workerFactory ?? defaultWorkerFactory(workerUrl);
    this.#admission = options.admission ?? getPageRequestAdmission();
    this.#slots = Array.from({ length: this.#size });
  }

  hasCapacity(): boolean {
    if (this.#disposed) return false;
    return (
      this.#slots.some((slot) => slot?.active === null) ||
      (this.#slots.includes(undefined) &&
        this.#admission.canAcquireWorker("decode"))
    );
  }

  decode(
    request: DecodeTileRequest,
    options: { readonly memory?: MemoryWork } = {},
  ): DecodeJob {
    if (this.#disposed) throw new DecodeWorkerPoolDisposedError();
    let slot = this.#slots.find((candidate) => candidate?.active === null);
    if (!slot) {
      const index = this.#slots.indexOf(undefined);
      if (index < 0) throw new DecodeWorkerError("All decode workers are busy");
      slot = this.#makeSlot(index);
      this.#slots[index] = slot;
    }
    let resolve!: (value: DecodedTileContent) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<DecodedTileContent>(
      (resolvePromise, rejectPromise) => {
        resolve = resolvePromise;
        reject = rejectPromise;
      },
    );
    const record: JobRecord = {
      jobId: this.#nextJobId,
      request,
      state: "active",
      startedAt: globalThis.performance?.now?.() ?? Date.now(),
      resolve,
      reject,
      memory: options.memory,
    };
    this.#nextJobId += 1;
    slot.active = record;
    try {
      slot.worker.postMessage(
        {
          kind: "decode",
          jobId: record.jobId,
          request: record.request,
          managedMemory: record.memory !== undefined,
        },
        [record.request.content],
      );
    } catch (cause) {
      this.#onWorkerFailure(
        slot,
        new DecodeWorkerError(
          cause instanceof Error ? cause.message : "Could not send decode work",
        ),
      );
    }
    return {
      promise,
      cancel: () => this.#cancel(record),
    };
  }

  stats() {
    return {
      size: this.#size,
      activeJobs: this.#slots.filter((slot) => slot?.active).length,
      completedJobs: this.#completedJobs,
      failedJobs: this.#failedJobs,
      cancelledJobs: this.#cancelledJobs,
      workerElapsedMs: this.#workerElapsedMs,
      basisRuntimeInitializationMs: this.#basisRuntimeInitializationMs,
      basisTranscodeMs: this.#basisTranscodeMs,
      basisTextures: this.#basisTextures,
      basisTargets: Object.fromEntries(this.#basisTargets),
      basisTargetTimings: Object.fromEntries(
        [...this.#basisTargetTimings].map(([target, timing]) => [
          target,
          { ...timing, samplesMs: [...timing.samplesMs] },
        ]),
      ),
    };
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const error = new DecodeWorkerPoolDisposedError();
    for (const slot of this.#slots) {
      if (!slot) continue;
      this.#retire(slot);
      if (slot.active) this.#reject(slot.active, error);
    }
    this.#slots = [];
  }

  #makeSlot(index: number): WorkerSlot {
    let slot: WorkerSlot | undefined;
    const release = this.#admission.acquireWorker("decode", {
      idle: () => slot?.active === null,
      retire: () => {
        if (slot) this.#retire(slot);
      },
    });
    if (!release)
      throw new DecodeWorkerError("Page decode worker capacity is occupied");
    try {
      slot = { worker: this.#factory(index), release, active: null };
      const next = slot;
      next.worker.onmessage = (event) => this.#onMessage(next, event.data);
      next.worker.onerror = (event) =>
        this.#onWorkerFailure(
          next,
          new DecodeWorkerError(event.message || "decode worker failed"),
        );
      return next;
    } catch (error) {
      release();
      throw error;
    }
  }

  #retire(slot: WorkerSlot): void {
    const index = this.#slots.indexOf(slot);
    if (index >= 0) this.#slots[index] = undefined;
    slot.worker.onmessage = null;
    slot.worker.onerror = null;
    slot.worker.terminate();
    slot.release();
  }

  #onMessage(slot: WorkerSlot, value: unknown): void {
    if (
      value &&
      typeof value === "object" &&
      "kind" in value &&
      value.kind === "memory-request"
    ) {
      const message = value as DecodeMemoryRequestMessage;
      const active = slot.active;
      if (!active || active.jobId !== message.jobId) return;
      const response: DecodeMemoryResponseMessage = {
        kind: "memory-response",
        jobId: message.jobId,
        requestId: message.requestId,
      };
      try {
        active.memory?.resize(message.bytes);
      } catch (error) {
        if (!(error instanceof ResourcePolicyError)) {
          this.#onWorkerFailure(
            slot,
            new DecodeWorkerError("invalid memory reservation request"),
          );
          return;
        }
        response.error = {
          domain: error.domain,
          requestedBytes: error.requestedBytes,
          limitBytes: error.limitBytes,
          retryable: error.retryable,
          operationBytes: error.operationBytes,
        };
      }
      slot.worker.postMessage(response);
      return;
    }
    if (
      !value ||
      typeof value !== "object" ||
      !("kind" in value) ||
      !("jobId" in value) ||
      (value.kind !== "complete" && value.kind !== "error") ||
      typeof value.jobId !== "number"
    ) {
      this.#onWorkerFailure(
        slot,
        new DecodeWorkerError("decode worker returned a malformed response"),
      );
      return;
    }
    const message = value as DecodeWorkerResponseMessage;
    const active = slot.active;
    // Job ids are never reused, so the id alone tells a stale message apart.
    if (!active || message.jobId !== active.jobId) return;
    slot.active = null;
    if (active.state === "active") {
      if (message.kind === "complete") {
        active.state = "settled";
        const now = globalThis.performance?.now?.() ?? Date.now();
        this.#workerElapsedMs += Math.max(0, now - (active.startedAt ?? now));
        this.#completedJobs += 1;
        const diagnostics = message.result.diagnostics;
        if (diagnostics) {
          this.#basisRuntimeInitializationMs +=
            diagnostics.basisRuntimeInitializationMs;
          this.#basisTranscodeMs += diagnostics.basisTranscodeMs;
          this.#basisTextures += diagnostics.basisTextures;
          if (diagnostics.basisTarget) {
            const target = diagnostics.basisTarget;
            this.#basisTargets.set(
              target,
              (this.#basisTargets.get(target) ?? 0) + diagnostics.basisTextures,
            );
            const timing = this.#basisTargetTimings.get(target) ?? {
              count: 0,
              totalMs: 0,
              samplesMs: [],
            };
            timing.count += diagnostics.basisTextures;
            timing.totalMs += diagnostics.basisTranscodeMs;
            for (const sample of diagnostics.basisTranscodeSamplesMs) {
              if (Number.isFinite(sample) && sample >= 0) {
                timing.samplesMs.push(sample);
              }
            }
            if (timing.samplesMs.length > BASIS_TIMING_SAMPLE_LIMIT) {
              timing.samplesMs.splice(
                0,
                timing.samplesMs.length - BASIS_TIMING_SAMPLE_LIMIT,
              );
            }
            this.#basisTargetTimings.set(target, timing);
          }
        }
        active.resolve(message.result);
      } else {
        this.#failedJobs += 1;
        const payload = message.error;
        const decodeError =
          payload?.name === "ResourcePolicyError" &&
          payload.domain &&
          typeof payload.requestedBytes === "number" &&
          typeof payload.limitBytes === "number"
            ? new ResourcePolicyError(
                payload.domain,
                payload.requestedBytes,
                payload.limitBytes,
                payload.retryable,
                payload.operationBytes,
              )
            : payload?.name === "TileUnsupportedExtensionError" &&
                typeof payload.tileUri === "string" &&
                typeof payload.extension === "string"
              ? new TileUnsupportedExtensionError(
                  payload.tileUri,
                  payload.extension,
                )
              : payload?.name === "TileUnsupportedGaussianProfileError" &&
                  typeof payload.tileUri === "string" &&
                  typeof payload.property === "string"
                ? new TileUnsupportedGaussianProfileError(
                    payload.tileUri,
                    payload.property,
                    payload.value,
                  )
                : payload?.name === "TileDecodeError" &&
                    typeof payload.tileUri === "string" &&
                    (payload.stage === "profile" ||
                      payload.stage === "decode") &&
                    typeof payload.reason === "string"
                  ? new TileDecodeError(
                      payload.tileUri,
                      payload.stage,
                      payload.reason,
                    )
                  : new DecodeWorkerError(
                      typeof payload?.message === "string"
                        ? payload.message
                        : "decode worker returned an invalid error",
                    );
        this.#reject(active, decodeError);
      }
    }
    this.#admission.wake();
  }

  #onWorkerFailure(slot: WorkerSlot, error: DecodeWorkerError): void {
    const active = slot.active;
    this.#retire(slot);
    if (active) {
      this.#failedJobs += 1;
      this.#reject(active, error);
    }
  }

  #cancel(record: JobRecord): void {
    if (record.state !== "active") return;
    const slot = this.#slots.find((candidate) => candidate?.active === record);
    if (slot) this.#retire(slot);
    this.#cancelledJobs += 1;
    this.#reject(record, abortError());
  }

  #reject(record: JobRecord, reason: unknown): void {
    if (record.state === "settled") return;
    record.state = "settled";
    record.reject(reason);
  }
}

export type {
  DecodeWorkerCompleteMessage,
  DecodeWorkerErrorMessage,
  DecodeWorkerRequestMessage,
};
