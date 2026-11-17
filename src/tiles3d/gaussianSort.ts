import { affineProblem, type Mat16 } from "../core/mat4";
import { safeCall } from "../core/observers";

export type GaussianCameraPosition = readonly [number, number, number];

/**
 * Positions remain primitive-local and are borrowed, never transferred.
 * Keep them immutable until this scene and any in-flight jobs have retired.
 */
export type GaussianSortRange = {
  id: string;
  generation: number;
  positions: Float32Array;
  modelToScene: Mat16;
};

export type GaussianSortRangeIdentity = {
  id: string;
  generation: number;
  offset: number;
  count: number;
};

export type GaussianSortJob = {
  sceneRevision: number;
  ranges: readonly GaussianSortRange[];
  cameraPosition: GaussianCameraPosition;
};

export type GaussianSortPresentation = {
  sceneRevision: number;
  cameraPosition: GaussianCameraPosition;
  ranges: readonly GaussianSortRangeIdentity[];
  /** IDs into the concatenated ranges, farthest first; ties retain input order. */
  order: Uint32Array;
};

export type GaussianSortBackend = {
  sort(job: GaussianSortJob): Promise<Uint32Array>;
  dispose(): void;
};

/** The backend must be replaced before another job can use it. */
export class GaussianSortWorkerError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "GaussianSortWorkerError";
  }
}

const checkCamera = (camera: GaussianCameraPosition): void => {
  if (camera.length !== 3 || !camera.every(Number.isFinite)) {
    throw new TypeError(
      "Gaussian camera position must contain three finite numbers",
    );
  }
};

const rangeCount = (ranges: readonly GaussianSortRange[]): number => {
  let count = 0;
  const ids = new Set<string>();
  for (const range of ranges) {
    if (!range.id || ids.has(range.id)) {
      throw new TypeError(
        "Gaussian sort range identities must be unique and nonempty",
      );
    }
    ids.add(range.id);
    if (!Number.isSafeInteger(range.generation) || range.generation < 0) {
      throw new TypeError(
        "Gaussian sort range generation must be a nonnegative integer",
      );
    }
    if (range.positions.length % 3 !== 0 || affineProblem(range.modelToScene)) {
      throw new TypeError(
        "Gaussian sort range requires XYZ positions and a finite affine placement",
      );
    }
    count += range.positions.length / 3;
  }
  if (count > 0xffffffff) {
    throw new RangeError("Gaussian sort exceeds the Uint32 draw-order limit");
  }
  return count;
};

/** Stable LSD radix sorting over all 64 distance bits, without Float32 keys. */
export const createGaussianDistanceSorter = () => {
  let keys = new DataView(new ArrayBuffer(0));
  let scratch = new Uint32Array(0);
  const offsets = new Uint32Array(256);

  return {
    sort(
      ranges: readonly GaussianSortRange[],
      cameraPosition: GaussianCameraPosition,
    ): Uint32Array {
      checkCamera(cameraPosition);
      const count = rangeCount(ranges);
      if (scratch.length !== count) {
        keys = new DataView(new ArrayBuffer(count * 8));
        scratch = new Uint32Array(count);
      }
      const order = new Uint32Array(count);
      let id = 0;
      for (const range of ranges) {
        const matrix = range.modelToScene;
        // Subtract the camera before adding local offsets: nearby ECEF points
        // retain their separation without constructing Earth-scale float arrays.
        const tx = matrix[12]! - cameraPosition[0];
        const ty = matrix[13]! - cameraPosition[1];
        const tz = matrix[14]! - cameraPosition[2];
        for (
          let position = 0;
          position < range.positions.length;
          position += 3
        ) {
          const x = range.positions[position]!;
          const y = range.positions[position + 1]!;
          const z = range.positions[position + 2]!;
          const dx = tx + matrix[0]! * x + matrix[4]! * y + matrix[8]! * z;
          const dy = ty + matrix[1]! * x + matrix[5]! * y + matrix[9]! * z;
          const dz = tz + matrix[2]! * x + matrix[6]! * y + matrix[10]! * z;
          const distance = dx * dx + dy * dy + dz * dz;
          if (!Number.isFinite(distance)) {
            throw new RangeError(
              `Gaussian range ${range.id} has a nonfinite camera distance`,
            );
          }
          keys.setFloat64(id * 8, distance, true);
          order[id] = id;
          id += 1;
        }
      }

      let input = order;
      let output = scratch;
      // Positive IEEE-754 representations sort numerically as unsigned bits.
      // Complement every byte to obtain descending distance, retaining ties.
      for (let byte = 0; byte < 8; byte += 1) {
        offsets.fill(0);
        for (let index = 0; index < count; index += 1) {
          const bucket = keys.getUint8(input[index]! * 8 + byte) ^ 0xff;
          offsets[bucket]! += 1;
        }
        let start = 0;
        for (let bucket = 0; bucket < 256; bucket += 1) {
          const length = offsets[bucket]!;
          offsets[bucket] = start;
          start += length;
        }
        for (let index = 0; index < count; index += 1) {
          const point = input[index]!;
          const bucket = keys.getUint8(point * 8 + byte) ^ 0xff;
          output[offsets[bucket]!] = point;
          offsets[bucket]! += 1;
        }
        [input, output] = [output, input];
      }
      return order;
    },
  };
};

const sameCamera = (
  a: GaussianCameraPosition,
  b: GaussianCameraPosition,
): boolean => a.every((value, axis) => value === b[axis]);

/** One running job and one latest request. Compatible results publish in dispatch order. */
export const createGaussianSortController = ({
  backend,
  schedule = (run) => run(),
  canPresent,
  onOrder,
  onError,
  onWorkerError,
  onIdle,
}: {
  backend: GaussianSortBackend;
  /** Invoke once work is admitted; queued scene and camera changes coalesce. */
  schedule?: (run: () => Promise<void>) => Promise<void>;
  /** Check hard scene validity. The job may have an older camera or scene revision. */
  canPresent?(job: GaussianSortJob): boolean;
  onOrder(presentation: GaussianSortPresentation): void;
  onError(error: unknown): void;
  onWorkerError?(error: GaussianSortWorkerError): "retry" | "stop";
  onIdle?(): void;
}) => {
  let sceneRevision = 0;
  let ranges: readonly GaussianSortRange[] = [];
  let identities: readonly GaussianSortRangeIdentity[] = [];
  let camera: GaussianCameraPosition | null = null;
  let running = false;
  let completed: GaussianSortJob | null = null;
  let disposed = false;
  let stopped = false;
  const idleWaiters = new Set<() => void>();

  const settleIdle = (): void => {
    for (const resolve of idleWaiters) resolve();
    idleWaiters.clear();
    if (!disposed) safeCall(onIdle);
  };

  const current = (job: GaussianSortJob): boolean =>
    job.sceneRevision === sceneRevision &&
    camera !== null &&
    sameCamera(job.cameraPosition, camera);

  const start = (): void => {
    if (disposed || running || !camera) return;
    if (stopped) {
      settleIdle();
      return;
    }
    if (completed && current(completed)) {
      settleIdle();
      return;
    }
    let job: GaussianSortJob | null = null;
    running = true;
    void Promise.resolve()
      .then(() => {
        if (disposed) return;
        return schedule(async () => {
          if (disposed || !camera || (completed && current(completed))) return;
          const dispatched: GaussianSortJob = {
            sceneRevision,
            ranges,
            cameraPosition: camera,
          };
          job = dispatched;
          const jobIdentities = identities;
          const order = await backend.sort(dispatched);
          if (!disposed) {
            completed = dispatched;
            if (
              canPresent?.(dispatched) ??
              dispatched.sceneRevision === sceneRevision
            )
              safeCall(() =>
                onOrder({
                  sceneRevision: dispatched.sceneRevision,
                  cameraPosition: dispatched.cameraPosition,
                  ranges: jobIdentities,
                  order,
                }),
              );
          }
        });
      })
      .catch((error: unknown) => {
        if (disposed || !camera) return;
        const failed = job ?? { sceneRevision, ranges, cameraPosition: camera };
        completed = failed;
        if (error instanceof GaussianSortWorkerError && onWorkerError)
          safeCall(() => {
            const action = onWorkerError(error);
            if (action === "retry") completed = null;
            else stopped = true;
          });
        else if (current(failed)) safeCall(() => onError(error));
      })
      .finally(() => {
        running = false;
        if (disposed) settleIdle();
        else start();
      });
  };

  return {
    setScene(
      next: readonly GaussianSortRange[],
      options: { defer?: boolean } = {},
    ): void {
      if (disposed) return;
      rangeCount(next);
      completed = null;
      ranges = next.map((range) => ({
        ...range,
        modelToScene: Float64Array.from(range.modelToScene),
      }));
      let offset = 0;
      identities = ranges.map((range) => {
        const count = range.positions.length / 3;
        const identity = {
          id: range.id,
          generation: range.generation,
          offset,
          count,
        };
        offset += count;
        return identity;
      });
      sceneRevision += 1;
      if (!options.defer) start();
    },
    request(cameraPosition: GaussianCameraPosition): void {
      if (disposed) return;
      checkCamera(cameraPosition);
      camera = [...cameraPosition];
      start();
    },
    whenIdle(): Promise<void> {
      return !running
        ? Promise.resolve()
        : new Promise((resolve) => {
            idleWaiters.add(resolve);
          });
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      ranges = [];
      identities = [];
      camera = null;
      completed = null;
      backend.dispose();
    },
  };
};

export type GaussianSortWorkerLike = {
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror?: ((event: MessageEvent) => void) | null;
  postMessage(message: unknown): void;
  terminate(): void;
};

export type GaussianSortWorkerRequest =
  | {
      kind: "scene";
      sceneRevision: number;
      ranges: readonly GaussianSortRange[];
    }
  | {
      kind: "sort";
      jobId: number;
      sceneRevision: number;
      cameraPosition: GaussianCameraPosition;
    };

export type GaussianSortWorkerResponse =
  | { kind: "order"; jobId: number; order: Uint32Array }
  | { kind: "error"; jobId: number; message: string };

/** Owns and terminates one worker. Scene attributes are cloned once per revision. */
export const createGaussianSortWorkerBackend = (
  worker: GaussianSortWorkerLike,
): GaussianSortBackend => {
  let disposed = false;
  let revision: number | null = null;
  let sequence = 0;
  let pending: {
    id: number;
    count: number;
    resolve(order: Uint32Array): void;
    reject(error: Error): void;
  } | null = null;

  const fail = (error: Error): void => {
    const active = pending;
    pending = null;
    active?.reject(error);
  };
  const terminate = (error: Error): void => {
    if (disposed) return;
    disposed = true;
    worker.onmessage = null;
    worker.onerror = null;
    worker.onmessageerror = null;
    worker.terminate();
    fail(error);
  };
  worker.onmessage = ({ data }: MessageEvent<unknown>) => {
    if (disposed || !pending || !data || typeof data !== "object") return;
    const reply = data as {
      jobId?: unknown;
      kind?: unknown;
      order?: unknown;
      message?: unknown;
    };
    if (reply.jobId !== pending.id) return;
    if (reply.kind === "error" && typeof reply.message === "string") {
      fail(new Error(reply.message));
    } else if (
      reply.kind === "order" &&
      reply.order instanceof Uint32Array &&
      reply.order.length === pending.count
    ) {
      const active = pending;
      pending = null;
      active.resolve(reply.order);
    } else {
      terminate(
        new GaussianSortWorkerError(
          "Gaussian sort worker returned an invalid response",
        ),
      );
    }
  };
  worker.onerror = (event) =>
    terminate(
      new GaussianSortWorkerError(
        event.message || "Gaussian sort worker failed",
      ),
    );
  worker.onmessageerror = () =>
    terminate(
      new GaussianSortWorkerError(
        "Gaussian sort worker response could not be read",
      ),
    );

  return {
    sort(job) {
      if (disposed)
        return Promise.reject(
          new GaussianSortWorkerError("Gaussian sort worker is disposed"),
        );
      if (pending)
        return Promise.reject(new Error("Gaussian sort worker is busy"));
      return new Promise((resolve, reject) => {
        sequence += 1;
        pending = {
          id: sequence,
          count: job.ranges.reduce(
            (sum, range) => sum + range.positions.length / 3,
            0,
          ),
          resolve,
          reject,
        };
        try {
          if (revision !== job.sceneRevision) {
            worker.postMessage({
              kind: "scene",
              sceneRevision: job.sceneRevision,
              ranges: job.ranges,
            } satisfies GaussianSortWorkerRequest);
            revision = job.sceneRevision;
          }
          worker.postMessage({
            kind: "sort",
            jobId: sequence,
            sceneRevision: job.sceneRevision,
            cameraPosition: job.cameraPosition,
          } satisfies GaussianSortWorkerRequest);
        } catch (error) {
          terminate(
            new GaussianSortWorkerError("Could not send Gaussian sort work", {
              cause: error,
            }),
          );
        }
      });
    },
    dispose() {
      terminate(
        new GaussianSortWorkerError("Gaussian sort worker is disposed"),
      );
    },
  };
};
