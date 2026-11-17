import { describe, expect, it, vi } from "vitest";

// Record tuple ownership and presentation while running the real manager,
// scheduler, sorter, and shared resource ledger without a browser.
const vtk = vi.hoisted(() => {
  const instance = (initial: Record<string, any> = {}): any => {
    let data = initial.values ?? new Float32Array();
    let components = initial.numberOfComponents ?? 1;
    let mtime = 0;
    let mapper: any, pointOrder: any, inputData: any, userMatrix: number[];
    const children = new Map();
    const target = {
      ...initial,
      setData(values: ArrayLike<number>, width = 1) {
        data = values;
        components = width;
        mtime++;
      },
      getData: () => data,
      getNumberOfTuples: () => data.length / components,
      setMapper(value: any) {
        mapper = value;
      },
      getMapper: () => mapper,
      setPointOrder(value: any) {
        pointOrder = value;
      },
      getPointOrder: () => pointOrder,
      setInputData(value: any) {
        inputData = value;
      },
      getInputData: () => inputData,
      setUserMatrix(value: number[]) {
        userMatrix = value;
      },
      getUserMatrix: () => userMatrix,
      getMTime: () => mtime,
      modified() {
        mtime++;
      },
    };
    return new Proxy(target, {
      get(value, key) {
        if (key in value) return value[key as keyof typeof value];
        if (
          key === "getPoints" ||
          key === "getPointData" ||
          key === "getProperty" ||
          key === "getPolys"
        )
          return () => {
            if (!children.has(key)) children.set(key, instance());
            return children.get(key);
          };
        return () => {};
      },
    });
  };
  return { default: { newInstance: instance, extend() {} } };
});
vi.mock("@kitware/vtk.js/Common/Core/DataArray", () => vtk);
vi.mock("@kitware/vtk.js/Common/DataModel/PolyData", () => vtk);
vi.mock("@kitware/vtk.js/Rendering/Core/Actor", () => vtk);
vi.mock("@kitware/vtk.js/Rendering/Core/Mapper", () => vtk);
vi.mock("@kitware/vtk.js/Rendering/Core/PointGaussianMapper", () => vtk);
vi.mock("@kitware/vtk.js/Rendering/SceneGraph/RenderPass", () => vtk);

import { IDENTITY } from "../core/mat4";
import { createMemoryPool } from "../core/memoryPool";
import { ResourcePolicyError } from "../core/memoryResources";
import { createSubmissionScheduler } from "../core/submissionScheduler";
import { createGaussianContentAdapterFactory } from "./gaussianAdapter";
import {
  createGaussianDistanceSorter,
  createGaussianSortWorkerBackend,
  type GaussianSortBackend,
  type GaussianSortRange,
  type GaussianSortWorkerLike,
  type GaussianSortWorkerRequest,
} from "./gaussianSort";
import type { DecodedTileContent } from "./decode";

const field = (count: number, mixed = false): DecodedTileContent => ({
  origin: [0, 0, 0],
  byteEstimate: { geometry: count * 56, textures: 0 },
  primitives: [
    {
      kind: "gaussian",
      positions: new Float32Array(count * 3),
      scales: new Float32Array(count * 3).fill(0.1),
      rotations: Float32Array.from({ length: count * 4 }, (_, index) =>
        index % 4 === 3 ? 1 : 0,
      ),
      opacities: new Float32Array(count).fill(0.5),
      sphericalHarmonics: new Float32Array(count * 3),
      harmonicDegree: 0,
      modelToScene: new Float64Array(IDENTITY),
      profile: {
        kernel: "ellipse",
        colorSpace: "srgb_rec709_display",
        projection: "perspective",
        sortingMethod: "cameraDistance",
      },
    },
    ...(mixed
      ? [
          {
            kind: "mesh" as const,
            positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]),
            material: {
              baseColorFactor: [1, 0, 0, 1] as [number, number, number, number],
              raw: {
                version: 1 as const,
                kind: "gltf-material" as const,
                alphaMode: "OPAQUE" as const,
                alphaCutoff: 0.5,
                doubleSided: false,
                unlit: true,
                metallicFactor: 0,
                roughnessFactor: 1,
                emissiveFactor: [0, 0, 0] as [number, number, number],
              },
            },
          },
        ]
      : []),
  ],
});

const fixture = (
  createSortBackend?: () => GaussianSortBackend,
  maximumVisibleSplats?: number,
  maxBytesPerFrame?: number,
) => {
  const actors = new Set<any>();
  const renderer = {
    addActor(actor: any) {
      actors.add(actor);
    },
    removeActor(actor: any) {
      actors.delete(actor);
    },
  };
  const pool = createMemoryPool({
    totalBytes: 1000000,
    cpuBytes: 1000000,
    workBytes: 1000000,
  });
  const memory = pool.createOwner();
  const extraMemory: (typeof memory)[] = [];
  const submissions = createSubmissionScheduler({
    scheduleRender() {},
    ...(maxBytesPerFrame === undefined ? {} : { maxBytesPerFrame }),
    now: () => 0,
  });
  const distance = createGaussianDistanceSorter();
  const errors: unknown[] = [],
    refusals: unknown[] = [];
  let held = false;
  const pending: (() => void)[] = [];
  const factory = createGaussianContentAdapterFactory({
    maximumVisibleSplats,
    createSortBackend:
      createSortBackend ??
      (() => ({
        sort(job) {
          const order = distance.sort(job.ranges, job.cameraPosition);
          return held
            ? new Promise<Uint32Array>((resolve) => {
                pending.push(() => resolve(order));
              })
            : Promise.resolve(order);
        },
        dispose() {
          for (const release of pending.splice(0)) release();
        },
      })),
  });
  const adapter = factory({
    renderer,
    memoryPool: pool,
    memory,
    submissions,
    scheduleRender() {},
    onError: (error) => errors.push(error),
    onPolicyRefusal: (error) => refusals.push(error),
  });
  const move = (x: number) =>
    adapter.setCamera({
      projection: "perspective",
      fovY: Math.PI / 3,
      position: [x, 0, 10],
      viewProj: IDENTITY,
      viewportWidthCssPx: 100,
      viewportHeightCssPx: 100,
    });
  move(0);
  const drainUntil = async (condition: () => boolean) => {
    while (!condition()) {
      submissions.prepareFrame();
      await new Promise<void>((resolve) => setImmediate(resolve));
    }
  };
  const withFrames = async <T>(pending: Promise<T>): Promise<T> => {
    let done = false;
    void pending.finally(() => (done = true));
    await drainUntil(() => done);
    return pending;
  };
  const settle = async () => {
    let idle = false;
    void factory.whenIdle(renderer).then(() => (idle = true));
    await drainUntil(() => idle && !submissions.hasPending());
  };
  return {
    adapter,
    pool,
    factory,
    renderer,
    errors,
    refusals,
    submissions,
    settle,
    drainUntil,
    withFrames,
    move,
    drawnX() {
      for (const actor of actors) {
        const mapper = actor.getMapper();
        const order = mapper?.getPointOrder();
        if (!order) continue;
        const positions: Float32Array = mapper
          .getInputData()
          .getPoints()
          .getData();
        return Array.from(
          order.getData() as Uint32Array,
          (id) => positions[id * 3]! + actor.getUserMatrix()[12],
        );
      }
      return [];
    },
    additionalAdapter() {
      const otherMemory = pool.createOwner();
      extraMemory.push(otherMemory);
      return factory({
        renderer,
        memoryPool: pool,
        memory: otherMemory,
        submissions,
        scheduleRender() {},
        onError: (error) => errors.push(error),
        onPolicyRefusal: (error) => refusals.push(error),
      });
    },
    hold() {
      held = true;
    },
    releaseNext() {
      pending.shift()!();
    },
    release() {
      held = false;
      for (const release of pending.splice(0)) release();
    },
    async started() {
      await drainUntil(() => pending.length > 0);
      submissions.prepareFrame();
      expect(pending).toHaveLength(1);
    },
    async show(id: string, count: number, mixed = false) {
      expect(adapter.submitTile(id, field(count, mixed))).toBe("queued");
      await settle();
      adapter.setDrawnTiles([id]);
      await settle();
    },
    dispose() {
      factory.dispose(renderer);
      memory.close();
      for (const owner of extraMemory) owner.close();
      submissions.dispose();
    },
  };
};

const preparePartial = async (f: ReturnType<typeof fixture>, mixed = false) => {
  await f.show("a", 1, mixed);
  expect(f.adapter.submitTile("b", field(4, mixed))).toBe("queued");
  await f.settle();
  f.adapter.setDrawnTiles(["a", "b"]);
  await f.settle();
  f.adapter.setResourceCeilingBytes(mixed ? 636 : 480);
  expect(f.adapter.submitTile("newA", field(4, mixed), undefined, ["a"])).toBe(
    "queued",
  );
  await f.settle();
  expect(f.adapter.tileState("newA")).toBe("submitted");
};

const meshOnly = (): DecodedTileContent => {
  const content = field(0, true);
  return {
    ...content,
    primitives: content.primitives.filter(
      (primitive) => primitive.kind === "mesh",
    ),
  };
};
const showMeshParent = async (f: ReturnType<typeof fixture>) => {
  expect(f.adapter.submitTile("parent", meshOnly())).toBe("queued");
  await f.settle();
  f.adapter.setDrawnTiles(["parent"]);
  await f.settle();
};

const locatedField = (xs: readonly number[], mixed = false) => {
  const content = field(xs.length, mixed);
  const primitive = content.primitives[0]!;
  xs.forEach((x, index) => {
    primitive.positions[index * 3] = x;
  });
  return content;
};

const controlledSortWorkers = () => {
  type Dispatch = {
    camera: readonly number[];
    worker: GaussianSortWorkerLike;
    finish(): void;
    crash(): void;
  };
  let held = false;
  let started: ((dispatch: Dispatch) => void) | undefined;
  let pending: Dispatch | undefined;
  const workers: GaussianSortWorkerLike[] = [];
  const create = () => {
    const sorter = createGaussianDistanceSorter();
    let ranges: readonly GaussianSortRange[] = [];
    const worker: GaussianSortWorkerLike = {
      onmessage: null,
      onerror: null,
      terminate: vi.fn(),
      postMessage(message) {
        const request = message as GaussianSortWorkerRequest;
        if (request.kind === "scene") {
          ranges = request.ranges;
          return;
        }
        const deliver = worker.onmessage!;
        const order = sorter.sort(ranges, request.cameraPosition);
        const dispatch: Dispatch = {
          camera: request.cameraPosition,
          worker,
          finish() {
            deliver({
              data: { kind: "order", jobId: request.jobId, order },
            } as MessageEvent);
          },
          crash() {
            worker.onerror!({ message: "worker crashed" } as ErrorEvent);
          },
        };
        if (!held) dispatch.finish();
        else if (started) {
          const resolve = started;
          started = undefined;
          resolve(dispatch);
        } else pending = dispatch;
      },
    };
    workers.push(worker);
    return createGaussianSortWorkerBackend(worker);
  };
  return {
    create,
    workers,
    hold() {
      held = true;
    },
    next(): Promise<Dispatch> {
      if (!pending)
        return new Promise((resolve) => {
          started = resolve;
        });
      const dispatch = pending;
      pending = undefined;
      return Promise.resolve(dispatch);
    },
  };
};

describe("Gaussian sort publication", () => {
  it("publishes captured geometry, order, and meshes while camera and selection keep changing", async () => {
    const f = fixture();
    try {
      f.adapter.submitTile("a", locatedField([0, 2], true));
      f.adapter.submitTile("b", locatedField([5, 9, 11], true));
      await f.settle();
      f.adapter.setDrawnTiles(["a"]);
      await f.settle();

      f.hold();
      f.adapter.setDrawnTiles(["a", "b"]);
      f.move(1);
      await f.started();
      f.adapter.setDrawnTiles(["b"]);
      f.move(7);
      f.releaseNext();
      await f.started();
      expect(f.drawnX()).toEqual([11, 9, 5, 0, 2]);
      expect(f.adapter.pickableMeshTiles().map(({ id }) => id)).toEqual([
        "a",
        "b",
      ]);
      expect(f.factory.stats(f.renderer)?.pendingSort).toBe(true);

      f.adapter.setDrawnTiles(["a"]);
      f.move(10);
      f.releaseNext();
      await f.started();
      expect(f.drawnX()).toEqual([11, 5, 9]);
      expect(f.adapter.pickableMeshTiles().map(({ id }) => id)).toEqual(["b"]);

      f.releaseNext();
      await f.settle();
      expect(f.drawnX()).toEqual([0, 2]);
      expect(f.adapter.pickableMeshTiles().map(({ id }) => id)).toEqual(["a"]);
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("keeps the accepted placement until its replacement sort completes", async () => {
    const f = fixture();
    try {
      await f.show("field", 1, true);
      f.hold();
      f.move(1);
      await f.started();
      const revision = f.factory.stats(f.renderer)?.presentationRevision;
      const matrix = [...IDENTITY];
      matrix[12] = 10;
      f.adapter.setBaseMatrix(matrix);
      f.releaseNext();
      await f.started();
      expect(f.factory.stats(f.renderer)?.presentationRevision).toBe(revision);
      expect(f.drawnX()).toEqual([0]);
      expect(f.adapter.presentationMatrix()[12]).toBe(0);

      f.releaseNext();
      await f.settle();
      expect(f.drawnX()).toEqual([10]);
      expect(f.adapter.presentationMatrix()[12]).toBe(10);
    } finally {
      f.dispose();
    }
  });

  it("keeps the newer mesh selection when a cached order publishes before an in-flight sort", async () => {
    const f = fixture();
    try {
      await f.show("field", 1);
      f.adapter.submitTile("previous", meshOnly());
      f.adapter.submitTile("current", meshOnly());
      await f.settle();
      f.adapter.setDrawnTiles(["field", "previous"]);
      await f.settle();

      f.hold();
      f.move(1);
      await f.started();
      f.adapter.setDrawnTiles(["field", "current"]);
      f.move(0);
      await f.drainUntil(() =>
        f.adapter.pickableMeshTiles().some(({ id }) => id === "current"),
      );
      expect(f.adapter.pickableMeshTiles().map(({ id }) => id)).toEqual([
        "current",
      ]);
      f.releaseNext();
      await f.started();
      expect(f.adapter.pickableMeshTiles().map(({ id }) => id)).toEqual([
        "current",
      ]);

      f.releaseNext();
      await f.settle();
      expect(f.adapter.pickableMeshTiles().map(({ id }) => id)).toEqual([
        "current",
      ]);
    } finally {
      f.dispose();
    }
  });

  it("keeps physical work charged and rejects a cleared generation before reusing its slots", async () => {
    const f = fixture();
    try {
      await f.show("field", 1);
      f.hold();
      f.move(1);
      await f.started();
      const bytes = f.pool.stats().workReservedBytes;
      expect(bytes).toBeGreaterThan(0);
      f.adapter.clearTiles();
      await Promise.resolve();
      expect(f.drawnX()).toEqual([]);
      expect(f.pool.stats().workReservedBytes).toBe(bytes);
      expect(f.adapter.submitTile("field", locatedField([10]))).toBe("queued");
      f.submissions.prepareFrame();
      f.releaseNext();
      await f.started();
      expect(f.drawnX()).toEqual([]);

      f.release();
      await f.settle();
      f.adapter.setDrawnTiles(["field"]);
      await f.settle();
      expect(f.drawnX()).toEqual([10]);
      expect(f.pool.stats().workReservedBytes).toBe(0);
    } finally {
      f.dispose();
    }
  });

  it("restarts a crashed worker for the latest coverage and ignores late replies", async () => {
    const workers = controlledSortWorkers();
    const f = fixture(workers.create);
    try {
      await f.show("parent", 1, true);
      f.adapter.submitTile("child", locatedField([5, 9], true), undefined, [
        "parent",
      ]);
      await f.settle();
      workers.hold();
      f.move(1);
      const crashed = await workers.next();

      f.adapter.setDrawnTiles(["child"]);
      f.move(7);
      crashed.crash();
      const retry = await f.withFrames(workers.next());
      expect(crashed.worker.terminate).toHaveBeenCalledOnce();
      expect(retry.worker).not.toBe(crashed.worker);
      expect(retry.camera).toEqual([7, 0, 10]);
      expect(f.pool.stats().activeOperations).toBe(1);
      crashed.finish();
      expect(f.drawnX()).toEqual([0]);
      expect(f.adapter.pickableMeshTiles().map(({ id }) => id)).toEqual([
        "parent",
      ]);

      retry.finish();
      await f.settle();
      expect(f.drawnX()).toEqual([5, 9]);
      expect(f.adapter.pickableMeshTiles().map(({ id }) => id)).toEqual([
        "child",
      ]);
      expect(f.pool.stats().workReservedBytes).toBe(0);
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("restarts a lazy backend when worker construction fails at dispatch", async () => {
    const workers = controlledSortWorkers();
    const unavailable = {
      sort() {
        throw new Error("worker construction failed");
      },
      dispose: vi.fn(),
    };
    const create = vi.fn(workers.create).mockReturnValueOnce(unavailable);
    const f = fixture(create);
    try {
      await f.show("field", 1);
      expect(create).toHaveBeenCalledTimes(2);
      expect(unavailable.dispose).toHaveBeenCalledOnce();
      expect(f.drawnX()).toEqual([0]);
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("keeps the last coherent draw and preserves admissions when the restart also fails", async () => {
    const workers = controlledSortWorkers();
    const f = fixture(workers.create);
    try {
      await f.show("accepted", 1, true);
      const other = f.additionalAdapter();
      workers.hold();
      f.move(1);
      const crashed = await workers.next();
      expect(other.submitTile("unrelated", locatedField([9], true))).toBe(
        "queued",
      );
      f.submissions.prepareFrame();
      crashed.crash();
      const retry = await f.withFrames(workers.next());
      retry.crash();
      await f.settle();
      expect(other.tileState("unrelated")).toBe("submitted");
      other.setDrawnTiles(["unrelated"]);
      f.move(2);
      await f.settle();
      expect(workers.workers).toHaveLength(2);
      expect(f.errors).toHaveLength(2);
      expect(f.drawnX()).toEqual([0]);
      expect(f.adapter.pickableMeshTiles().map(({ id }) => id)).toEqual([
        "accepted",
      ]);
      expect(f.pool.stats()).toMatchObject({
        activeOperations: 0,
        workReservedBytes: 0,
      });
    } finally {
      f.dispose();
    }
  });
});

describe("Gaussian admission ownership", () => {
  it("admits another pure-mesh successor while a retained parent lease is already claimed", async () => {
    const f = fixture();
    try {
      await showMeshParent(f);
      const bytes = f.adapter.estimateTileBytes(meshOnly());
      expect(f.adapter.submitTile("a", meshOnly(), undefined, ["parent"])).toBe(
        "queued",
      );
      await f.settle();
      expect(f.adapter.tileState("a")).toBe("submitted");
      expect(f.adapter.stats().drawnTileIds).toEqual(["parent"]);
      expect(f.adapter.submitTile("b", meshOnly(), undefined, ["parent"])).toBe(
        "queued",
      );
      await f.settle();
      expect(f.adapter.tileState("b")).toBe("submitted");
      expect(f.pool.stats().rendererBytes).toBe(bytes * 3);
      f.adapter.setDrawnTiles(["a", "b"]);
      f.adapter.retireTile("parent");
      await f.settle();
      expect(f.adapter.pickableMeshTiles().map((tile) => tile.id)).toEqual([
        "a",
        "b",
      ]);
      expect(f.pool.stats().rendererBytes).toBe(bytes * 2);
      expect(f.errors).toEqual([]);
      expect(f.refusals).toEqual([]);
      f.dispose();
      expect(f.pool.stats()).toMatchObject({
        rendererBytes: 0,
        cpuRetainedBytes: 0,
        cpuReservedBytes: 0,
        activeOperations: 0,
        owners: 0,
      });
    } finally {
      f.dispose();
    }
  });

  it("keeps retained mesh credit unavailable while a Gaussian atlas holder survives successor mesh retirement", async () => {
    const f = fixture();
    try {
      await showMeshParent(f);
      const bytes = f.adapter.estimateTileBytes(meshOnly());
      expect(
        f.adapter.submitTile("mixed", field(1, true), undefined, ["parent"]),
      ).toBe("queued");
      await f.settle();
      expect(f.adapter.tileState("mixed")).toBe("submitted");
      expect(f.adapter.retireTile("mixed")).toBe(true);
      // The mesh is gone synchronously, but its composite atlas still owns
      // the old parent's credit until the queued refresh physically clears it.
      expect(
        f.adapter.submitTile("next", meshOnly(), undefined, ["parent"]),
      ).toBe("queued");
      await f.settle();
      expect(f.adapter.tileState("next")).toBe("submitted");
      expect(f.pool.stats().rendererBytes).toBe(bytes * 2);
      expect(f.adapter.stats().drawnTileIds).toEqual(["parent"]);
      expect(f.errors).toEqual([]);
      expect(f.refusals).toEqual([]);
      f.dispose();
      expect(f.pool.stats()).toMatchObject({
        rendererBytes: 0,
        cpuRetainedBytes: 0,
        cpuReservedBytes: 0,
        activeOperations: 0,
        owners: 0,
      });
    } finally {
      f.dispose();
    }
  });

  it("keeps a retained parent's credit claimed until every successor-group lease releases", async () => {
    const f = fixture();
    try {
      await showMeshParent(f);
      const bytes = f.adapter.estimateTileBytes(meshOnly());
      expect(
        f.adapter.submitTileGroup(
          ["a", "b"].map((id) => ({ id, content: meshOnly() })),
          ["parent"],
        ),
      ).toBe("queued");
      await f.settle();
      f.adapter.retireTile("b");
      await f.settle();
      expect(f.adapter.submitTile("c", meshOnly(), undefined, ["parent"])).toBe(
        "queued",
      );
      await f.settle();
      expect(f.pool.stats().rendererBytes).toBe(bytes * 3);
      f.adapter.retireTile("a");
      await f.settle();
      f.pool.configure({ totalBytes: bytes * 2, transitionBytes: bytes });
      expect(f.adapter.submitTile("d", meshOnly(), undefined, ["parent"])).toBe(
        "queued",
      );
      await f.settle();
      expect(f.adapter.tileState("d")).toBe("submitted");
      expect(f.pool.stats().rendererBytes).toBe(bytes * 3);
      expect(f.errors).toEqual([]);
      expect(f.refusals).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("refuses reusing retained-parent credit under a tight pool and recovers after the claimant releases", async () => {
    const f = fixture();
    try {
      await showMeshParent(f);
      const bytes = f.adapter.estimateTileBytes(meshOnly());
      expect(f.adapter.submitTile("a", meshOnly(), undefined, ["parent"])).toBe(
        "queued",
      );
      await f.settle();
      f.pool.configure({ totalBytes: bytes * 2, transitionBytes: bytes });
      expect(
        f.adapter.submitTile("b", meshOnly(), undefined, ["parent"]),
      ).toMatchObject({
        status: "blocked",
        refusal: {
          domain: "renderer",
          estimate: {
            reason: "contention",
            wake: "availability",
            finalBytes: bytes * 3,
            peakBytes: bytes * 3,
          },
        },
      });
      await f.settle();
      expect(f.adapter.stats().drawnTileIds).toEqual(["parent"]);
      expect(f.pool.stats().rendererBytes).toBe(bytes * 2);
      expect(f.errors).toEqual([]);
      f.adapter.retireTile("a");
      await f.settle();
      expect(f.adapter.submitTile("b", meshOnly(), undefined, ["parent"])).toBe(
        "queued",
      );
      await f.settle();
      expect(f.adapter.tileState("b")).toBe("submitted");
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("releases the complete canceled mesh group's replacement eligibility", async () => {
    const f = fixture();
    try {
      await showMeshParent(f);
      const bytes = f.adapter.estimateTileBytes(meshOnly());
      expect(
        f.adapter.submitTileGroup(
          ["a", "b"].map((id) => ({ id, content: meshOnly() })),
          ["parent"],
        ),
      ).toBe("queued");
      expect(f.adapter.cancelTile("a")).toBe(true);
      await f.settle();
      expect(f.adapter.tileState("a")).toBe("absent");
      expect(f.adapter.tileState("b")).toBe("absent");
      expect(f.pool.stats().rendererBytes).toBe(bytes);
      f.pool.configure({ totalBytes: bytes, transitionBytes: bytes });
      expect(f.adapter.submitTile("d", meshOnly(), undefined, ["parent"])).toBe(
        "queued",
      );
      await f.settle();
      expect(f.adapter.tileState("d")).toBe("submitted");
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("does not attach a retired parent generation's credit claim to a newly admitted parent with the same ID", async () => {
    const f = fixture();
    try {
      await showMeshParent(f);
      const bytes = f.adapter.estimateTileBytes(meshOnly());
      expect(f.adapter.submitTile("a", meshOnly(), undefined, ["parent"])).toBe(
        "queued",
      );
      await f.settle();
      f.adapter.setDrawnTiles(["a"]);
      f.adapter.retireTile("parent");
      await f.settle();
      expect(f.adapter.submitTile("parent", meshOnly())).toBe("queued");
      await f.settle();
      f.adapter.setDrawnTiles(["parent"]);
      await f.settle();
      f.pool.configure({ totalBytes: bytes * 2, transitionBytes: bytes });
      expect(f.adapter.submitTile("b", meshOnly(), undefined, ["parent"])).toBe(
        "queued",
      );
      await f.settle();
      expect(f.adapter.tileState("b")).toBe("submitted");
      expect(f.errors).toEqual([]);
      expect(f.refusals).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("coarsens under a smaller grant while retaining both atlases until the sort publishes", async () => {
    const f = fixture();
    try {
      await f.show("detail", 8);
      expect(f.pool.stats().rendererBytes).toBe(480);
      f.pool.configure({ totalBytes: 240, transitionBytes: 300 });
      f.adapter.setResourceCeilingBytes(240);
      f.hold();
      expect(
        f.adapter.submitTile(
          "parent",
          field(1),
          () => {
            f.adapter.setDrawnTiles(["parent"]);
            f.adapter.retireTile("detail");
          },
          ["detail"],
        ),
      ).toBe("queued");
      f.submissions.prepareFrame();
      await f.started();
      expect(f.pool.stats().rendererBytes).toBe(540);
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(8);
      expect(f.adapter.tileState("parent")).toBe("submitted");
      expect(f.adapter.setResourceCeilingBytes(240)).toEqual([]);
      expect(f.adapter.tileState("parent")).toBe("submitted");
      f.release();
      await f.settle();
      expect(f.pool.stats().rendererBytes).toBe(60);
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(1);
      expect(f.adapter.stats().drawnTileIds).toEqual(["parent"]);
      expect(f.errors).toEqual([]);
      expect(f.refusals).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("keeps the caller-selected mixed frontier while a replacement is only preadmitted", async () => {
    const f = fixture();
    try {
      await f.show("parent", 1, true);
      expect(
        f.adapter.submitTile("child", field(1, true), undefined, ["parent"]),
      ).toBe("queued");
      await f.settle();
      expect(f.adapter.tileState("child")).toBe("submitted");
      expect(f.adapter.stats().drawnTileIds).toEqual(["parent"]);
      expect(f.adapter.pickableMeshTiles().map((tile) => tile.id)).toEqual([
        "parent",
      ]);
      f.hold();
      f.adapter.setDrawnTiles(["child"]);
      await f.started();
      expect(f.adapter.pickableMeshTiles().map((tile) => tile.id)).toEqual([
        "parent",
      ]);
      f.release();
      await f.settle();
      expect(f.adapter.stats().drawnTileIds).toEqual(["child"]);
      expect(f.adapter.pickableMeshTiles().map((tile) => tile.id)).toEqual([
        "child",
      ]);
      expect(f.errors).toEqual([]);
      expect(f.refusals).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("retains compact preadmission capacity against competing owners until explicit selection", async () => {
    const f = fixture();
    const competitor = f.pool.createOwner();
    try {
      f.pool.configure({ totalBytes: 540, transitionBytes: 0 });
      await f.show("parent", 1);
      expect(
        f.adapter.submitTile("child", field(8), undefined, ["parent"]),
      ).toBe("queued");
      await f.settle();
      expect(f.adapter.tileState("child")).toBe("submitted");
      expect(f.pool.stats().rendererBytes).toBe(540);
      expect(f.adapter.stats().drawnTileIds).toEqual(["parent"]);
      expect(() => competitor.reserveRenderer(480)).toThrow(
        ResourcePolicyError,
      );
      f.adapter.setDrawnTiles(["child"]);
      await f.settle();
      expect(f.adapter.stats().drawnTileIds).toEqual(["child"]);
      expect(f.pool.stats().rendererBytes).toBe(480);
      expect(f.errors).toEqual([]);
      expect(f.refusals).toEqual([]);
    } finally {
      competitor.close();
      f.dispose();
    }
  });

  it("retains tuples for multiple preadmitted groups against competing owners", async () => {
    const f = fixture();
    const competitor = f.pool.createOwner();
    try {
      f.pool.configure({ totalBytes: 4096, transitionBytes: 0 });
      await f.show("a", 1);
      expect(f.adapter.submitTile("b", field(1))).toBe("queued");
      await f.settle();
      f.adapter.setDrawnTiles(["a", "b"]);
      await f.settle();
      for (const parent of ["a", "b"]) {
        expect(
          f.adapter.submitTileGroup(
            ["L", "R"].map((suffix) => ({
              id: parent + suffix,
              content: field(4),
            })),
            [parent],
          ),
        ).toBe("queued");
        await f.settle();
      }
      const children = ["aL", "aR", "bL", "bR"];
      for (const id of children)
        expect(f.adapter.tileState(id)).toBe("submitted");
      expect(f.adapter.stats().drawnTileIds).toEqual(["a", "b"]);
      expect(() => competitor.reserveRenderer(3137)).toThrow(
        ResourcePolicyError,
      );
      f.adapter.setDrawnTiles(children);
      await f.settle();
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(16);
      expect(f.adapter.stats().drawnTileIds).toEqual(children);
      expect(f.errors).toEqual([]);
      expect(f.refusals).toEqual([]);
    } finally {
      competitor.close();
      f.dispose();
    }
  });

  it("coalesces a selected mixed frontier and placement while sort admission waits", async () => {
    const f = fixture();
    const blocker = f.pool.createOwner();
    const work = [];
    try {
      await f.show("parent", 1, true);
      expect(
        f.adapter.submitTile("child", field(1, true), undefined, ["parent"]),
      ).toBe("queued");
      await f.settle();
      work.push(blocker.work(undefined, f.pool.stats().workBytes));
      f.hold();
      const matrix = [...IDENTITY];
      matrix[12] = 0.6;
      f.adapter.setBaseMatrix(matrix);
      await f.drainUntil(() => f.pool.stats().queuedOperations > 0);
      expect(f.pool.stats().queuedOperations).toBe(1);
      matrix[12] = -0.6;
      f.adapter.setBaseMatrix(matrix);
      f.adapter.setDrawnTiles(["child"]);
      expect(f.adapter.pickableMeshTiles().map((tile) => tile.id)).toEqual([
        "parent",
      ]);
      for (const entry of work.splice(0)) entry.release();
      await f.started();
      f.releaseNext();
      await f.settle();
      expect(f.adapter.presentationMatrix()[12]).toBe(-0.6);
      expect(f.adapter.pickableMeshTiles().map((tile) => tile.id)).toEqual([
        "child",
      ]);
      expect(f.errors).toEqual([]);
      expect(f.refusals).toEqual([]);
    } finally {
      for (const entry of work) entry.release();
      blocker.close();
      f.dispose();
    }
  });

  it.each([false, true])(
    "publishes a compact partial frontier coherently (mixed=%s)",
    async (mixed) => {
      const f = fixture();
      try {
        await preparePartial(f, mixed);
        f.hold();
        const matrix = [...IDENTITY];
        matrix[12] = 0.6;
        f.adapter.setBaseMatrix(matrix);
        f.adapter.setDrawnTiles(["newA", "b"]);
        await f.started();
        expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(5);
        expect(f.adapter.stats().drawnTileIds).toEqual(["a", "b"]);
        expect(f.adapter.pickableMeshTiles().map((tile) => tile.id)).toEqual(
          mixed ? ["a", "b"] : [],
        );
        expect(f.adapter.presentationMatrix()[12]).toBe(0);
        f.release();
        await f.settle();
        expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(8);
        expect([...f.adapter.stats().drawnTileIds].sort()).toEqual([
          "b",
          "newA",
        ]);
        expect(
          f.adapter
            .pickableMeshTiles()
            .map((tile) => tile.id)
            .sort(),
        ).toEqual(mixed ? ["b", "newA"] : []);
        expect(f.adapter.presentationMatrix()[12]).toBe(0.6);
        expect(f.pool.stats().rendererBytes).toBe(
          f.factory.stats(f.renderer)?.residentBytes,
        );
        expect(f.errors).toEqual([]);
        expect(f.refusals).toEqual([]);
        f.dispose();
        expect(f.pool.stats()).toMatchObject({
          rendererBytes: 0,
          cpuRetainedBytes: 0,
          cpuReservedBytes: 0,
          activeOperations: 0,
          owners: 0,
        });
      } finally {
        f.dispose();
      }
    },
  );

  it("publishes a broad ready region while a larger unrelated field is still packing", async () => {
    const f = fixture(undefined, undefined, 64);
    try {
      await f.show("root/0/0", 1);
      expect(f.adapter.submitTile("root/2/0/0", field(200))).toBe("queued");
      await f.drainUntil(() => f.submissions.hasPending());
      f.submissions.prepareFrame();
      expect(f.adapter.submitTile("root/1", locatedField([10]))).toBe("queued");
      await f.drainUntil(() => f.adapter.tileState("root/1") === "submitted");
      expect(f.adapter.tileState("root/2/0/0")).toBe("queued");
      f.adapter.setDrawnTiles(["root/1"]);
      await f.drainUntil(() => f.drawnX()[0] === 10);
      expect(f.adapter.tileState("root/2/0/0")).toBe("queued");
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("keeps a blocked region while unrelated complete replacements publish", async () => {
    const f = fixture();
    try {
      expect(f.adapter.submitTile("left", locatedField([0, 1], true))).toBe(
        "queued",
      );
      expect(
        f.adapter.submitTile("right", locatedField([20, 21, 22], true)),
      ).toBe("queued");
      await f.settle();
      f.adapter.setDrawnTiles(["left", "right"]);
      await f.settle();
      const accepted = f.pool.stats().rendererBytes;
      // The right group fits after its predecessor is gone, but its atlas
      // cannot coexist with current coverage under this hard peak.
      f.pool.configure({ totalBytes: accepted + 2 * 52, transitionBytes: 0 });
      const blocked = f.adapter.submitTile(
        "right/0",
        locatedField([30, 31, 32, 33, 34, 35], true),
        undefined,
        ["right"],
      );
      expect(blocked).toMatchObject({ status: "blocked" });
      expect(
        f.adapter.submitTileGroup(
          [
            { id: "left/0", content: locatedField([10], true) },
            { id: "left/1", content: locatedField([11], true) },
          ],
          ["left"],
        ),
      ).toBe("queued");
      await f.settle();
      f.hold();
      f.adapter.setDrawnTiles(["left/0", "left/1", "right"]);
      await f.started();
      expect(
        f.adapter
          .pickableMeshTiles()
          .map(({ id }) => id)
          .sort(),
      ).toEqual(["left", "right"]);
      f.release();
      await f.settle();
      expect(
        f.adapter
          .pickableMeshTiles()
          .map(({ id }) => id)
          .sort(),
      ).toEqual(["left/0", "left/1", "right"]);
      expect(f.drawnX().sort((a, b) => a - b)).toEqual([10, 11, 20, 21, 22]);
      expect(f.pool.stats().rendererBytes).toBeLessThanOrEqual(
        accepted + 2 * 52,
      );
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
    expect(f.pool.stats()).toMatchObject({
      rendererBytes: 0,
      cpuRetainedBytes: 0,
      owners: 0,
    });
  });

  it("keeps Gaussian coverage until a staged mesh-only replacement is selected", async () => {
    const f = fixture();
    try {
      await f.show("fine", 8, true);
      const accepted = f.pool.stats().rendererBytes;
      const meshBytes = f.adapter.estimateTileBytes(meshOnly());
      f.pool.configure({ totalBytes: meshBytes, transitionBytes: accepted });
      f.adapter.setResourceCeilingBytes(meshBytes);
      expect(
        f.adapter.submitTile("coarse", meshOnly(), undefined, ["fine"]),
      ).toBe("queued");
      await f.settle();
      expect(f.adapter.tileState("coarse")).toBe("submitted");
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(8);
      expect(f.adapter.pickableMeshTiles().map(({ id }) => id)).toEqual([
        "fine",
      ]);
      f.adapter.setDrawnTiles(["coarse"]);
      f.adapter.retireTile("fine");
      await f.settle();
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(0);
      expect(f.adapter.pickableMeshTiles().map(({ id }) => id)).toEqual([
        "coarse",
      ]);
      expect(f.pool.stats().rendererBytes).toBe(meshBytes);
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("releases canceled compact scratch while the old frontier remains pickable", async () => {
    const f = fixture();
    try {
      await preparePartial(f, true);
      f.hold();
      f.adapter.setDrawnTiles(["newA", "b"]);
      await f.started();
      f.adapter.retireTile("newA");
      f.adapter.setDrawnTiles(["a", "b"]);
      f.release();
      await f.settle();
      expect(
        f.adapter
          .pickableMeshTiles()
          .map(({ id }) => id)
          .sort(),
      ).toEqual(["a", "b"]);
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(5);
      expect(f.pool.stats().rendererBytes).toBe(
        f.factory.stats(f.renderer)?.residentBytes,
      );
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("preserves an admitted mixed replacement when its final grant is reapplied during sorting", async () => {
    const f = fixture();
    try {
      await f.show("detail", 8, true);
      const oldBytes = f.pool.stats().rendererBytes;
      const meshBytes =
        f.adapter.estimateTileBytes(field(1, true)) -
        f.adapter.estimateTileBytes(field(1));
      const finalBytes = 60 + meshBytes;
      f.pool.configure({ totalBytes: finalBytes, transitionBytes: oldBytes });
      f.adapter.setResourceCeilingBytes(finalBytes);
      f.hold();
      expect(
        f.adapter.submitTile(
          "parent",
          field(1, true),
          () => {
            f.adapter.setDrawnTiles(["parent"]);
            f.adapter.retireTile("detail");
          },
          ["detail"],
        ),
      ).toBe("queued");
      f.submissions.prepareFrame();
      await f.started();
      expect(f.pool.stats().rendererBytes).toBe(oldBytes + finalBytes);
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(8);
      expect(f.adapter.setResourceCeilingBytes(finalBytes)).toEqual([]);
      expect(f.adapter.tileState("parent")).toBe("submitted");
      f.release();
      await f.settle();
      expect(f.pool.stats().rendererBytes).toBe(finalBytes);
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(1);
      expect(f.adapter.pickableMeshTiles()).toHaveLength(1);
      expect(f.errors).toEqual([]);
      expect(f.refusals).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("compacts one member without retiring another member's shared atlas tuples", async () => {
    const f = fixture();
    try {
      await f.show("detail", 8);
      const other = f.additionalAdapter();
      expect(other.submitTile("other", field(2))).toBe("queued");
      await f.settle();
      other.setDrawnTiles(["other"]);
      await f.settle();
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(10);
      f.adapter.setResourceCeilingBytes(90);
      expect(
        f.adapter.submitTile(
          "parent",
          field(1),
          () => {
            f.adapter.setDrawnTiles(["parent"]);
            f.adapter.retireTile("detail");
          },
          ["detail"],
        ),
      ).toBe("queued");
      await f.settle();
      expect(f.pool.stats().rendererBytes).toBe(240);
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(3);
      expect(other.stats().drawnTileIds).toEqual(["other"]);
      expect(f.adapter.stats().drawnTileIds).toEqual(["parent"]);
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("releases a failed mixed realization and keeps the accepted parent", async () => {
    const f = fixture();
    try {
      await f.show("parent", 2);
      const before = f.pool.stats().rendererBytes;
      f.renderer.addActor = () => {
        throw new Error("mesh attachment failed");
      };
      expect(
        f.adapter.submitTile("child", field(1, true), undefined, ["parent"]),
      ).toBe("queued");
      await f.settle();
      expect(f.adapter.tileState("child")).toBe("failed");
      expect(f.adapter.stats()).toMatchObject({
        drawnTileIds: ["parent"],
        pendingTiles: 0,
      });
      expect(f.pool.stats().rendererBytes).toBe(before);
      expect(f.errors).toEqual([new Error("mesh attachment failed")]);
    } finally {
      f.dispose();
    }
  });

  it("refuses an over-count representation independently of renderer memory", async () => {
    const f = fixture(undefined, 2);
    try {
      await f.show("parent", 2);
      const before = f.pool.stats().rendererBytes;
      expect(
        f.adapter.submitTile("child", field(3), undefined, ["parent"]),
      ).toMatchObject({
        status: "blocked",
        refusal: {
          domain: "count",
          finalCount: 3,
          peakCount: 5,
          limitCount: 2,
          wake: "selection",
        },
      });
      expect(f.adapter.stats().drawnTileIds).toEqual(["parent"]);
      expect(f.pool.stats().rendererBytes).toBe(before);
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("preserves coverage on transition refusal and recovers when overlap capacity increases", async () => {
    const f = fixture();
    try {
      await f.show("detail", 8);
      const baseline = f.pool.stats().cpuRetainedBytes;
      f.pool.configure({ totalBytes: 240, transitionBytes: 0 });
      f.adapter.setResourceCeilingBytes(240);
      const submit = () =>
        f.adapter.submitTile(
          "parent",
          field(1),
          () => {
            f.adapter.setDrawnTiles(["parent"]);
            f.adapter.retireTile("detail");
          },
          ["detail"],
        );
      expect(submit()).toMatchObject({
        status: "blocked",
        refusal: {
          domain: "renderer",
          estimate: {
            reason: "transition-unavailable",
            wake: "policy",
            finalBytes: 60,
            peakBytes: 540,
          },
        },
      });
      await f.settle();
      expect(f.pool.stats().rendererBytes).toBe(480);
      expect(f.pool.stats().cpuRetainedBytes).toBe(baseline);
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(8);
      f.pool.configure({ transitionBytes: 300 });
      expect(submit()).toBe("queued");
      await f.settle();
      expect(f.pool.stats().rendererBytes).toBe(60);
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(1);
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("returns to the displayed atlas when a compact candidate is canceled during sorting", async () => {
    const f = fixture();
    try {
      await f.show("detail", 8);
      f.adapter.setResourceCeilingBytes(240);
      f.hold();
      expect(
        f.adapter.submitTile(
          "parent",
          field(1),
          () => f.adapter.setDrawnTiles(["parent"]),
          ["detail"],
        ),
      ).toBe("queued");
      f.submissions.prepareFrame();
      await f.started();
      expect(f.pool.stats().rendererBytes).toBe(540);
      expect(f.adapter.retireTile("parent")).toBe(true);
      f.adapter.setDrawnTiles(["detail"]);
      f.release();
      await f.settle();
      expect(f.adapter.tileState("parent")).toBe("absent");
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(8);
      expect(f.pool.stats().rendererBytes).toBe(480);
      expect(f.errors).toEqual([]);
      expect(f.refusals).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("releases speculative growth before compacting after a sort workspace refusal", async () => {
    const f = fixture();
    try {
      await f.show("field", 1);
      f.pool.configure({ workBytes: 0 });
      f.adapter.setCamera({
        projection: "perspective",
        fovY: Math.PI / 3,
        position: [0.1, 0, 10],
        viewProj: IDENTITY,
        viewportWidthCssPx: 100,
        viewportHeightCssPx: 100,
      });
      expect(f.adapter.submitTile("growth", field(8))).toBe("queued");
      await f.settle();
      expect(f.adapter.submitTile("cancelled", field(32))).toBe("queued");
      expect(f.adapter.cancelTile("cancelled")).toBe(true);
      await f.settle();
      expect(f.pool.stats().rendererBytes).toBe(
        f.factory.stats(f.renderer)?.residentBytes,
      );
      f.pool.configure({ workBytes: 1000000 });
      f.adapter.clearRequiredBytes();
      expect(f.adapter.submitTile("retry", field(32))).toBe("queued");
      // Match separate host RPC turns: policy refresh runs before queued
      // upload work is prepared, so speculative retry ownership must survive.
      await Promise.resolve();
      expect(f.adapter.tileState("retry")).toBe("queued");
      await f.settle();
      expect(f.pool.stats().rendererBytes).toBe(
        f.factory.stats(f.renderer)?.residentBytes,
      );
      expect(f.adapter.tileState("retry")).toBe("submitted");
      expect(f.adapter.stats().drawnTileIds).toEqual(["field"]);
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("retires part of a frontier without reuploading its accepted geometry", async () => {
    const f = fixture();
    try {
      await f.show("a", 8);
      expect(f.adapter.submitTile("b", field(5))).toBe("queued");
      await f.settle();
      f.adapter.setDrawnTiles(["a", "b"]);
      await f.settle();
      const before = f.factory.stats(f.renderer)?.attributeUploadBytes;
      f.adapter.retireTile("b");
      await f.settle();
      expect(f.adapter.stats().drawnTileIds).toEqual(["a"]);
      expect(f.factory.stats(f.renderer)).toMatchObject({
        displayedSplats: 8,
        attributeUploadBytes: before,
        pendingPreparation: false,
      });
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("retires stable tuples without requiring extra capacity for optional compaction", async () => {
    const f = fixture();
    try {
      await f.show("a", 1);
      expect(f.adapter.submitTile("b", field(1))).toBe("queued");
      await f.settle();
      f.adapter.setDrawnTiles(["a", "b"]);
      await f.settle();
      f.pool.configure({ totalBytes: 120, transitionBytes: 0 });
      f.adapter.retireTile("b");
      await f.settle();
      expect(f.adapter.stats().drawnTileIds).toEqual(["a"]);
      expect(f.factory.stats(f.renderer)?.displayedSplats).toBe(1);
      expect(f.pool.stats().rendererBytes).toBe(120);
      expect(f.refusals).toEqual([]);
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("readmits a group of retained mixed generations without deleting their meshes", async () => {
    const f = fixture();
    try {
      await f.show("a", 1, true);
      expect(f.adapter.submitTile("b", field(1, true))).toBe("queued");
      await f.settle();
      f.adapter.setDrawnTiles(["a", "b"]);
      await f.settle();
      const baseline = f.pool.stats().cpuRetainedBytes;
      f.adapter.retireTile("a");
      f.adapter.retireTile("b");
      expect(
        f.adapter.submitTileGroup(
          [
            { id: "a", content: field(1, true) },
            { id: "b", content: field(1, true) },
          ],
          [],
        ),
      ).toBe("queued");
      await f.settle();
      f.adapter.setDrawnTiles(["a", "b"]);
      await f.settle();
      expect(f.adapter.stats().drawnTileIds).toEqual(["a", "b"]);
      expect(f.adapter.pickableMeshTiles()).toHaveLength(2);
      expect(f.pool.stats().cpuRetainedBytes).toBe(baseline);
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it("rebuilds an undrawn mixed generation whose mesh resources already retired", async () => {
    const f = fixture();
    try {
      await f.show("a", 1, true);
      f.adapter.setDrawnTiles([]);
      await f.settle();
      f.adapter.retireTile("a");
      expect(f.adapter.submitTile("a", field(1, true))).toBe("queued");
      await f.settle();
      f.adapter.setDrawnTiles(["a"]);
      await f.settle();
      expect(f.adapter.pickableMeshTiles()).toHaveLength(1);
      expect(f.errors).toEqual([]);
    } finally {
      f.dispose();
    }
  });

  it.each([false, true])(
    "readmits the retained immutable generation (mixed=%s)",
    async (mixed) => {
      const f = fixture();
      try {
        await f.show("a", 1, mixed);
        const baseline = f.pool.stats().cpuRetainedBytes;
        expect(f.adapter.retireTile("a")).toBe(true);
        expect(f.adapter.submitTile("a", field(1, mixed))).toBe("queued");
        await f.settle();
        f.adapter.setDrawnTiles(["a"]);
        await f.settle();
        expect(f.adapter.tileState("a")).toBe("submitted");
        expect(f.adapter.stats().drawnTileIds).toEqual(["a"]);
        expect(f.pool.stats().cpuRetainedBytes).toBe(baseline);
        expect(f.errors).toEqual([]);
      } finally {
        f.dispose();
      }
    },
  );
});
