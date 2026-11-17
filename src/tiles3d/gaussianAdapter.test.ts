import { describe, expect, it, vi } from "vitest";

// Record tuple ownership and presentation while running the real manager,
// scheduler, sorter, and shared resource ledger without a browser.
const vtk = vi.hoisted(() => {
  const instance = (initial: Record<string, any> = {}): any => {
    let data = initial.values ?? new Float32Array();
    let components = initial.numberOfComponents ?? 1;
    let mtime = 0;
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
import { createSubmissionScheduler } from "../core/submissionScheduler";
import { createGaussianContentAdapterFactory } from "./gaussianAdapter";
import { createGaussianDistanceSorter } from "./gaussianSort";
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

const fixture = () => {
  const renderer = { addActor() {}, removeActor() {} };
  const pool = createMemoryPool({
    totalBytes: 1000000,
    cpuBytes: 1000000,
    workBytes: 1000000,
  });
  const memory = pool.createOwner();
  const extraMemory: (typeof memory)[] = [];
  const submissions = createSubmissionScheduler({
    scheduleRender() {},
    now: () => 0,
  });
  const distance = createGaussianDistanceSorter();
  const errors: unknown[] = [],
    refusals: unknown[] = [];
  let held = false;
  const pending: (() => void)[] = [];
  const factory = createGaussianContentAdapterFactory({
    createSortBackend: () => ({
      sort(job) {
        const order = distance.sort(job.ranges, job.cameraPosition);
        return held
          ? new Promise<Uint32Array>((resolve) =>
              pending.push(() => resolve(order)),
            )
          : Promise.resolve(order);
      },
      dispose() {
        for (const release of pending.splice(0)) release();
      },
    }),
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
  adapter.setCamera({
    projection: "perspective",
    fovY: Math.PI / 3,
    position: [0, 0, 10],
    viewProj: IDENTITY,
    viewportWidthCssPx: 100,
    viewportHeightCssPx: 100,
  });
  const settle = async () => {
    while (submissions.hasPending()) submissions.prepareFrame();
    await factory.whenIdle(renderer);
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
    release() {
      held = false;
      for (const release of pending.splice(0)) release();
    },
    async started() {
      for (let index = 0; index < 100 && pending.length === 0; index++)
        await Promise.resolve();
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

describe("Gaussian admission ownership", () => {
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
      expect(f.adapter.tileState("parent")).toBe("queued");
      expect(f.adapter.setResourceCeilingBytes(240)).toEqual([]);
      expect(f.adapter.tileState("parent")).toBe("queued");
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
      expect(f.adapter.tileState("parent")).toBe("queued");
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
      expect(submit()).toBe("budget-blocked");
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
        f.adapter.submitTile("parent", field(1), undefined, ["detail"]),
      ).toBe("queued");
      f.submissions.prepareFrame();
      await f.started();
      expect(f.pool.stats().rendererBytes).toBe(540);
      expect(f.adapter.cancelTile("parent")).toBe(true);
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
