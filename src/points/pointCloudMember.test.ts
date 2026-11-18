import { describe, expect, it, vi } from "vitest";

import { perspectiveView } from "../../test/helpers";
import { createMemoryPool } from "../core/memoryPool";
import { ROOT_KEY } from "./octree";
import { createHttpTileSource, PCT1_HEADER_BYTES } from "./httpTileSource";
import {
  createPointCloudMember,
  type PointCloudMemberConfig,
  type PointCloudMemberServices,
  type PointCloudMemberStats,
} from "./pointCloudMember";
import { createStreamedSceneCoordinator } from "../core/streamedSceneCoordinator";
import {
  createSubmissionFrameBudget,
  createSubmissionScheduler,
} from "../core/submissionScheduler";
import { createMeshAdapter } from "../tiles3d/meshAdapter";
import type { DecodedMeshTileContent } from "../tiles3d/decode";
import type { StreamedMemberContext } from "../core/streamedMember";
import type { TileSource } from "./tileSource";
import type {
  StubActor,
  StubMapper,
  StubPolyData,
} from "../../test/stubs/vtkStub";

const settle = async (): Promise<void> => {
  await Promise.resolve();
  await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
};

const VIEW = perspectiveView({ position: [0, 0, -1] });

const source = (pointCount = 1_000_000): TileSource => ({
  metadata: () => ({ pointCount }),
  nodes: async () => [
    {
      key: ROOT_KEY,
      pointCount,
      bounds: { min: [-1, -1, -0.5], max: [1, 1, 0.5] },
      spacing: 1,
      children: [],
    },
  ],
  loadTile: async () => ({
    origin: [0, 0, 0],
    positions: new Float32Array(pointCount * 3),
    pointCount,
  }),
});

const makeContext = (memory = createMemoryPool({ totalBytes: 64_000_000 })) => {
  const renderer = {
    addActor: vi.fn(),
    removeActor: vi.fn(),
  };
  const scheduleRender = vi.fn();
  const context: StreamedMemberContext = {
    renderer,
    scheduleRender,
    memory,
    submissions: createSubmissionScheduler({
      scheduleRender,
      maxTimeMsPerFrame: 100,
      now: () => 0,
    }),
    devicePixelRatio: 1,
    onWorkChange: vi.fn(),
  };
  return { context, renderer };
};

const config = (
  overrides: Partial<PointCloudMemberConfig> = {},
): PointCloudMemberConfig => ({
  source: source(),
  pointCount: 1_000_000,
  presentation: { mode: "fixed", diameterCssPx: 2 },
  adaptive: true,
  adaptiveOptions: { minBudget: 1, maxBudget: 1_000_000 },
  selectionDelayMs: 0,
  ...overrides,
});

const preparationView = (
  memory = createMemoryPool({ totalBytes: 16_000 }),
  budget = createSubmissionFrameBudget(160, 100),
) => {
  const { context, renderer } = makeContext(memory);
  let requestPreparation!: () => void;
  const prepare = new Promise<void>((resolve) => {
    requestPreparation = resolve;
  });
  const paint = vi.fn();
  const submissions = createSubmissionScheduler({
    budget,
    schedulePreparation: requestPreparation,
    scheduleRender: paint,
    now: () => 0,
  });
  let queueSize = 0;
  let resolveQueue: (() => void) | undefined;
  const onWorkChange = (): void => {
    if (submissions.stats().queuedJobs >= queueSize) resolveQueue?.();
  };
  const whenQueued = (count: number): Promise<void> => {
    if (submissions.stats().queuedJobs >= count) return Promise.resolve();
    queueSize = count;
    return new Promise((resolve) => {
      resolveQueue = resolve;
    });
  };
  const release = vi.fn();
  const view = {
    getContext: () =>
      ({ isContextLost: () => false }) as WebGL2RenderingContext,
    preparePolygonalGeometry: vi.fn((owner: unknown, actor: unknown) => {
      expect(owner).toBe(renderer);
      const { visibility, mapper } = actor as StubActor;
      expect(visibility).toBe(false);
      const { points } = (mapper as StubMapper).inputData as StubPolyData;
      return { bytes: (points as Float32Array).byteLength, release };
    }),
  };
  const services: PointCloudMemberServices = {
    getOpenGLRenderWindow: () => view,
  };
  const createPoint = (
    sourceConfig = config({ source: source(1), pointCount: 1 }),
  ) => {
    const point = createPointCloudMember(
      { ...context, submissions, scheduleRender: paint, onWorkChange },
      sourceConfig,
      services,
    );
    point.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 1024,
      regime: "stationary",
    });
    point.setCamera(VIEW);
    return point;
  };
  return {
    prepare,
    paint,
    submissions,
    renderer,
    view,
    release,
    createPoint,
    whenQueued,
  };
};

const meshContent: DecodedMeshTileContent = {
  origin: [0, 0, 0],
  primitives: [
    {
      kind: "mesh",
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: {
        baseColorFactor: [1, 1, 1, 1],
        raw: {
          version: 1,
          kind: "gltf-material",
          alphaMode: "OPAQUE",
          alphaCutoff: 0.5,
          doubleSided: false,
          unlit: false,
          metallicFactor: 0,
          roughnessFactor: 1,
          emissiveFactor: [0, 0, 0],
        },
      },
    },
  ],
  byteEstimate: { geometry: 72, textures: 0 },
};

describe("createPointCloudMember", () => {
  it.each([false, true])(
    "shares revision-scoped HTTP payloads across independent views (late=%s)",
    async (late) => {
      const memory = createMemoryPool({ totalBytes: 16_000 });
      const first = preparationView(memory);
      const second = preparationView(memory);
      const buffer = new ArrayBuffer(PCT1_HEADER_BYTES + 12);
      const header = new DataView(buffer);
      header.setUint32(0, 0x31544350, true);
      header.setUint32(4, 1, true);
      const fetch = vi.fn(async (input: string) =>
        input.endsWith(".json")
          ? Response.json({
              nodes: {
                "0-0-0-0": {
                  pointCount: 1,
                  bounds: { min: [-1, -1, -0.5], max: [1, 1, 0.5] },
                  spacing: 1,
                  children: [],
                },
              },
            })
          : new Response(buffer.slice(0)),
      );
      vi.stubGlobal("fetch", fetch);
      const makeSource = () =>
        createHttpTileSource({
          endpoint: "/points/r1",
          metadata: { pointCount: 1 },
        });
      const pointConfig = () => config({ source: makeSource(), pointCount: 1 });
      const draw = async (view: ReturnType<typeof preparationView>) => {
        await view.whenQueued(1);
        view.submissions.prepareFrame(1, false);
        view.submissions.prepareFrame(1, true);
      };
      const a = first.createPoint(pointConfig());
      let b: ReturnType<typeof createPointCloudMember> | undefined;
      try {
        const firstDrawn = draw(first);
        if (late) await firstDrawn;
        b = second.createPoint(pointConfig());
        await firstDrawn;
        await draw(second);
        expect(a.pick(VIEW, 50, 50)?.status).toBe("hit");
        expect(b.pick(VIEW, 50, 50)?.status).toBe("hit");
        expect(fetch.mock.calls.map(([url]) => url)).toEqual([
          "/points/r1/hierarchy/0-0-0-0.json",
          "/points/r1/tile/0-0-0-0.bin",
        ]);
        expect(memory.stats().cpuRetainedBytes).toBe(buffer.byteLength);
        a.dispose();
        expect(first.release).toHaveBeenCalledOnce();
        expect(second.release).not.toHaveBeenCalled();
        expect(b.pick(VIEW, 50, 50)?.status).toBe("hit");
        expect(memory.stats().cpuRetainedBytes).toBe(buffer.byteLength);
      } finally {
        a.dispose();
        b?.dispose();
        first.submissions.dispose();
        second.submissions.dispose();
        vi.unstubAllGlobals();
      }
      expect(memory.stats()).toMatchObject({
        activeOperations: 0,
        cpuRetainedBytes: 0,
        cpuReservedBytes: 0,
        rendererBytes: 0,
      });
    },
  );

  it("shares preparation with mesh and another view, publishing picks only at paint", async () => {
    const memory = createMemoryPool({ totalBytes: 16_000 });
    const budget = createSubmissionFrameBudget(96, 100);
    const first = preparationView(memory, budget);
    const second = preparationView(memory, budget);
    const mesh = createMeshAdapter({
      renderer: first.renderer,
      scheduleRender: () => {},
      submissions: first.submissions,
    });
    expect(
      mesh.submitTile("mesh", meshContent, () => mesh.setDrawnTiles(["mesh"])),
    ).toBe("queued");
    const a = first.createPoint(config({ source: source(5), pointCount: 5 }));
    const b = second.createPoint(config({ source: source(5), pointCount: 5 }));
    // Both decoders have supplied their payloads before the first shared slice.
    await Promise.all([first.whenQueued(2), second.whenQueued(1)]);
    try {
      first.submissions.prepareFrame(1, false);
      second.submissions.prepareFrame(1, false);
      expect(mesh.stats().drawnTileIds).toEqual(["mesh"]);
      expect(first.view.preparePolygonalGeometry).not.toHaveBeenCalled();
      expect(second.view.preparePolygonalGeometry).not.toHaveBeenCalled();
      expect(a.pick(VIEW, 50, 50)).toEqual({ status: "miss" });
      expect(b.pick(VIEW, 50, 50)).toEqual({ status: "miss" });

      first.submissions.prepareFrame(2, false);
      second.submissions.prepareFrame(2, false);
      expect(first.view.preparePolygonalGeometry).not.toHaveBeenCalled();
      expect(second.view.preparePolygonalGeometry).toHaveBeenCalledOnce();
      expect(b.pick(VIEW, 50, 50)).toEqual({ status: "miss" });
      expect((b.stats() as PointCloudMemberStats).renderer).toMatchObject({
        drawnPoints: 0,
        gpuGeometryUploadBytes: 60,
      });
      expect(second.paint).toHaveBeenCalledOnce();
      second.submissions.prepareFrame(2, true);
      expect(second.paint).toHaveBeenCalledOnce();
      expect(b.pick(VIEW, 50, 50)).toMatchObject({
        status: "hit",
        rayDepth: 1,
      });
      expect(a.pick(VIEW, 50, 50)).toEqual({ status: "miss" });

      first.submissions.prepareFrame(3, false);
      first.submissions.prepareFrame(3, true);
      expect(first.view.preparePolygonalGeometry).toHaveBeenCalledOnce();
      expect(a.pick(VIEW, 50, 50)).toMatchObject({
        status: "hit",
        rayDepth: 1,
      });
      a.dispose();
      expect(first.release).toHaveBeenCalledOnce();
      expect(second.release).not.toHaveBeenCalled();
      expect(b.pick(VIEW, 50, 50)).toMatchObject({
        status: "hit",
        rayDepth: 1,
      });
    } finally {
      a.dispose();
      b.dispose();
      mesh.dispose();
      first.submissions.dispose();
      second.submissions.dispose();
    }
    expect(memory.stats()).toMatchObject({
      owners: 0,
      rendererBytes: 0,
      cpuRetainedBytes: 0,
    });
  });

  it("discards hidden uploads on deactivation before they are pickable", async () => {
    const memory = createMemoryPool({ totalBytes: 16_000 });
    const view = preparationView(memory);
    const point = view.createPoint();
    try {
      await view.prepare;
      view.submissions.prepareFrame(1, false);
      expect(view.view.preparePolygonalGeometry).toHaveBeenCalledOnce();
      expect(point.pick(VIEW, 50, 50)).toEqual({ status: "miss" });

      point.setActive(false);
      await Promise.resolve();
      view.submissions.prepareFrame(1, true);
      expect(point.pick(VIEW, 50, 50)).toBeNull();
      expect(view.release).toHaveBeenCalledOnce();
      expect(view.renderer.removeActor).toHaveBeenCalledOnce();
      expect(
        (point.stats() as PointCloudMemberStats).renderer.drawnPoints,
      ).toBe(0);
      expect(memory.stats().rendererBytes).toBe(0);
    } finally {
      point.dispose();
      view.submissions.dispose();
    }
    expect(memory.stats()).toMatchObject({
      owners: 0,
      rendererBytes: 0,
      cpuRetainedBytes: 0,
    });
  });

  it("invalidates a prepared actor synchronously when the source is replaced", async () => {
    const view = preparationView();
    const point = view.createPoint();
    try {
      await view.prepare;
      view.submissions.prepareFrame(1, false);
      expect(view.view.preparePolygonalGeometry).toHaveBeenCalledOnce();
      point.setConfig(config({ source: source(2), pointCount: 2 }));
      expect(view.release).toHaveBeenCalledOnce();
      view.submissions.prepareFrame(1, true);
      expect(
        (point.stats() as PointCloudMemberStats).renderer.drawnPoints,
      ).toBe(0);
      expect(point.pick(VIEW, 50, 50)).toEqual({ status: "miss" });
    } finally {
      point.dispose();
      view.submissions.dispose();
    }
  });

  it("uploads an oversized root in bounded spans and publishes its original progressive prefix", async () => {
    const memory = createMemoryPool({ totalBytes: 16_000 });
    const view = preparationView(memory, createSubmissionFrameBudget(24, 100));
    const count = 7;
    const positions = new Float32Array(count * 3);
    for (let index = 0; index < count; index++)
      positions[index * 3 + 2] = (count - index) / 16;
    const point = view.createPoint(
      config({
        pointCount: count,
        source: {
          ...source(count),
          loadTile: async () => ({
            origin: [0, 0, 0],
            pointCount: count,
            positions,
          }),
        },
      }),
    );
    try {
      await view.prepare;
      for (let frame = 1; frame <= 3; frame++) {
        view.submissions.prepareFrame(frame, false);
        expect(view.submissions.stats().lastFrameAdmittedBytes).toBe(24);
        expect(view.view.preparePolygonalGeometry).toHaveBeenCalledTimes(frame);
        expect(view.paint).not.toHaveBeenCalled();
        expect(point.pick(VIEW, 50, 50)).toEqual({ status: "miss" });
      }
      view.submissions.prepareFrame(4, false);
      expect(view.submissions.stats().lastFrameAdmittedBytes).toBe(12);
      expect(point.pick(VIEW, 50, 50)).toEqual({ status: "miss" });
      view.submissions.prepareFrame(4, true);
      expect(point.pick(VIEW, 50, 50)).toMatchObject({
        status: "hit",
        rayDepth: 1.0625,
      });

      const actors = view.view.preparePolygonalGeometry.mock.calls.map(
        ([, actor]) => actor as StubActor,
      );
      const mappers = actors.map(({ mapper }) => mapper as StubMapper);
      const spans = mappers.map(
        ({ inputData }) => (inputData as StubPolyData).points as Float32Array,
      );
      expect(spans.map(({ length }) => length / 3)).toEqual([2, 2, 2, 1]);
      expect(spans.every(({ buffer }) => buffer === positions.buffer)).toBe(
        true,
      );
      const assembled = spans.flatMap((span) => [...span]);
      expect(assembled).toEqual([...positions]);
      point.applyAllocation({
        qualityFraction: 0.5,
        memoryBudgetBytes: 1024,
        regime: "moving",
      });
      expect(point.pick(VIEW, 50, 50)).toMatchObject({
        status: "hit",
        rayDepth: 1.0625,
      });
      view.submissions.prepareFrame(5, true);
      expect(mappers.map(({ maximumPointCount }) => maximumPointCount)).toEqual(
        [2, 1, 0, 0],
      );
      expect(point.pick(VIEW, 50, 50)).toMatchObject({
        status: "hit",
        rayDepth: 1.3125,
      });
      expect(view.view.preparePolygonalGeometry).toHaveBeenCalledTimes(4);

      point.setActive(false);
      await Promise.resolve();
      view.submissions.prepareFrame(6, true);
      expect(actors.every(({ visibility }) => !visibility)).toBe(true);
      point.setActive(true);
      await Promise.resolve();
      view.submissions.prepareFrame(7, true);
      expect(point.pick(VIEW, 50, 50)).toMatchObject({ status: "hit" });
      expect(view.view.preparePolygonalGeometry).toHaveBeenCalledTimes(4);
    } finally {
      point.dispose();
      view.submissions.dispose();
    }
    expect(view.release).toHaveBeenCalledTimes(4);
    expect(memory.stats()).toMatchObject({
      owners: 0,
      rendererBytes: 0,
      cpuRetainedBytes: 0,
    });
  });

  it("releases previously uploaded spans and the reservation after a later upload fails", async () => {
    const memory = createMemoryPool({ totalBytes: 16_000 });
    const view = preparationView(memory, createSubmissionFrameBudget(24, 100));
    const onError = vi.fn();
    const point = view.createPoint(
      config({ source: source(5), pointCount: 5, onError }),
    );
    try {
      await view.prepare;
      view.submissions.prepareFrame(1, false);
      expect(view.view.preparePolygonalGeometry).toHaveBeenCalledOnce();
      expect(memory.stats().rendererBytes).toBeGreaterThan(0);
      view.view.preparePolygonalGeometry.mockImplementationOnce(() => {
        throw new RangeError("buffer allocation failed");
      });
      view.submissions.prepareFrame(2, false);
      await Promise.resolve();
      expect(view.release).toHaveBeenCalledOnce();
      expect(view.renderer.removeActor).toHaveBeenCalledTimes(2);
      expect(view.view.preparePolygonalGeometry).toHaveBeenCalledTimes(2);
      expect(onError).toHaveBeenCalledOnce();
      expect(point.pick(VIEW, 50, 50)).toEqual({ status: "miss" });
      expect(memory.stats().rendererBytes).toBe(0);
      expect(view.submissions.hasPending()).toBe(false);
    } finally {
      point.dispose();
      view.submissions.dispose();
    }
  });

  it("maps configured maximum and normalized quality to point budget", () => {
    const { context } = makeContext();
    const member = createPointCloudMember(context, config());
    member.applyAllocation({
      qualityFraction: 0.5,
      memoryBudgetBytes: 32_000_000,
      regime: "stationary",
    });
    expect(member.stats()).toMatchObject({
      minimumPointBudget: 1,
      configuredPointCeiling: 1_000_000,
      fullPointCeiling: 1_000_000,
      controller: { pointBudget: 500_000 },
    });
    member.dispose();
  });

  it("reports an unconfigured adaptive point ceiling on the point member", () => {
    const { context } = makeContext();
    const member = createPointCloudMember(
      context,
      config({ adaptiveOptions: { minBudget: 10 } }),
    );
    expect(member.stats()).toMatchObject({
      minimumPointBudget: 10,
      configuredPointCeiling: null,
    });
    member.dispose();
  });

  it("maps the byte ceiling with measured/fallback bytes per point", () => {
    const { context } = makeContext();
    const member = createPointCloudMember(context, config());
    member.applyAllocation({
      qualityFraction: 0.5,
      memoryBudgetBytes: 400_000 * 16,
      regime: "stationary",
    });
    expect((member.stats() as any).controller.memoryCeilingPoints).toBe(
      400_000,
    );
    expect((member.stats() as any).controller.pointBudget).toBe(200_000);
    member.dispose();
  });

  it("retains stationary selection and thins density while moving", () => {
    const { context } = makeContext();
    const member = createPointCloudMember(context, config());
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 32_000_000,
      regime: "stationary",
    });
    member.applyAllocation({
      qualityFraction: 0.25,
      memoryBudgetBytes: 32_000_000,
      regime: "moving",
    });
    expect((member.stats() as any).controller).toMatchObject({
      pointBudget: 1_000_000,
      densityFraction: 0.25,
    });
    member.dispose();
  });

  it("caps normalized allocation at current camera demand", async () => {
    const { context } = makeContext();
    const member = createPointCloudMember(
      context,
      config({ source: source(100_000), pointCount: 1_000_000 }),
    );
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 32_000_000,
      regime: "stationary",
    });
    member.setCamera(VIEW);
    await settle();
    expect(member.governorInputs().qualityDemand).toBeCloseTo(0.1);
    member.applyAllocation({
      qualityFraction: 0.5,
      memoryBudgetBytes: 32_000_000,
      regime: "stationary",
    });
    expect((member.stats() as any).controller.pointBudget).toBe(100_000);
    member.dispose();
  });

  it("reports a tile waiting out a retry as outstanding work", async () => {
    vi.useFakeTimers();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const { context } = makeContext();
      const flaky = source(100);
      let failing = true;
      const member = createPointCloudMember(
        context,
        config({
          source: {
            ...flaky,
            loadTile: (key, options) =>
              failing
                ? Promise.reject(new Error("500"))
                : flaky.loadTile(key, options),
          },
          pointCount: 100,
        }),
      );
      member.applyAllocation({
        qualityFraction: 1,
        memoryBudgetBytes: 32_000_000,
        regime: "stationary",
      });
      member.setCamera(VIEW);
      await vi.advanceTimersByTimeAsync(0);
      expect(member.governorInputs()).toMatchObject({
        physicalTileOperations: 0,
        physicalHierarchyOperations: 0,
        work: { operations: 1 },
      });

      failing = false;
      await vi.advanceTimersByTimeAsync(1_000);
      context.submissions.prepareFrame();
      expect(member.governorInputs().work.operations).toBe(0);
      member.dispose();
    } finally {
      warn.mockRestore();
      vi.useRealTimers();
    }
  });

  it("draws a visible cloud whose root alone exceeds the minimum budget", async () => {
    const rootPoints = 250_000;
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 512 * 1024 * 1024 }),
    });
    const member = createPointCloudMember(
      coordinator.context({ addActor: vi.fn(), removeActor: vi.fn() }),
      config({
        source: {
          ...source(rootPoints),
          metadata: () => ({ pointCount: rootPoints + 50_000 }),
          nodes: async () => [
            {
              key: ROOT_KEY,
              pointCount: rootPoints,
              bounds: { min: [-1, -1, -0.5], max: [1, 1, 0.5] },
              spacing: 1,
              children: [{ level: 1, x: 0, y: 0, z: 0 }],
            },
            {
              key: { level: 1, x: 0, y: 0, z: 0 },
              pointCount: 50_000,
              bounds: { min: [-1, -1, -0.5], max: [0, 0, 0] },
              spacing: 0.5,
              children: [],
            },
          ],
        },
        adaptive: true,
        selectionDelayMs: 0,
      }),
    );
    coordinator
      .register(member, { id: "cloud", qualityManaged: true })
      .setCamera(VIEW);
    for (let frame = 1; frame <= 10; frame += 1) {
      await settle();
      coordinator.prepareFrame(frame);
      coordinator.recordHostFrame({ hostFrameMs: 10 });
    }
    const stats = member.stats() as PointCloudMemberStats;
    expect(stats.controller.residentTiles).toBeGreaterThan(0);
    coordinator.dispose();
  });

  it("delegates submitted-point picking and positive occlusion depth", async () => {
    const { context } = makeContext();
    const member = createPointCloudMember(
      context,
      config({ source: source(1), pointCount: 1 }),
    );
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 1024,
      regime: "stationary",
    });
    member.setCamera(VIEW);
    await settle();
    expect(member.pick(VIEW, 50, 50)).toEqual({ status: "miss" });
    context.submissions.prepareFrame();
    const pick = member.pick(VIEW, 50, 50);
    expect(pick).toMatchObject({ status: "hit", rayDepth: 1 });
    expect(member.occlusionDepth(VIEW, 50, 50)).toEqual({
      status: "hit",
      rayDepth: 1,
    });
    member.dispose();
  });

  it("never self-registers with the page memory pool", () => {
    const memory = createMemoryPool({ totalBytes: 1_000 });
    const { context } = makeContext(memory);
    const member = createPointCloudMember(context, config());
    expect(memory.memberCount()).toBe(0);
    member.dispose();
  });

  it("keeps displayed points and a persistent refusal when a lower limit cannot fit their root", async () => {
    const memory = createMemoryPool({ totalBytes: 1024 });
    const coordinator = createStreamedSceneCoordinator({
      memory,
      scheduleRender: vi.fn(),
    });
    const onResourceState = vi.fn();
    const member = createPointCloudMember(
      coordinator.context({ addActor: vi.fn(), removeActor: vi.fn() }),
      config({
        source: source(1),
        pointCount: 1,
        pointBudget: 1,
        adaptive: false,
        onResourceState,
      }),
    );
    coordinator.register(member).setCamera(VIEW);
    await settle();
    coordinator.prepareFrame(1);
    await settle();
    coordinator.prepareFrame(2);
    expect((member.stats() as PointCloudMemberStats).renderer.drawnPoints).toBe(
      1,
    );
    memory.configure({ totalBytes: 50 });
    coordinator.prepareFrame(3);
    await settle();
    coordinator.prepareFrame(4);
    expect(member.stats()).toMatchObject({
      resourceState: { status: "blocked", domain: "renderer" },
      renderer: { drawnPoints: 1 },
    });
    memory.configure({ totalBytes: 1024 });
    coordinator.prepareFrame(5);
    await settle();
    coordinator.prepareFrame(6);
    expect(member.stats()).toMatchObject({
      resourceState: { status: "ready" },
      renderer: { drawnPoints: 1 },
    });
    expect(onResourceState).toHaveBeenLastCalledWith({ status: "ready" });
    coordinator.dispose();
  });

  it("integrates coordinator quality, memory, live DPR, and config lifecycle", async () => {
    const memory = createMemoryPool({ totalBytes: 16_000 });
    const scheduleRender = vi.fn();
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender,
      memory,
    });
    const initial = config({
      source: source(1_000),
      pointCount: 1_000,
      adaptive: false,
      pointBudget: 1_000,
    });
    const member = createPointCloudMember(
      coordinator.context(renderer, 1),
      initial,
    );
    const registration = coordinator.register(member, {
      id: "cloud",
      qualityManaged: false,
    });
    registration.setCamera(VIEW);
    await settle();
    coordinator.prepareFrame(1);
    let stats = member.stats() as PointCloudMemberStats;
    expect(memory.memberCount()).toBe(1);
    expect(stats.allocation).toMatchObject({
      qualityFraction: 1,
      memoryBudgetBytes: 15_064,
    });
    expect(stats.controller).toMatchObject({
      pointBudget: 1_000,
      memoryCeilingPoints: 1_000,
    });

    registration.setDevicePixelRatio(2);
    registration.setConfig({
      ...initial,
      presentation: { mode: "fixed", diameterCssPx: 5 },
      refinementCutoffPx: 2,
    });
    stats = member.stats() as PointCloudMemberStats;
    expect(stats.renderer.devicePixelRatio).toBe(2);
    expect(stats.controller.presentation).toMatchObject({
      diameterCssPx: 5,
    });
    expect(stats.controller.refinementCutoffPx).toBe(2);

    registration.setActive(false);
    expect(memory.memberCount()).toBe(0);
    expect((member.stats() as PointCloudMemberStats).controller.active).toBe(
      false,
    );
    coordinator.dispose();
  });
});
