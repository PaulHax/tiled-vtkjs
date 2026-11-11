import {
  createMemoryPool,
  defaultMemoryBudgetBytes,
  type MemoryPool,
  type MemoryPoolOptions,
  type StreamedMemberRegistration,
  type StreamedMember,
} from "../../../src/core";
import { IDENTITY } from "../../../src/core/mat4";
import { createPointCloudMember } from "../../../src/points/vtk";
import { createTiles3dMember } from "../../../src/tiles3d/vtk";
import { createSceneHost } from "../scene/host";
import { decodeWasmUrls } from "../scene/decodeAssets";

const parameters = new URLSearchParams(location.search);
const budget =
  Number(parameters.get("budget") ?? defaultMemoryBudgetBytes() / 1048576) *
  1048576;
const allowances = JSON.parse(parameters.get("allowances") ?? "null") as
  | number[]
  | null;
let registered = 0;
const shared = createMemoryPool({
  totalBytes: budget,
  ...(parameters.has("work")
    ? { workBytes: Number(parameters.get("work")) }
    : {}),
  ...(parameters.has("cpu") ? { cpuBytes: Number(parameters.get("cpu")) } : {}),
  ...(parameters.has("transition")
    ? { transitionBytes: Number(parameters.get("transition")) }
    : {}),
});
const memory: MemoryPool =
  allowances === null
    ? shared
    : {
        ...shared,
        register(onChange) {
          const bytes = allowances[registered++]! * 1048576;
          return { ...shared.register(onChange), budgetBytes: () => bytes };
        },
        memberCount: () => registered,
      };
if (allowances && allowances.reduce((a, b) => a + b, 0) * 1048576 > budget)
  throw new Error("Fixture allowances exceed the aggregate budget");
if (parameters.get("views") === "2")
  document.querySelector<HTMLElement>("#viewer")!.style.height = "45vh";
const host = createSceneHost(
  document.querySelector<HTMLElement>("#viewer")!,
  memory,
);
const canvas = document.querySelector("canvas")!;
const errors: string[] = [];
let contextLost = false;
let restoreContext: (() => void) | undefined;
canvas.addEventListener("webglcontextlost", () => {
  contextLost = true;
  errors.push("webglcontextlost");
});
canvas.addEventListener("webglcontextrestored", () => {
  contextLost = false;
  host.scheduleRender();
});
const timings: { cpu: number[]; gpu: number[] } = { cpu: [], gpu: [] };
host.setFrameProbe({
  onFrame: (frame) => timings.cpu.push(frame.vtkFrameMs),
  onGpuResolved: (frame) => {
    if (frame.status === "valid" && frame.gpuMs !== null)
      timings.gpu.push(frame.gpuMs);
  },
});
const hosts = [host];
if (parameters.get("views") === "2") {
  const viewer = document.createElement("div");
  viewer.id = "viewer-2";
  viewer.style.height = "45vh";
  document.querySelector("#viewer")!.after(viewer);
  document.querySelector<HTMLElement>("#viewer")!.style.height = "45vh";
  const second = createSceneHost(viewer, memory);
  second.lookAt([0, 0, 0], 15);
  hosts.push(second);
}
let nextId = 0;
const delayed: (() => void)[] = [];
const members: {
  id: string;
  asset: string;
  member: StreamedMember;
  registration: StreamedMemberRegistration;
  view: number;
}[] = [];
const add = (asset: string, view = 0): string => {
  const ownerHost = hosts[view]!;
  const context = ownerHost.coordinator.context(
    asset === "allocation-failure"
      ? {
          addActor() {
            throw new RangeError("Array buffer allocation failed (injected)");
          },
          removeActor() {},
        }
      : ownerHost.renderer,
  );
  const member =
    asset === "points" || asset === "delayed-points"
      ? createPointCloudMember(context, {
          source: {
            metadata: () => ({ pointCount: 16 }),
            nodes: async () => [
              {
                key: { level: 0, x: 0, y: 0, z: 0 },
                pointCount: 16,
                bounds: { min: [-1, -1, 0], max: [1, 1, 1] },
                spacing: 1,
                children: [],
              },
            ],
            loadTile: async () => {
              if (asset === "delayed-points")
                await new Promise<void>((resolve) => delayed.push(resolve));
              return {
                origin: [0, 0, 1],
                positions: new Float32Array(
                  Array.from({ length: 48 }, (_, i) =>
                    i % 3 === 2 ? 0 : (i % 12) / 12 - 0.5,
                  ),
                ),
                rgb: new Uint8Array(
                  Array.from({ length: 48 }, (_, i) => (i % 3 === 2 ? 255 : 0)),
                ),
                pointCount: 16,
              };
            },
          },
          adaptive: false,
          pointBudget: 16,
          selectionDelayMs: 0,
          presentation: { mode: "fixed", diameterCssPx: 16 },
        })
      : createTiles3dMember(
          context,
          {
            endpoint: new URL(
              `/memory-fixtures/${asset === "allocation-failure" ? "large" : asset}`,
              location.href,
            ).href,
            revision: asset,
            tilesetToScene: IDENTITY,
            wasm: decodeWasmUrls(),
            ...(parameters.has("cache")
              ? { cacheBytes: Number(parameters.get("cache")) }
              : {}),
            onError: (error) => errors.push(String(error)),
          },
          ownerHost.tiles3dServices,
        );
  const id = `asset-${nextId++}`;
  const registration = ownerHost.coordinator.register(member, {
    id,
    qualityManaged: false,
  });
  members.push({ id, asset, member, registration, view });
  ownerHost.scheduleRender();
  return id;
};
const remove = (id: string): void => {
  const index = members.findIndex((entry) => entry.id === id);
  if (index < 0) return;
  members[index]!.registration.release();
  members.splice(index, 1);
  for (const item of hosts) item.scheduleRender();
};
for (const [index, item] of hosts.entries()) {
  item.onBeforeFrame((view, ratio) => {
    for (const entry of members.filter((entry) => entry.view === index)) {
      entry.registration.setCamera(view);
      entry.registration.setDevicePixelRatio(ratio);
    }
  });
}
for (const asset of parameters.getAll("asset")) add(asset);
host.lookAt([0, 0, 0], 15);
const started = performance.now();
let firstDrawMs: number | null = null;
const snapshot = () => {
  const stats = members.map(
    ({ member }) => member.stats() as Record<string, any>,
  );
  const drawn = stats.reduce(
    (sum, value) =>
      sum +
      (value.renderer.drawnTiles ?? (value.renderer.drawnPoints > 0 ? 1 : 0)),
    0,
  );
  if (drawn > 0 && firstDrawMs === null)
    firstDrawMs = performance.now() - started;
  const settled =
    stats.length > 0 &&
    stats.every(
      (value) =>
        value.active === false ||
        ((value.kind === "pointCloud"
          ? !value.controller.workPending
          : value.sourceState === "ready" || value.sourceState === "failed") &&
          !value.queue?.workPending &&
          (value.renderer.pendingJobs ?? 0) === 0 &&
          (value.irreducibleBudget ||
            (value.renderer.drawnTiles ?? value.renderer.drawnPoints) > 0 ||
            value.resourceState?.status === "blocked" ||
            value.errorCount > 0)),
    );
  return {
    stats,
    members: members.map(({ id, asset, view }) => ({ id, asset, view })),
    memory: shared.stats(),
    settled,
    drawn,
    budget,
    firstDrawMs,
    elapsedMs: performance.now() - started,
    timings,
    errors,
    contextLost,
    paints: host.paintCount(),
    renderer: host.rendererName,
    deviceMemory: (navigator as Navigator & { deviceMemory?: number })
      .deviceMemory,
    heap: (performance as Performance & { memory?: object }).memory,
    coordinator: host.coordinator.stats(),
  };
};
host.onFrame(() => {
  const state = snapshot();
  document.querySelector("#status")!.textContent =
    `Drawn tiles: ${state.drawn}; budget: ${budget / 1048576} MiB; ${state.settled ? "settled" : "working"}`;
});
Object.assign(window, {
  memoryMeasurement: {
    snapshot,
    loseContext() {
      const extension = host.glContext()?.getExtension("WEBGL_lose_context");
      if (!extension) throw new Error("Context loss extension unavailable");
      restoreContext = () => extension.restoreContext();
      extension.loseContext();
    },
    restoreContext() {
      restoreContext?.();
    },
    add,
    remove,
    releaseDelayed() {
      for (const done of delayed.splice(0)) done();
    },
    setActive(id: string, active: boolean) {
      members.find((entry) => entry.id === id)?.registration.setActive(active);
    },
    configure(options: MemoryPoolOptions) {
      shared.configure(options);
      for (const item of hosts) item.scheduleRender();
    },
    pick(x: number, y: number, view = 0) {
      return members
        .filter((entry) => entry.view === view)
        .map((entry) => ({
          id: entry.id,
          result: entry.member.pick(hosts[view]!.cameraView(), x, y),
        }));
    },
    lookAt(distance: number) {
      for (const item of hosts) item.lookAt([0, 0, 0], distance);
    },
    render: host.scheduleRender,
  },
});
host.scheduleRender();

document
  .querySelector("#add")!
  .addEventListener("click", () =>
    add(document.querySelector<HTMLInputElement>("#asset")!.value),
  );
document.querySelector("#remove")!.addEventListener("click", () => {
  const last = members.at(-1);
  if (last) remove(last.id);
});
document.querySelector("#apply-budget")!.addEventListener("click", () => {
  shared.configure({
    totalBytes: Number(
      document.querySelector<HTMLInputElement>("#budget")!.value,
    ),
  });
  for (const item of hosts) item.scheduleRender();
});
