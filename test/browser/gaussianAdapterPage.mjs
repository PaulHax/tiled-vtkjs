import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkForwardPass from "@kitware/vtk.js/Rendering/OpenGL/ForwardPass";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";

import { createSubmissionScheduler } from "../../src/core/submissionScheduler";
import { IDENTITY } from "../../src/core/mat4";
import { createMeshPickSet } from "../../src/tiles3d/meshPicking";
import { createMemoryPool } from "../../src/core/memoryPool";
import {
  createGaussianAlphaPass,
  createGaussianContentAdapterFactory,
} from "../../src/tiles3d/gaussianAdapter";
import { createGaussianDistanceSorter } from "../../src/tiles3d/gaussianSort";

const mode = new URLSearchParams(location.search).get("mode") ?? "shared";
const generic = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
generic.setContainer(document.querySelector("#view"));
generic.resize();
const renderer = generic.getRenderer();
const renderWindow = generic.getRenderWindow();
const openGL = generic.getApiSpecificRenderWindow();
if (mode !== "mesh-blend" && mode !== "blend-before-gaussian")
  openGL.setRenderPasses([
    vtkForwardPass.newInstance({
      translucentPass: createGaussianAlphaPass(() => openGL.getContext()),
    }),
  ]);
const camera = renderer.getActiveCamera();
camera.setPosition(0, 0, 10);
camera.setFocalPoint(0, 0, 0);
camera.setViewUp(0, 1, 0);
camera.setParallelProjection(true);
camera.setParallelScale(1.2);
camera.setClippingRange(0.1, 100);
const scheduler = createSubmissionScheduler({
  scheduleRender() {},
  maxBytesPerFrame: 1_000_000,
  now: () => 0,
});
const errors = [];
const refusals = [];
const admissionRefusals = [];
const memoryPool = createMemoryPool({
  totalBytes: mode === "budget" ? 300 : mode === "group-budget" ? 360 : 131072,
  cpuBytes: 1048576,
  workBytes: 1048576,
  transitionBytes:
    mode === "budget" ? 0 : mode === "group-budget" ? 200 : 32768,
});
const memoryOwners = [];
const distanceSorter = createGaussianDistanceSorter();
let hold = false;
const pending = [];
const factory = createGaussianContentAdapterFactory({
  resourceBudgetBytes: mode === "budget" ? 1600 : 131072,
  maximumVisibleSplats: 100,
  createSortBackend: () => ({
    sort(job) {
      if (hold)
        return new Promise((resolve, reject) =>
          pending.push({ job, resolve, reject }),
        );
      return Promise.resolve(
        distanceSorter.sort(job.ranges, job.cameraPosition),
      );
    },
    dispose() {
      for (const entry of pending.splice(0))
        entry.reject(new Error("disposed"));
    },
  }),
});
const makeAdapter = () => {
  const memory = memoryPool.createOwner();
  memoryOwners.push(memory);
  return factory({
    realizationRenderer: renderer,
    memoryPool,
    memory,
    onPolicyRefusal: (error) => refusals.push(String(error)),
    onAdmissionRefusal: (ids, error) =>
      admissionRefusals.push({ ids: [...ids], message: String(error) }),
    renderer:
      mode === "wrapped"
        ? {
            addActor: (actor) => renderer.addActor(actor),
            removeActor: (actor) => renderer.removeActor(actor),
          }
        : renderer,
    scheduleRender() {},
    submissions: scheduler,
    onError: (error) => errors.push(String(error)),
  });
};
const adapters = [makeAdapter(), makeAdapter()];
const view = () => ({
  projection: "orthographic",
  position: Array.from(camera.getPosition()),
  viewProj: Array.from(
    { length: 16 },
    (_, index) =>
      camera.getCompositeProjectionMatrix(1, -1, 1)[
        (index % 4) * 4 + Math.floor(index / 4)
      ],
  ),
  viewportWidthCssPx: 256,
  viewportHeightCssPx: 256,
  parallelScale: 1.2,
});
const field = (entries, degree = 0) => {
  const coefficients = (degree + 1) ** 2;
  const sphericalHarmonics = new Float32Array(
    entries.length * coefficients * 3,
  );
  entries.forEach((entry, index) => {
    for (let channel = 0; channel < 3; channel++)
      sphericalHarmonics[index * coefficients * 3 + channel] =
        (entry.color[channel] - 0.5) / 0.2820947917738781;
  });
  if (degree) sphericalHarmonics[6] = 1;
  return {
    origin: [0, 0, 0],
    primitives: [
      {
        kind: "gaussian",
        positions: Float32Array.from(
          entries.flatMap((entry) => [0, 0, entry.z]),
        ),
        rotations: Float32Array.from(entries.flatMap(() => [0, 0, 0, 1])),
        scales: Float32Array.from(entries.flatMap(() => [0.4, 0.4, 0.4])),
        opacities: Float32Array.from(entries.map(() => 0.5)),
        sphericalHarmonics,
        harmonicDegree: degree,
        modelToScene: new Float64Array(IDENTITY),
        profile: {
          kernel: "ellipse",
          colorSpace: "srgb_rec709_display",
          projection: "perspective",
          sortingMethod: "cameraDistance",
        },
      },
    ],
    byteEstimate: {
      geometry: entries.length * (44 + coefficients * 12),
      textures: 0,
    },
  };
};
const red = { color: [1, 0, 0], z: 0 },
  green = { color: [0, 1, 0], z: 1 },
  blue = { color: [0, 0, 1], z: 2 };
const triangle = (z, color, alphaMode) => ({
  kind: "mesh",
  positions: new Float32Array([-1, -1, z, 1, -1, z, 0, 1, z]),
  material: {
    baseColorFactor: color,
    raw: {
      version: 1,
      kind: "gltf-material",
      alphaMode,
      alphaCutoff: 0.5,
      doubleSided: false,
      unlit: true,
      metallicFactor: 0,
      roughnessFactor: 1,
      emissiveFactor: [0, 0, 0],
    },
  },
});
const translucentMesh = {
  origin: [0, 0, 0],
  primitives: [
    triangle(0, [0, 0, 1, 1], "OPAQUE"),
    triangle(1, [1, 0, 0, 0.5], "BLEND"),
  ],
  byteEstimate: { geometry: 72, textures: 0 },
};
if (mode === "no-camera") {
} else if (mode === "blend") {
  const content = field([red]);
  content.primitives.push(triangle(1, [1, 1, 1, 0.5], "BLEND"));
  adapters[0].submitTile("field", content);
} else if (mode === "group-blend-gaussian") {
  adapters[0].submitTileGroup(
    [
      { id: "field", content: field([red]) },
      { id: "mesh", content: translucentMesh },
    ],
    [],
  );
} else if (mode === "mesh-blend" || mode === "blend-before-gaussian") {
  adapters[1].submitTile("field", translucentMesh);
  if (mode === "blend-before-gaussian")
    adapters[0].submitTile("field", field([red]));
} else if (mode === "gaussian-only" || mode === "gaussian-before-blend") {
  adapters[0].submitTile("field", field([red]));
  if (mode === "gaussian-before-blend")
    adapters[1].submitTile("field", translucentMesh);
} else if (mode === "sh")
  adapters[0].submitTile("field", field([{ color: [0.5, 0.5, 0.5], z: 0 }], 1));
else if (mode === "single")
  adapters[0].submitTile("field", field([red, blue, green]));
else {
  adapters[0].submitTile("field", field([red, blue]));
  adapters[1].submitTile("field", field([green]));
}
if (mode === "occlusion") {
  const poly = vtkPolyData.newInstance();
  poly
    .getPoints()
    .setData(new Float32Array([-2, -2, 4, 0, -2, 4, 0, 2, 4, -2, 2, 4]), 3);
  poly.getPolys().setData(new Uint32Array([3, 0, 1, 2, 3, 0, 2, 3]));
  const mapper = vtkMapper.newInstance();
  mapper.setInputData(poly);
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  actor.getProperty().setColor(0.25, 0.25, 0.25);
  actor.getProperty().setLighting(false);
  renderer.addActor(actor);
}
const setCamera = (position) => {
  camera.setPosition(...position);
  for (const adapter of adapters) adapter.setCamera(view());
};
const settle = async () => {
  while (scheduler.hasPending()) scheduler.prepareFrame();
  await factory.whenIdle(renderer);
  renderWindow.render();
  document.documentElement.dataset.stats = JSON.stringify(
    factory.stats(renderer),
  );
  return factory.stats(renderer);
};
globalThis.gaussianFixture = {
  stats: () => factory.stats(renderer),
  submitWithoutCamera(prepared) {
    const content = field([red]);
    content.primitives.push(triangle(-1, [0, 0, 1, 1], "OPAQUE"));
    const outcome = adapters[0].submitTile("unplaced", content);
    if (prepared) while (scheduler.hasPending()) scheduler.prepareFrame();
    return outcome;
  },
  cancelWithoutCamera(clear) {
    if (clear) adapters[0].clearTiles();
    else adapters[0].cancelTile("unplaced");
  },
  async drawUnplaced() {
    adapters[0].setDrawnTiles(["unplaced"]);
    await settle();
  },
  ledger: () => memoryPool.stats(),
  setWorkLimit(bytes) {
    memoryPool.configure({ workBytes: bytes });
    for (const adapter of adapters) adapter.clearRequiredBytes();
  },
  refusals: () => refusals,
  admissionRefusals: () => admissionRefusals,
  flushUploads() {
    while (scheduler.hasPending()) scheduler.prepareFrame();
  },
  replacementGaussian() {
    return adapters[0].submitTile(
      "replacement",
      field([green, green]),
      () => {
        hold = true;
        adapters[0].setDrawnTiles(["replacement"]);
        adapters[0].retireTile("parent");
      },
      ["parent"],
    );
  },
  presentation: () => {
    const adapter = adapters[0],
      picks = adapter.pickableMeshTiles(),
      placement = adapter.presentationMatrix();
    const pickSet = createMeshPickSet();
    pickSet.replaceDrawn(picks);
    pickSet.setPlacement({ drawn: placement, scene: placement });
    const hit = pickSet.pick(view(), 128, 128);
    pickSet.dispose();
    return {
      stats: adapter.stats(),
      picks: picks.map((tile) => tile.id),
      placement: Array.from(placement),
      hit,
    };
  },
  async setupDemand(gaussian) {
    for (const adapter of adapters) adapter.clearTiles();
    await settle();
    adapters[0].setResourceCeilingBytes(4096);
    const content = gaussian
      ? field([red])
      : {
          origin: [0, 0, 0],
          primitives: [],
          byteEstimate: { geometry: 0, textures: 0 },
        };
    content.primitives.push(triangle(-1, [0, 0, 1, 1], "OPAQUE"));
    adapters[0].submitTile("parent", content);
    await settle();
    adapters[0].setDrawnTiles(["parent"]);
    await settle();
  },
  queueDemand(gaussian) {
    let ready = 0;
    const entries = ["left", "right"].map((id) => {
      const content = gaussian
        ? field([green])
        : {
            origin: [0, 0, 0],
            primitives: [],
            byteEstimate: { geometry: 0, textures: 0 },
          };
      content.primitives.push(triangle(-1, [0, 1, 0, 1], "OPAQUE"));
      return {
        id,
        content,
        onSubmitted() {
          if (++ready === 2) {
            adapters[0].setDrawnTiles(["left", "right"]);
            adapters[0].retireTile("parent");
          }
        },
      };
    });
    return adapters[0].submitTileGroup(entries, ["parent"]);
  },
  additionalGaussian() {
    return adapters[0].submitTile("additional", field([blue, blue]), () =>
      adapters[0].setDrawnTiles(["left", "right", "additional"]),
    );
  },
  demandState() {
    return {
      required: adapters[0].requiredBytes(),
      renderer: adapters[0].stats(),
      queue: scheduler.stats(),
    };
  },
  reduceDemand() {
    return adapters[0].setResourceCeilingBytes(
      adapters[0].workState().residentBytes,
    );
  },
  raiseDemand(bytes) {
    adapters[0].setResourceCeilingBytes(bytes);
    adapters[0].clearRequiredBytes();
  },
  async setupGroup() {
    for (const adapter of adapters) adapter.clearTiles();
    await settle();
    const parent = field([red]);
    parent.primitives.push(triangle(-1, [0, 0, 1, 1], "OPAQUE"));
    adapters[0].submitTile("parent", parent);
    await settle();
    adapters[0].setDrawnTiles(["parent"]);
    await settle();
  },
  async admitGroup() {
    let ready = 0;
    const entries = [
      { id: "left", entry: green },
      { id: "right", entry: blue },
    ].map(({ id, entry }) => {
      const content = field([entry]);
      content.primitives.push(triangle(-1, [1, 1, 1, 1], "OPAQUE"));
      return {
        id,
        content,
        onSubmitted() {
          if (++ready === 2) {
            adapters[0].setDrawnTiles(["left", "right"]);
            adapters[0].retireTile("parent");
          }
        },
      };
    });
    const outcome = adapters[0].submitTileGroup(entries, ["parent"]);
    await settle();
    return outcome;
  },
  async setupMixed() {
    for (const adapter of adapters) adapter.clearTiles();
    await settle();
    const parent = field([red]);
    parent.primitives.push(triangle(-1, [0, 0, 1, 1], "OPAQUE"));
    const child = field([{ color: [0, 1, 0], z: 0 }]);
    child.primitives.push(triangle(-1, [1, 1, 1, 1], "OPAQUE"));
    adapters[0].submitTile("parent", parent);
    await settle();
    adapters[0].setDrawnTiles(["parent"]);
    await settle();
    adapters[0].submitTile("child", child, undefined, ["parent"]);
    await settle();
  },
  selectMixed() {
    hold = true;
    adapters[0].setDrawnTiles(["child"]);
    renderWindow.render();
  },
  moveMixed() {
    hold = true;
    const matrix = [...IDENTITY];
    matrix[12] = 0.6;
    adapters[0].setBaseMatrix(matrix);
    renderWindow.render();
  },
  hideMixed() {
    adapters[0].setVisible(false);
    renderWindow.render();
  },
  showMixed() {
    hold = true;
    adapters[0].setVisible(true);
    renderWindow.render();
  },
  clearMixed() {
    adapters[0].clearTiles();
    renderWindow.render();
  },
  growMixed() {
    hold = true;
    const content = field(Array.from({ length: 8 }, () => blue));
    adapters[0].submitTile("growth", content);
    while (scheduler.hasPending()) scheduler.prepareFrame();
  },
  removeMixed() {
    adapters[0].dispose();
    renderWindow.render();
  },
  errors: () => errors,
  pending: () => pending.length,
  hold: () => {
    hold = true;
  },
  release() {
    hold = false;
    for (const entry of pending.splice(0))
      entry.resolve(
        distanceSorter.sort(entry.job.ranges, entry.job.cameraPosition),
      );
  },
  setCamera,
  settle,
  async moveRepeatedly(count) {
    for (let index = 1; index <= count; index++) {
      const matrix = [...IDENTITY];
      matrix[12] = index * 0.001;
      adapters[0].setBaseMatrix(matrix);
      await Promise.resolve();
    }
    return factory.stats(renderer);
  },
  dispose() {
    for (const adapter of adapters) adapter.dispose();
    scheduler.dispose();
    for (const memory of memoryOwners) memory.close();
    generic.delete();
  },
};
if (mode !== "no-camera") setCamera([0, 0, 10]);
await settle();
for (const adapter of adapters) adapter.setDrawnTiles(["field"]);
await settle();
document.documentElement.dataset.ready = "true";
