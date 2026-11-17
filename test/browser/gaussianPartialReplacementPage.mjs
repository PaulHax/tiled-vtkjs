import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkForwardPass from "@kitware/vtk.js/Rendering/OpenGL/ForwardPass";
import { IDENTITY } from "../../src/core/mat4";
import { createMemoryPool } from "../../src/core/memoryPool";
import { createSubmissionScheduler } from "../../src/core/submissionScheduler";
import { createMeshPickSet } from "../../src/tiles3d/meshPicking";
import { createGaussianDistanceSorter } from "../../src/tiles3d/gaussianSort";
import {
  createGaussianAlphaPass,
  createGaussianContentAdapterFactory,
} from "../../src/tiles3d/gaussianAdapter";

const generic = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
generic.setContainer(document.querySelector("#view"));
generic.resize();
const renderer = generic.getRenderer();
const renderWindow = generic.getRenderWindow();
const openGL = generic.getApiSpecificRenderWindow();
openGL.setRenderPasses([
  vtkForwardPass.newInstance({
    translucentPass: createGaussianAlphaPass(() => openGL.getContext()),
  }),
]);
const camera = renderer.getActiveCamera();
camera.setPosition(0, 0, 10);
camera.setFocalPoint(0, 0, 0);
camera.setParallelProjection(true);
camera.setParallelScale(1.2);
camera.setClippingRange(0.1, 100);
const view = () => ({
  projection: "orthographic",
  position: [0, 0, 10],
  parallelScale: 1.2,
  viewportWidthCssPx: 256,
  viewportHeightCssPx: 256,
  viewProj: Array.from(
    { length: 16 },
    (_, index) =>
      camera.getCompositeProjectionMatrix(1, -1, 1)[
        (index % 4) * 4 + Math.floor(index / 4)
      ],
  ),
});
const pool = createMemoryPool({
  totalBytes: 2048,
  transitionBytes: 2048,
  cpuBytes: 32768,
  workBytes: 8192,
});
const memory = pool.createOwner();
let frameQueued = false;
const scheduleFrame = () => {
  if (frameQueued) return;
  frameQueued = true;
  requestAnimationFrame(() => {
    frameQueued = false;
    scheduler.prepareFrame();
    renderWindow.render();
  });
};
const scheduler = createSubmissionScheduler({
  scheduleRender: scheduleFrame,
  now: () => 0,
});
const sorter = createGaussianDistanceSorter();
const errors = [],
  pending = [],
  ledgers = [];
let hold = false,
  started = () => {};
const factory = createGaussianContentAdapterFactory({
  createSortBackend: () => ({
    sort(job) {
      if (!hold)
        return Promise.resolve(sorter.sort(job.ranges, job.cameraPosition));
      return new Promise((resolve, reject) => {
        pending.push({
          resolve: () => resolve(sorter.sort(job.ranges, job.cameraPosition)),
          reject,
        });
        started();
      });
    },
    dispose() {
      for (const entry of pending.splice(0))
        entry.reject(new DOMException("Disposed", "AbortError"));
    },
  }),
});
const adapter = factory({
  renderer,
  memoryPool: pool,
  memory,
  submissions: scheduler,
  scheduleRender() {},
  onError: (error) => errors.push(String(error)),
  onPolicyRefusal: (error) => errors.push(String(error)),
  onAdmissionRefusal: (_ids, error) => errors.push(String(error)),
});
adapter.setCamera(view());
const flush = () => {
  while (scheduler.hasPending()) scheduler.prepareFrame();
};
const settle = async () => {
  flush();
  await factory.whenIdle(renderer);
};
const xA = -0.55,
  xB = 0.55,
  shift = 0.3;
const pixelPosition = (x, y) => [
  Math.round(128 + (x * 256) / 2.4),
  Math.round(128 + (y * 256) / 2.4),
];
const field = (count, x, color, mixed) => ({
  origin: [0, 0, 0],
  byteEstimate: { geometry: count * 56 + (mixed ? 36 : 0), textures: 0 },
  primitives: [
    {
      kind: "gaussian",
      positions: Float32Array.from({ length: count * 3 }, (_, i) =>
        i % 3 === 0 ? x : i % 3 === 1 ? 0.25 : 0,
      ),
      rotations: Float32Array.from({ length: count * 4 }, (_, i) =>
        i % 4 === 3 ? 1 : 0,
      ),
      scales: new Float32Array(count * 3).fill(0.13),
      opacities: new Float32Array(count).fill(0.75),
      harmonicDegree: 0,
      sphericalHarmonics: Float32Array.from(
        { length: count * 3 },
        (_, i) => (color[i % 3] - 0.5) / 0.2820947917738781,
      ),
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
            kind: "mesh",
            positions: Float32Array.of(
              x - 0.16,
              -0.6,
              0,
              x + 0.16,
              -0.6,
              0,
              x,
              -0.25,
              0,
            ),
            material: {
              baseColorFactor: [...color, 1],
              raw: {
                version: 1,
                kind: "gltf-material",
                alphaMode: "OPAQUE",
                alphaCutoff: 0.5,
                doubleSided: false,
                unlit: true,
                metallicFactor: 0,
                roughnessFactor: 1,
                emissiveFactor: [0, 0, 0],
              },
            },
          },
        ]
      : []),
  ],
});
const snapshot = (offset) => {
  renderWindow.render();
  const gl = openGL.getContext();
  const read = (x, y) => {
    const rgba = new Uint8Array(4);
    gl.readPixels(
      ...pixelPosition(x, y),
      1,
      1,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      rgba,
    );
    return Array.from(rgba);
  };
  const picks = adapter.pickableMeshTiles();
  const pickSet = createMeshPickSet();
  pickSet.replaceDrawn(picks);
  const matrix = adapter.presentationMatrix();
  pickSet.setPlacement({ drawn: matrix, scene: matrix });
  const hits = [xA, xB].map((x) => {
    const [px, py] = pixelPosition(x + offset, -0.45);
    return pickSet.pick(view(), px, 256 - py)?.status ?? null;
  });
  pickSet.dispose();
  const ledger = pool.stats();
  ledgers.push(ledger);
  return {
    drawn: adapter.stats().drawnTileIds,
    submitted: adapter.stats().submittedTileIds,
    splats: factory.stats(renderer).displayedSplats,
    grant: adapter.stats().resourceCeilingBytes,
    gpuBytes: factory.stats(renderer).gpuAttributeBytes,
    attributeCapacity: renderer
      .getActors()
      .find((actor) => actor.getMapper().isA("vtkPointGaussianMapper"))
      .getMapper()
      .getInputData()
      .getPoints()
      .getNumberOfPoints(),
    picks: picks.map((tile) => tile.id),
    hits,
    translation: matrix[12],
    a: read(xA + offset, 0.25),
    b: read(xB + offset, 0.25),
    meshA: read(xA + offset, -0.45),
    meshB: read(xB + offset, -0.45),
    ledger,
  };
};
const submit = async (id, content, replaces = []) => {
  const result = adapter.submitTile(id, content, undefined, replaces);
  if (result !== "queued")
    throw new Error(id + " admission returned " + result);
  await settle();
};
window.fixture = {
  async setup(mixed) {
    await submit("oldA", field(1, xA, [1, 0, 0], mixed));
    adapter.setDrawnTiles(["oldA"]);
    await settle();
    await submit("oldB", field(4, xB, [0, 1, 0], mixed));
    adapter.setDrawnTiles(["oldA", "oldB"]);
    await settle();
    adapter.setResourceCeilingBytes(mixed ? 636 : 480);
    const old = snapshot(0);
    await submit("newA", field(4, xA, [0, 0, 1], mixed), ["oldA"]);
    const first = { state: adapter.tileState("newA"), ...snapshot(0) };
    await submit("newB", field(1, xB, [1, 1, 0], mixed), ["oldB"]);
    return {
      old,
      first,
      second: { state: adapter.tileState("newB"), ...snapshot(0) },
    };
  },
  async selectPartial() {
    const waitForSort = new Promise((resolve) => {
      started = resolve;
    });
    hold = true;
    const matrix = [...IDENTITY];
    matrix[12] = shift;
    adapter.setBaseMatrix(matrix);
    adapter.setDrawnTiles(["newA", "oldB"]);
    flush();
    await waitForSort;
    return snapshot(0);
  },
  async release() {
    hold = false;
    for (const entry of pending.splice(0)) entry.resolve();
    await settle();
    return { ...snapshot(shift), errors, ledgers };
  },
  async dispose() {
    const idle = factory.whenIdle(renderer);
    adapter.dispose();
    factory.dispose(renderer);
    memory.close();
    scheduler.dispose();
    await idle;
    renderWindow.render();
    const result = { ...pool.stats(), actors: renderer.getActors().length };
    generic.delete();
    return result;
  },
};
renderWindow.render();
document.documentElement.dataset.ready = "true";
