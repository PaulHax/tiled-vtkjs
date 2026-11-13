import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkForwardPass from "@kitware/vtk.js/Rendering/OpenGL/ForwardPass";
import {
  createMemoryPool,
  createSubmissionScheduler,
  createStreamedSceneCoordinator,
} from "__PACKAGE__/core";
import { createPointCloudMember } from "__PACKAGE__/points/vtk";
import { DecodeWorkerPool } from "__PACKAGE__/tiles3d";
import {
  createGaussianAlphaPass,
  createGaussianContentAdapterFactory,
  createTiles3dMember,
} from "__PACKAGE__/tiles3d/vtk";

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const size = 128;
const errors = [];
const generic = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
generic.setContainer(document.querySelector("#view"));
generic.resize();
const renderer = generic.getRenderer();
const renderWindow = generic.getRenderWindow();
const camera = renderer.getActiveCamera();
camera.setParallelProjection(true);
camera.setPosition(0, 0, 2);
camera.setFocalPoint(0, 0, 0);
camera.setViewUp(0, 1, 0);
camera.setParallelScale(1);
camera.setClippingRange(0.01, 10);
renderWindow.render();
const gl = renderWindow.getViews()[0].getContext();
const debugRenderer = gl.getExtension("WEBGL_debug_renderer_info");
const rendererName = debugRenderer
  ? gl.getParameter(debugRenderer.UNMASKED_RENDERER_WEBGL)
  : gl.getParameter(gl.RENDERER);
const matrix = camera.getCompositeProjectionMatrix(1, -1, 1);
const view = {
  projection: "orthographic",
  viewProj: Array.from(
    { length: 16 },
    (_, i) => matrix[(i % 4) * 4 + Math.floor(i / 4)],
  ),
  position: [0, 0, 2],
  parallelScale: 1,
  viewportWidthCssPx: size,
  viewportHeightCssPx: size,
};
const memory = createMemoryPool({ totalBytes: 16_384 });
let renderRequested = false;
const coordinator = createStreamedSceneCoordinator({
  memory,
  scheduleRender() {
    renderRequested = true;
  },
});
const pointContext = coordinator.context(renderer);
const meshContext = coordinator.context(renderer);
const source = {
  metadata: () => ({ pointCount: 1 }),
  nodes: async () => [
    {
      key: { level: 0, x: 0, y: 0, z: 0 },
      pointCount: 1,
      bounds: { min: [-0.7, -0.1, 0], max: [-0.5, 0.1, 0.2] },
      spacing: 0.1,
      children: [],
    },
  ],
  loadTile: async () => ({
    origin: [0, 0, 0],
    positions: new Float32Array([-0.625, 0, 0.1]),
    rgb: new Uint8Array([255, 0, 0]),
    pointCount: 1,
  }),
};
const point = createPointCloudMember(pointContext, {
  source,
  adaptive: false,
  pointBudget: 1,
  selectionDelayMs: 0,
  presentation: { mode: "fixed", diameterCssPx: 9 },
});
const pointRegistration = coordinator.register(point, { id: "points" });

// Author the glTF Y-up coordinates so its decoded surface lies in scene XY.
const glb = (x0, x1, color) => {
  const positions = new Float32Array([
    x0,
    0,
    0.5,
    x1,
    0,
    0.5,
    x1,
    0,
    -0.5,
    x0,
    0,
    -0.5,
  ]);
  const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
  const binary = new Uint8Array(positions.byteLength + indices.byteLength);
  binary.set(new Uint8Array(positions.buffer));
  binary.set(new Uint8Array(indices.buffer), positions.byteLength);
  const document = {
    asset: { version: "2.0" },
    extensionsUsed: ["KHR_materials_unlit"],
    buffers: [{ byteLength: binary.byteLength }],
    bufferViews: [
      {
        buffer: 0,
        byteOffset: 0,
        byteLength: positions.byteLength,
        target: 34962,
      },
      {
        buffer: 0,
        byteOffset: positions.byteLength,
        byteLength: indices.byteLength,
        target: 34963,
      },
    ],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: 4,
        type: "VEC3",
        min: [x0, 0, -0.5],
        max: [x1, 0, 0.5],
      },
      {
        bufferView: 1,
        componentType: 5123,
        count: 6,
        type: "SCALAR",
        min: [0],
        max: [3],
      },
    ],
    materials: [
      {
        pbrMetallicRoughness: { baseColorFactor: color },
        doubleSided: true,
        extensions: { KHR_materials_unlit: {} },
      },
    ],
    meshes: [
      {
        primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }],
      },
    ],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0,
  };
  const json = new TextEncoder().encode(JSON.stringify(document));
  const jsonLength = Math.ceil(json.byteLength / 4) * 4;
  const binaryLength = Math.ceil(binary.byteLength / 4) * 4;
  const result = new ArrayBuffer(12 + 8 + jsonLength + 8 + binaryLength);
  const header = new DataView(result);
  header.setUint32(0, 0x46546c67, true);
  header.setUint32(4, 2, true);
  header.setUint32(8, result.byteLength, true);
  header.setUint32(12, jsonLength, true);
  header.setUint32(16, 0x4e4f534a, true);
  const bytes = new Uint8Array(result);
  bytes.fill(32, 20, 20 + jsonLength);
  bytes.set(json, 20);
  header.setUint32(20 + jsonLength, binaryLength, true);
  header.setUint32(24 + jsonLength, 0x004e4942, true);
  bytes.set(binary, 28 + jsonLength);
  return result;
};
const files = {
  "root.glb": glb(0.1, 0.8, [0, 1, 0, 1]),
  "left.glb": glb(0.1, 0.45, [0, 0, 1, 1]),
  "right.glb": glb(0.45, 0.8, [0, 0, 1, 1]),
};
const tile = (uri, geometricError, children = []) => ({
  geometricError,
  refine: "REPLACE",
  boundingVolume: { box: [0.45, 0, 0, 0.4, 0, 0, 0, 0.6, 0, 0, 0, 0.1] },
  content: { uri },
  children,
});
const requests = [];
const childResponses = [];
const workers = new DecodeWorkerPool({ size: 1 });
const meshConfig = {
  endpoint: `${location.origin}/tiles`,
  revision: "packed-fixture",
  tilesetToScene: identity,
  maximumScreenSpaceErrorPx: 1_000_000,
  concurrency: 3,
  fetchTileset: async () => ({
    ok: true,
    status: 200,
    statusText: "OK",
    json: async () => ({
      asset: { version: "1.1" },
      geometricError: 8,
      root: tile("root.glb", 8, [tile("left.glb", 0), tile("right.glb", 0)]),
    }),
  }),
  fetchContent: async (url) => {
    const file = new URL(url).pathname.split("/").at(-1);
    requests.push(file);
    if (file !== "root.glb")
      await new Promise((resolve) => childResponses.push(resolve));
    return {
      ok: true,
      status: 200,
      statusText: "OK",
      arrayBuffer: async () => files[file].slice(0),
    };
  },
  onError: (error) => errors.push(String(error)),
};
const mesh = createTiles3dMember(meshContext, meshConfig, {
  workers,
  textureCapabilities: { capabilityKey: "packed-rgba", compressedFormats: [] },
});
const meshRegistration = coordinator.register(mesh, { id: "tiles" });
pointRegistration.setCamera(view);
meshRegistration.setCamera(view);
let frame = 0;
let blockerRuns = 0;
const pixel = (x, y) => {
  const color = new Uint8Array(4);
  gl.readPixels(x, size - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, color);
  return [...color];
};
const paint = () => {
  renderRequested = false;
  coordinator.prepareFrame(++frame);
  renderWindow.render();
};
const snapshot = () => ({
  point: point.stats(),
  mesh: mesh.stats(),
  coordinator: coordinator.stats(),
  memory: memory.stats(),
  actors: renderer.getActors().length,
  pixels: { point: pixel(24, 64), mesh: pixel(92, 64) },
  picks: { point: point.pick(view, 24, 64), mesh: mesh.pick(view, 92, 64) },
  requests: [...requests],
  blockerRuns,
  sharedScheduler: pointContext.submissions === meshContext.submissions,
  renderRequested,
  rendererName,
  errors: [...errors],
});
const until = async (predicate, paints = true) => {
  while (true) {
    if (paints) paint();
    if (errors.length) throw new Error(errors.join("\n"));
    if (predicate()) return snapshot();
    await new Promise(requestAnimationFrame);
  }
};
window.packedScene = {
  snapshot,
  root: () =>
    until(
      () =>
        point.stats().renderer.drawnPoints === 1 &&
        mesh.stats().renderer.drawnTileIds.join() === "root",
    ),
  requestRefinement: async () => {
    meshRegistration.setConfig({ ...meshConfig, maximumScreenSpaceErrorPx: 4 });
    return until(() => childResponses.length === 2);
  },
  releaseChildren: async () => {
    meshContext.submissions.enqueue({
      bytes: meshContext.submissions.stats().maxBytesPerFrame,
      run: () => blockerRuns++,
    });
    childResponses.splice(0).forEach((resolve) => resolve());
    return until(() => meshContext.submissions.stats().queuedJobs >= 3, false);
  },
  paint: () => {
    paint();
    return snapshot();
  },
  refined: () =>
    until(() => mesh.stats().renderer.drawnTileIds.join() === "root/0,root/1"),
  releasePoint: () => {
    pointRegistration.release();
    paint();
    return snapshot();
  },
  dispose: () => {
    coordinator.dispose();
    workers.dispose();
    paint();
    const result = snapshot();
    generic.delete();
    return result;
  },
  gaussian: async () => {
    const window = vtkGenericRenderWindow.newInstance({
      background: [0, 0, 0],
    });
    window.setContainer(document.querySelector("#view"));
    window.resize();
    const renderer = window.getRenderer();
    const renderWindow = window.getRenderWindow();
    const openGlWindow = renderWindow.getViews()[0];
    openGlWindow.setRenderPasses([
      vtkForwardPass.newInstance({
        translucentPass: createGaussianAlphaPass(() =>
          openGlWindow.getContext(),
        ),
      }),
    ]);
    const camera = renderer.getActiveCamera();
    camera.setParallelProjection(true);
    camera.setPosition(0, 0, 2);
    camera.setFocalPoint(0, 0, 0);
    camera.setViewUp(0, 1, 0);
    camera.setParallelScale(1);
    camera.setClippingRange(0.01, 10);
    const submissions = createSubmissionScheduler({
      scheduleRender() {},
      maxBytesPerFrame: 1_000_000,
      maxTimeMsPerFrame: 100,
      now: () => 0,
    });
    const factory = createGaussianContentAdapterFactory({
      maximumVisibleSplats: 16,
      resourceBudgetBytes: 16 * 1024 * 1024,
    });
    const owner = () =>
      factory({
        renderer,
        submissions,
        scheduleRender() {},
        onError(error) {
          errors.push(String(error));
        },
      });
    const far = owner();
    const near = owner();
    const gaussian = (positions, color) => ({
      kind: "gaussian",
      positions: new Float32Array(positions),
      rotations: new Float32Array(
        Array.from({ length: positions.length / 3 }, () => [0, 0, 0, 1]).flat(),
      ),
      scales: new Float32Array(positions.map(() => 0.16)),
      opacities: new Float32Array(positions.length / 3).fill(0.5),
      sphericalHarmonics: new Float32Array(
        Array.from({ length: positions.length / 3 }, () =>
          color.map((channel) => (channel - 0.5) / 0.28209479177387814),
        ).flat(),
      ),
      harmonicDegree: 0,
      profile: {
        kernel: "ellipse",
        colorSpace: "srgb_rec709_display",
        projection: "perspective",
        sortingMethod: "cameraDistance",
      },
      modelToScene: new Float64Array(identity),
    });
    const content = (primitives) => ({
      origin: [0, 0, 0],
      primitives,
      byteEstimate: { geometry: 1024, textures: 0 },
    });
    const outcomes = [
      far.submitTile(
        "far",
        content([
          gaussian([-0.45, 0, -0.2, 0.45, 0, -0.2], [1, 0, 0]),
          {
            kind: "mesh",
            positions: new Float32Array([
              0.1, -0.5, 0, 0.8, -0.5, 0, 0.8, 0.5, 0, 0.1, 0.5, 0,
            ]),
            indices: new Uint16Array([0, 1, 2, 0, 2, 3]),
            material: {
              baseColorFactor: [0, 1, 0, 1],
              raw: {
                version: 1,
                kind: "gltf-material",
                alphaMode: "OPAQUE",
                doubleSided: true,
                unlit: true,
                metallicFactor: 0,
                roughnessFactor: 1,
                emissiveFactor: [0, 0, 0],
              },
            },
          },
        ]),
      ),
      near.submitTile("near", content([gaussian([-0.45, 0, 0.2], [0, 0, 1])])),
    ];
    if (outcomes.some((outcome) => outcome !== "queued"))
      throw new Error(`packed Gaussian submission failed: ${outcomes}`);
    far.setCamera(view);
    near.setCamera(view);
    submissions.prepareFrame();
    await factory.whenIdle(renderer);
    far.setDrawnTiles(["far"]);
    near.setDrawnTiles(["near"]);
    await factory.whenIdle(renderer);
    if (errors.length) throw new Error(errors.join("\n"));
    renderWindow.render();
    const gl = renderWindow.getViews()[0].getContext();
    const pixel = (x) => {
      const value = new Uint8Array(4);
      gl.readPixels(x, 64, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, value);
      return [...value];
    };
    const ready = factory.stats(renderer);
    const pixels = { sorted: pixel(35), occluded: pixel(93) };
    camera.setPosition(0, 0, -2);
    const reversedMatrix = camera.getCompositeProjectionMatrix(1, -1, 1);
    const reversedView = {
      ...view,
      position: [0, 0, -2],
      viewProj: Array.from(
        { length: 16 },
        (_, i) => reversedMatrix[(i % 4) * 4 + Math.floor(i / 4)],
      ),
    };
    far.setCamera(reversedView);
    near.setCamera(reversedView);
    await factory.whenIdle(renderer);
    renderWindow.render();
    const reversed = { stats: factory.stats(renderer), pixel: pixel(93) };
    far.dispose();
    near.dispose();
    submissions.dispose();
    renderWindow.render();
    const disposed = {
      actors: renderer.getActors().length,
      stats: factory.stats(renderer),
      queuedJobs: submissions.stats().queuedJobs,
      pixel: pixel(35),
    };
    window.delete();
    return { ready, pixels, reversed, disposed };
  },
};
document.documentElement.dataset.ready = "true";
