import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkForwardPass from "@kitware/vtk.js/Rendering/OpenGL/ForwardPass";
import { withPointTextureUploads } from "@kitware/vtk.js/Rendering/OpenGL/PointGaussianMapper/dataTextures";
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
import { createTiles3dMember } from "../../src/tiles3d/member";

const mode = new URLSearchParams(location.search).get("mode") ?? "shared";
const measureUploads = mode === "budgeted-uploads";
let uploads = [];
let phase = "prepare";
let uploadStateReads = 0;
let externalTexture;
if (measureUploads) {
  const prototype = WebGL2RenderingContext.prototype;
  const getParameter = prototype.getParameter;
  prototype.getParameter = function (parameter) {
    if (phase === "prepare" && parameter === this.PIXEL_UNPACK_BUFFER_BINDING)
      uploadStateReads++;
    return getParameter.call(this, parameter);
  };
  const texSubImage = prototype.texSubImage2D;
  prototype.texSubImage2D = function (...args) {
    if (ArrayBuffer.isView(args[8]))
      uploads.push({
        phase,
        kind: args[6] === this.RED_INTEGER ? "order" : "attributes",
        x: args[2],
        y: args[3],
        count: args[4] * args[5],
        bytes: args[8].byteLength,
      });
    return texSubImage.apply(this, args);
  };
  const bufferData = prototype.bufferData;
  prototype.bufferData = function (...args) {
    if (ArrayBuffer.isView(args[1]))
      uploads.push({ phase, kind: "mesh", bytes: args[1].byteLength });
    return bufferData.apply(this, args);
  };
}
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
let frameQueued = false;
let suspendSubmissions = mode === "member-recovery";
const paintChannel = new MessageChannel();
paintChannel.port1.onmessage = () => {
  frameQueued = false;
  if (suspendSubmissions) return;
  scheduler.prepareFrame();
  renderWindow.render();
};
const scheduleFrame = () => {
  if (frameQueued || measureUploads || suspendSubmissions) return;
  frameQueued = true;
  paintChannel.port2.postMessage(null);
};
const scheduler = createSubmissionScheduler({
  scheduleRender: scheduleFrame,
  maxBytesPerFrame: measureUploads ? 1024 : 1_000_000,
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
const blockedWork = [];
let releaseSortLane;
const blockingOwner = memoryPool.createOwner();
const factory = createGaussianContentAdapterFactory({
  resourceBudgetBytes: mode === "budget" ? 1600 : 131072,
  maximumVisibleSplats: 100,
  ...(measureUploads ? { getOpenGLRenderWindow: () => openGL } : {}),
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
// Reuse decoded payload identities across the cold and warm visibility cycle.
const meshCache = new Map();
const meshCacheReads = [];
let meshCacheLease;
let meshCacheOwner;
if (mode === "mesh-cache-restoration") {
  const texture = {
    kind: "rgba",
    width: 2,
    height: 2,
    colorSpace: "srgb",
    rgba: new Uint8Array([
      0, 180, 220, 255, 0, 230, 180, 255, 0, 230, 180, 255, 0, 180, 220, 255,
    ]),
    sampler: { magFilter: 9728, minFilter: 9728, wrapS: 33071, wrapT: 33071 },
  };
  const patch = (left, right, bottom, top, z) => {
    const primitive = triangle(z, [1, 1, 1, 1], "OPAQUE");
    primitive.positions = new Float32Array([
      left,
      bottom,
      z,
      right,
      bottom,
      z,
      right,
      top,
      z,
      left,
      bottom,
      z,
      right,
      top,
      z,
      left,
      top,
      z,
    ]);
    primitive.uvs = new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]);
    primitive.material.baseColorTexture = texture;
    return {
      origin: [0, 0, 0],
      primitives: [primitive],
      byteEstimate: { geometry: 120, textures: texture.rgba.byteLength },
    };
  };
  meshCache.set("cache-parent", patch(-1, 1, -1, 1, -0.2));
  meshCache.set("cache-a", patch(-0.8, -0.15, -0.55, 0.55, 0));
  meshCache.set("cache-b", patch(0.15, 0.8, -0.55, 0.55, 0));
  meshCacheOwner = memoryPool.createOwner();
  meshCacheLease = meshCacheOwner.retain([
    texture.rgba.buffer,
    ...[...meshCache.values()].flatMap((content) => [
      content.primitives[0].positions.buffer,
      content.primitives[0].uvs.buffer,
    ]),
  ]);
}
const cachedMesh = (id) => {
  meshCacheReads.push(id);
  return meshCache.get(id);
};
const translucentMesh = {
  origin: [0, 0, 0],
  primitives: [
    triangle(0, [0, 0, 1, 1], "OPAQUE"),
    triangle(1, [1, 0, 0, 0.5], "BLEND"),
  ],
  byteEstimate: { geometry: 72, textures: 0 },
};
if (
  mode === "no-camera" ||
  mode === "member-recovery" ||
  mode === "mesh-cache-restoration"
) {
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
} else if (mode === "budgeted-uploads") {
  // The test supplies both frontiers and controls each paint.
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
  suspendSubmissions = mode === "member-recovery";
  while (scheduler.hasPending()) scheduler.prepareFrame();
  await factory.whenIdle(renderer);
  renderWindow.render();
  document.documentElement.dataset.stats = JSON.stringify(
    factory.stats(renderer),
  );
  return factory.stats(renderer);
};
let recoveryMember;
let recoveryConfig;
const recoveryFetches = [];
globalThis.gaussianFixture = {
  stageMixedProbe(id, count, triangleCount, replacement = []) {
    const parts = (Array.isArray(count) ? count : [count]).map((length) =>
      field(
        Array.from({ length }, () => (id === "parent" ? red : blue)),
        id === "parent" ? 0 : 1,
      ),
    );
    const content = {
      ...parts[0],
      primitives: parts.flatMap(({ primitives }) => primitives),
      byteEstimate: {
        geometry: parts.reduce(
          (sum, part) => sum + part.byteEstimate.geometry,
          0,
        ),
        textures: 0,
      },
    };
    const patch = triangle(-1, [0, 1, 0, 1], "OPAQUE");
    patch.positions = Float32Array.from(
      Array.from({ length: triangleCount }, () => [...patch.positions]).flat(),
    );
    content.primitives.push(patch);
    return adapters[0].submitTile(
      id,
      content,
      () => {
        adapters[0].setDrawnTiles([id]);
        for (const old of replacement) adapters[0].retireTile(old);
      },
      replacement,
    );
  },
  addInvisibleProbe(id) {
    return adapters[0].submitTile(id, field([blue]));
  },
  stepMixedProbe() {
    uploads = [];
    uploadStateReads = 0;
    const gl = openGL.getContext();
    phase = "prepare";
    const active = gl.getParameter(gl.ACTIVE_TEXTURE);
    const bound = gl.getParameter(gl.TEXTURE_BINDING_2D);
    const alignment = gl.getParameter(gl.UNPACK_ALIGNMENT);
    if (!externalTexture || !gl.isTexture(externalTexture)) {
      externalTexture = gl.createTexture();
    }
    gl.bindTexture(gl.TEXTURE_2D, externalTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 8);
    factory.noteContext(renderer);
    withPointTextureUploads(openGL, () => scheduler.prepareFrame());
    const stateRestored =
      gl.getParameter(gl.UNPACK_ALIGNMENT) === 8 &&
      gl.getParameter(gl.ACTIVE_TEXTURE) === active &&
      gl.getParameter(gl.TEXTURE_BINDING_2D) === externalTexture;
    gl.bindTexture(gl.TEXTURE_2D, bound && gl.isTexture(bound) ? bound : null);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, alignment);
    phase = "draw";
    renderWindow.render();
    const pixel = new Uint8Array(4);
    gl.readPixels(128, 128, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    return {
      pixel: [...pixel],
      uploads,
      uploadStateReads,
      stateRestored,
      stats: factory.stats(renderer),
      picks: adapters[0].pickableMeshTiles().map(({ id }) => id),
      queued: scheduler.hasPending(),
    };
  },
  async restoreMixedProbeContext() {
    const canvas = openGL.getCanvas();
    const extension = openGL.getContext().getExtension("WEBGL_lose_context");
    if (!extension)
      throw new Error("WebGL context loss simulation is unavailable");
    const before = openGL.getGraphicsResourceGeneration();
    const lost = new Promise((resolve) =>
      canvas.addEventListener("webglcontextlost", resolve, { once: true }),
    );
    extension.loseContext();
    await lost;
    factory.noteContext(renderer);
    // Dispatch must complete before the loss extension allows restoration.
    await new Promise(requestAnimationFrame);
    const restored = new Promise((resolve) =>
      canvas.addEventListener("webglcontextrestored", resolve, { once: true }),
    );
    extension.restoreContext();
    await restored;
    return { before, after: openGL.getGraphicsResourceGeneration() };
  },
  cancelMixedProbe(id) {
    adapters[0].retireTile(id);
    adapters[0].setDrawnTiles(["parent"]);
  },
  async showCachedMeshParent() {
    adapters[0].setVisible(true);
    const outcome = adapters[0].submitTile(
      "cache-parent",
      cachedMesh("cache-parent"),
    );
    await settle();
    adapters[0].setDrawnTiles(["cache-parent"]);
    await settle();
    return outcome;
  },
  async refineCachedMeshesCold() {
    const outcome = adapters[0].submitTileGroup(
      ["cache-a", "cache-b"].map((id) => ({ id, content: cachedMesh(id) })),
      ["cache-parent"],
    );
    await settle();
    adapters[0].setDrawnTiles(["cache-a", "cache-b"]);
    adapters[0].retireTile("cache-parent");
    await settle();
    return outcome;
  },
  async hideCachedMeshes() {
    adapters[0].setVisible(false);
    adapters[0].clearTiles();
    await settle();
  },
  async refineCachedMeshesWarm() {
    const first = adapters[0].submitTile(
      "cache-a",
      cachedMesh("cache-a"),
      undefined,
      ["cache-parent"],
    );
    await settle();
    const retainedAfterFirst = adapters[0].stats().drawnTileIds;
    const second = adapters[0].submitTile(
      "cache-b",
      cachedMesh("cache-b"),
      undefined,
      ["cache-parent"],
    );
    await settle();
    if (
      ["cache-a", "cache-b"].every(
        (id) => adapters[0].tileState(id) === "submitted",
      )
    ) {
      adapters[0].setDrawnTiles(["cache-a", "cache-b"]);
      adapters[0].retireTile("cache-parent");
      await settle();
    }
    return { first, second, retainedAfterFirst };
  },
  cachedMeshState() {
    const adapter = adapters[0];
    const picks = adapter.pickableMeshTiles();
    const pickSet = createMeshPickSet();
    const placement = adapter.presentationMatrix();
    pickSet.replaceDrawn(picks);
    pickSet.setPlacement({ drawn: placement, scene: placement });
    const hits = [80, 176].map((x) => pickSet.pick(view(), x, 128));
    const gap = pickSet.pick(view(), 128, 128);
    pickSet.dispose();
    const stats = adapter.stats();
    return {
      drawn: [...stats.drawnTileIds].sort(),
      picks: picks.map((tile) => tile.id).sort(),
      pendingJobs: stats.pendingJobs,
      pendingTiles: stats.pendingTiles,
      queuedJobs: scheduler.stats().queuedJobs,
      residentBytes: stats.residentBytes,
      residentTextureBytes: stats.residentTextureBytes,
      textureRepresentation: stats.textureRepresentation,
      hits,
      gap,
      camera: {
        position: Array.from(camera.getPosition()),
        focalPoint: Array.from(camera.getFocalPoint()),
        parallelScale: camera.getParallelScale(),
        viewport: Array.from(openGL.getSize()),
      },
      cached: [...meshCache.keys()],
      cacheReads: [...meshCacheReads],
      errors: [...errors],
      refusals: [...refusals],
    };
  },
  setupRecovery() {
    const box = (x, half) => [x, 0, 0, half, 0, 0, 0, half, 0, 0, 0, half];
    recoveryConfig = {
      endpoint: "/recovery",
      revision: "r1",
      tilesetToScene: IDENTITY,
      maximumScreenSpaceErrorPx: 1e9,
      maxAttempts: 1,
      fetchTileset: async () => ({
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => ({
          asset: { version: "1.1" },
          geometricError: 8,
          root: {
            geometricError: 8,
            refine: "REPLACE",
            boundingVolume: { box: box(0, 1) },
            content: { uri: "root.glb" },
            children: [-0.5, 0.5].map((x, i) => ({
              geometricError: 0,
              boundingVolume: { box: box(x, 0.5) },
              content: { uri: `child-${i}.glb` },
            })),
          },
        }),
      }),
      fetchContent: async (url) => {
        recoveryFetches.push(url);
        return {
          ok: true,
          status: 200,
          statusText: "OK",
          arrayBuffer: async () => new ArrayBuffer(1),
        };
      },
    };
    recoveryMember = createTiles3dMember(
      {
        renderer,
        submissions: scheduler,
        memory: memoryPool,
        devicePixelRatio: 1,
        scheduleRender() {},
        onWorkChange() {},
      },
      recoveryConfig,
      {
        textureCapabilities: { capabilityKey: "none", compressedFormats: [] },
        workers: {
          decode(request) {
            const root = request.contentUrl.endsWith("/root.glb");
            const left = request.contentUrl.endsWith("/child-0.glb");
            const mesh = triangle(
              0,
              root ? [1, 0, 0, 1] : left ? [0, 1, 0, 1] : [0, 0, 1, 1],
              "OPAQUE",
            );
            if (!root) {
              const x = left ? -0.5 : 0.5;
              mesh.positions = new Float32Array([
                x - 0.4,
                -0.5,
                0,
                x + 0.4,
                -0.5,
                0,
                x,
                0.5,
                0,
              ]);
            }
            const primitives = root
              ? [mesh, triangle(0, [1, 0, 0, 1], "OPAQUE")]
              : [mesh];
            if (root) {
              primitives[0].positions = new Float32Array([
                -1, -1, 0, 1, -1, 0, 1, 1, 0,
              ]);
              primitives[1].positions = new Float32Array([
                -1, -1, 0, 1, 1, 0, -1, 1, 0,
              ]);
            }
            return {
              promise: Promise.resolve({
                origin: [0, 0, 0],
                primitives,
                byteEstimate: { geometry: primitives.length * 36, textures: 0 },
              }),
              cancel() {},
            };
          },
        },
      },
    );
    recoveryMember.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 120,
      regime: "stationary",
    });
    recoveryMember.setCamera(view());
  },
  refineRecovery() {
    recoveryConfig = { ...recoveryConfig, maximumScreenSpaceErrorPx: 4 };
    recoveryMember.setConfig(recoveryConfig);
  },
  panRecovery(evict) {
    if (evict) {
      recoveryConfig = { ...recoveryConfig, cacheBytes: 0 };
      recoveryMember.setConfig(recoveryConfig);
    }
    camera.setPosition(-0.5, 0, 10);
    camera.setFocalPoint(-0.5, 0, 0);
    camera.setParallelScale(0.4);
    recoveryMember.setCamera(view());
  },
  recoveryStats() {
    return {
      stats: recoveryMember.stats(),
      fetches: [...recoveryFetches],
      work: recoveryMember.governorInputs().work.operations,
      pick: recoveryMember.pick(view(), 128, 128),
    };
  },
  flushRecovery() {
    while (scheduler.hasPending()) scheduler.prepareFrame();
    recoveryMember.prepareFrame();
    renderWindow.render();
  },
  stats: () => factory.stats(renderer),
  submitWithoutCamera(prepared) {
    suspendSubmissions = true;
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
    suspendSubmissions = true;
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
  async setupMixed(childCount = 1) {
    for (const adapter of adapters) adapter.clearTiles();
    await settle();
    const parent = field([red]);
    parent.primitives.push(triangle(-1, [0, 0, 1, 1], "OPAQUE"));
    const child = field(
      Array.from({ length: childCount }, () => ({ color: [0, 1, 0], z: 0 })),
    );
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
  moveMixed(x = 0.6) {
    hold = true;
    const matrix = [...IDENTITY];
    matrix[12] = x;
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
    setCamera([0.1, 0, 10]);
    const content = field(Array.from({ length: 8 }, () => blue));
    adapters[0].submitTile("growth", content);
    while (scheduler.hasPending()) scheduler.prepareFrame();
  },
  stageGrowth(id, count, restoreWork = false) {
    suspendSubmissions = true;
    if (restoreWork) {
      memoryPool.configure({ workBytes: 1048576 });
      for (const adapter of adapters) adapter.clearRequiredBytes();
    }
    return adapters[0].submitTile(
      id,
      field(Array.from({ length: count }, () => blue)),
    );
  },
  cancelGrowth(id) {
    return adapters[0].cancelTile(id);
  },
  selectGrowth(id) {
    hold = true;
    adapters[0].setDrawnTiles([id]);
    renderWindow.render();
  },
  prepareInvisible(ids) {
    const result = adapters[0].submitTileGroup(
      ids.map((id) => ({ id, content: field([blue]) })),
      [],
    );
    while (scheduler.hasPending()) scheduler.prepareFrame();
    return result;
  },
  removeMixed() {
    adapters[0].dispose();
    renderWindow.render();
  },
  errors: () => errors,
  pending: () => pending.length,
  blockWork(kind = "slots") {
    if (kind !== "slots") {
      memoryPool.configure({ workBytes: 2112 });
      blockedWork.push(
        blockingOwner.work(undefined, kind === "bytes" ? 1584 : 1056),
      );
    }
    if (kind !== "bytes") {
      void memoryPool.admission.run({
        signal: new AbortController().signal,
        stage: {
          lane: "sort",
          start: (signal) => {
            const work = blockingOwner.work(signal);
            return new Promise((resolve) => {
              releaseSortLane = () => {
                work.release();
                resolve({ value: undefined });
              };
            });
          },
        },
      });
    }
  },
  releaseWork() {
    for (const work of blockedWork.splice(0)) work.release();
    globalThis.gaussianFixture.releaseWorkSlot();
  },
  releaseWorkSlot() {
    releaseSortLane?.();
    releaseSortLane = undefined;
  },
  releaseNext() {
    const entry = pending.shift();
    entry.resolve(
      distanceSorter.sort(entry.job.ranges, entry.job.cameraPosition),
    );
  },
  releaseWhilePreparing(ids) {
    const preparation = new Promise((resolve) => {
      // Prepare between backend completion and presentation publication.
      queueMicrotask(() =>
        queueMicrotask(() =>
          resolve(globalThis.gaussianFixture.prepareInvisible(ids)),
        ),
      );
    });
    globalThis.gaussianFixture.releaseNext();
    return preparation;
  },
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
    meshCacheLease?.release();
    meshCacheOwner?.close();
    meshCache.clear();
    recoveryMember?.dispose();
    for (const adapter of adapters) adapter.dispose();
    scheduler.dispose();
    paintChannel.port1.close();
    paintChannel.port2.close();
    for (const memory of memoryOwners) memory.close();
    globalThis.gaussianFixture.releaseWork();
    blockingOwner.close();
    generic.delete();
  },
};
if (mode !== "no-camera") setCamera([0, 0, 10]);
await settle();
for (const adapter of adapters) adapter.setDrawnTiles(["field"]);
await settle();
document.documentElement.dataset.ready = "true";
