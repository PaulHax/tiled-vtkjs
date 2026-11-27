const container = () => document.querySelector("#viewer");
const vector = (T, values) => new T.Vector3(...values);
const verticalFov = 30;
const mib = 1024 * 1024;

export async function threeViewer(config) {
  const T = await import("three");
  const renderer = new T.WebGLRenderer({ antialias: false });
  renderer.setSize(innerWidth, innerHeight);
  container().appendChild(renderer.domElement);
  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(
    verticalFov,
    innerWidth / innerHeight,
    0.01,
    1e8,
  );
  scene.add(new T.AmbientLight(0xffffff, 3));

  let update = () => {};
  let stats = () => ({});
  if (config.viewer === "three-tiles") {
    const { TilesRenderer } = await import("3d-tiles-renderer");
    const tiles = new TilesRenderer(config.url);
    if (config.format === "3d-tiles") {
      const [{ DRACOLoader }, { KTX2Loader }, { GLTFExtensionsPlugin }] =
        await Promise.all([
          import("three/addons/loaders/DRACOLoader.js"),
          import("three/addons/loaders/KTX2Loader.js"),
          import("3d-tiles-renderer/plugins"),
        ]);
      const dracoLoader = new DRACOLoader(tiles.manager).setDecoderPath(
        "/tiles3d-codecs/",
      );
      const ktxLoader = new KTX2Loader(tiles.manager)
        .setTranscoderPath("/three-basis/")
        .detectSupport(renderer);
      tiles.registerPlugin(
        new GLTFExtensionsPlugin({ dracoLoader, ktxLoader }),
      );
    }
    tiles.errorTarget = config.sse ?? 8;
    if (config.loadAncestors !== undefined)
      tiles.loadAncestors = config.loadAncestors;
    if (config.memoryMiB !== undefined) {
      tiles.lruCache.maxBytesSize = config.memoryMiB * mib;
      tiles.lruCache.minBytesSize =
        (config.retentionFraction ?? 0.75) * tiles.lruCache.maxBytesSize;
    }
    tiles.setCamera(camera);
    tiles.setResolutionFromRenderer(camera, renderer);
    if (config.format === "gaussian-3d-tiles") {
      const [
        { GaussianSplatRenderer },
        { GaussianSplatPlugin },
        { TilesFadePlugin },
      ] = await Promise.all([
        import("gaussian-splat-lite"),
        import("3d-tiles-rendererjs-3dgs-plugin"),
        import("3d-tiles-renderer/plugins"),
      ]);
      scene.add(new GaussianSplatRenderer({ renderer }));
      tiles.registerPlugin(new TilesFadePlugin());
      tiles.registerPlugin(new GaussianSplatPlugin());
    }
    if (config.format === "pnts" && config.pointSize !== undefined) {
      tiles.addEventListener("load-model", ({ scene: model }) =>
        model.traverse((object) => {
          if (!object.isPoints) return;
          object.material.size = config.pointSize;
          object.material.sizeAttenuation = false;
          object.material.needsUpdate = true;
        }),
      );
    }
    let tileError;
    tiles.addEventListener("load-error", (event) => {
      tileError = event.error;
    });
    scene.add(tiles.group);
    update = () => {
      if (tileError) throw tileError;
      tiles.update();
    };
    stats = () => ({
      drawnPoints: renderer.info.render.points,
      drawCalls: renderer.info.render.calls,
      selected: [...tiles.visibleTiles]
        .map((tile) => tile.content?.uri ?? tile.content?.url)
        .filter(Boolean)
        .sort(),
      ready: tiles.visibleTiles.size > 0,
      settled: tiles.loadProgress === 1,
      residentBytes: tiles.lruCache.cachedBytes,
      residentLimitBytes: tiles.lruCache.maxBytesSize,
      pressure: tiles.lruCache.isFull(),
      residentTiles: tiles.lruCache.itemList.length,
    });
  } else if (config.viewer === "three-mesh") {
    const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
    const model = (await new GLTFLoader().loadAsync(config.url)).scene;
    model.rotateX(Math.PI / 2);
    scene.add(model);
    stats = () => ({ ready: true });
  } else if (config.viewer === "spark") {
    const { SparkRenderer, SplatMesh } = await import("@sparkjsdev/spark");
    scene.add(
      new SparkRenderer({ renderer, lodSplatCount: config.splatBudget }),
    );
    const paged = config.format === "rad";
    const splats = new SplatMesh({
      url: config.url,
      ...(paged ? { paged } : { lod: true }),
    });
    scene.add(splats);
    await splats.initialized;
    stats = () => ({ ready: true });
  } else if (config.viewer === "potree") {
    const { Potree } = await import("potree-core");
    const url = new URL(config.url, location.href);
    const potree = new Potree();
    if (config.pointBudget) potree.pointBudget = config.pointBudget;
    const cloud = await potree.loadPointCloud(
      url.pathname.split("/").pop(),
      new URL(".", url).href,
    );
    scene.add(cloud);
    update = () => potree.updatePointClouds([cloud], camera, renderer);
    stats = () => ({
      ready: cloud.visibleNodes.length > 0,
      selected: cloud.visibleNodes.map((node) => node.name).sort(),
    });
  }

  return {
    canvas: renderer.domElement,
    pose({ position, target, up }) {
      camera.position.copy(vector(T, position));
      camera.up.copy(vector(T, up));
      camera.lookAt(vector(T, target));
      camera.updateMatrixWorld();
    },
    draw() {
      update();
      renderer.render(scene, camera);
    },
    stats,
  };
}

async function deckTilesViewer(config) {
  const [{ Deck, FirstPersonView }, { Tile3DLayer }, { Matrix4 }] =
    await Promise.all([
      import("@deck.gl/core"),
      import("@deck.gl/geo-layers"),
      import("@math.gl/core"),
    ]);
  // The benchmark camera and local tiles share an east-north-up frame at 0, 0.
  const modelMatrix = new Matrix4([
    0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 6378137, 0, 0, 1,
  ]);
  let tileset;
  let failure;
  const deck = new Deck({
    parent: container(),
    controller: false,
    useDevicePixels: false,
    views: new FirstPersonView({ fovy: verticalFov, near: 0.01, far: 1e8 }),
    initialViewState: { latitude: 0, longitude: 0, position: [0, 0, 0] },
    onError: (error) => {
      failure = error;
    },
    layers: [
      new Tile3DLayer({
        id: "tiles",
        data: config.url,
        pointSize: (config.pointSize ?? 2) / 2,
        loadOptions: {
          tileset: {
            modelMatrix,
            maximumScreenSpaceError: config.sse ?? 8,
            ...(config.memoryMiB === undefined
              ? {}
              : { maximumMemoryUsage: config.memoryMiB }),
          },
        },
        onTilesetLoad: (loaded) => {
          tileset = loaded;
        },
        onTileError: (_tile, url, message) => {
          failure = new Error(url + ": " + message);
        },
      }),
    ],
  });
  return {
    canvas: () => deck.getCanvas(),
    pose({ position, target, up }) {
      const [east, north, vertical] = target.map((v, i) => v - position[i]);
      deck.setProps({
        viewState: {
          longitude: 0,
          latitude: 0,
          position,
          up,
          focalDistance: Math.hypot(east, north, vertical),
          bearing: (Math.atan2(east, north) * 180) / Math.PI,
          pitch:
            (-Math.atan2(vertical, Math.hypot(east, north)) * 180) / Math.PI,
        },
      });
    },
    stats() {
      if (failure) throw failure;
      const selected =
        tileset?.selectedTiles.map((tile) => tile.contentUrl).sort() ?? [];
      return {
        ready: selected.length > 0,
        selected,
        settled: tileset?.isLoaded() ?? false,
      };
    },
  };
}

export async function cesiumViewer(config) {
  globalThis.CESIUM_BASE_URL = "/cesium/";
  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/cesium/Cesium.js";
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
  const C = globalThis.Cesium;
  const widget = new C.CesiumWidget(container(), {
    baseLayer: false,
    globe: false,
    skyBox: false,
    skyAtmosphere: false,
    useDefaultRenderLoop: false,
    creditContainer: document.createElement("div"),
  });
  widget.resize();
  const { camera, scene } = widget;
  scene.backgroundColor = C.Color.BLACK;
  scene.sun = undefined;
  scene.moon = undefined;
  scene.fog.enabled = false;
  camera.frustum.fov =
    2 *
    Math.atan(
      Math.tan(C.Math.toRadians(verticalFov) / 2) *
        Math.max(1, innerWidth / innerHeight),
    );
  const tiles = scene.primitives.add(
    await C.Cesium3DTileset.fromUrl(config.url, {
      maximumScreenSpaceError: config.sse ?? 8,
      dynamicScreenSpaceError: false,
      ...(config.memoryMiB === undefined
        ? {}
        : {
            cacheBytes: (config.memoryMiB - (config.overflowMiB ?? 0)) * mib,
            maximumCacheOverflowBytes: (config.overflowMiB ?? 0) * mib,
          }),
    }),
  );
  // Cesium's Gaussian renderer builds its snapshot in a local ENU frame.
  // Place local captures on the ellipsoid and transform the camera with them.
  const placement =
    C.Cartesian3.magnitude(tiles.boundingSphere.center) < 1_000_000
      ? C.Transforms.eastNorthUpToFixedFrame(C.Cartesian3.fromDegrees(-77, 39))
      : C.Matrix4.IDENTITY;
  tiles.modelMatrix = placement;
  if (config.format === "pnts" && config.pointSize !== undefined)
    tiles.style = new C.Cesium3DTileStyle({
      pointSize: String(config.pointSize),
    });
  const point = (xyz) =>
    C.Matrix4.multiplyByPoint(
      placement,
      new C.Cartesian3(...xyz),
      new C.Cartesian3(),
    );
  const directionAt = (xyz) =>
    C.Matrix4.multiplyByPointAsVector(
      placement,
      new C.Cartesian3(...xyz),
      new C.Cartesian3(),
    );
  let visible = new Set();
  let tileError;
  scene.renderError.addEventListener((_scene, error) => {
    tileError = error;
  });
  tiles.tileFailed.addEventListener((event) => {
    tileError = new Error(event.message + " " + event.url);
  });
  tiles.tileVisible.addEventListener((tile) => visible.add(tile.content.url));
  return {
    canvas: scene.canvas,
    pose({ position, target, up }) {
      const destination = point(position);
      const direction = C.Cartesian3.normalize(
        C.Cartesian3.subtract(point(target), destination, new C.Cartesian3()),
        new C.Cartesian3(),
      );
      camera.setView({
        destination,
        orientation: { direction, up: directionAt(up) },
      });
    },
    draw() {
      if (tileError) throw tileError;
      visible = new Set();
      widget.render();
    },
    stats: () => ({
      ready: visible.size > 0,
      selected: [...visible].sort(),
      settled: tiles.tilesLoaded,
      residentBytes: tiles.totalMemoryUsageInBytes,
      residentLimitBytes: tiles.cacheBytes + tiles.maximumCacheOverflowBytes,
      pressure: tiles.memoryAdjustedScreenSpaceError > (config.sse ?? 8),
      effectiveSse: tiles.memoryAdjustedScreenSpaceError,
    }),
  };
}

export async function playcanvasViewer(config) {
  const pc = await import("playcanvas");
  const canvas = document.createElement("canvas");
  container().appendChild(canvas);
  const app = new pc.Application(canvas, {
    graphicsDeviceOptions: { antialias: false },
  });
  app.setCanvasResolution(pc.RESOLUTION_FIXED, innerWidth, innerHeight);
  app.scene.gsplat.splatBudget = config.splatBudget ?? 1_000_000;
  const camera = new pc.Entity("camera");
  camera.addComponent("camera", {
    clearColor: new pc.Color(0, 0, 0),
    fov: verticalFov,
    nearClip: 0.01,
    farClip: 1e8,
  });
  app.root.addChild(camera);
  const asset = new pc.Asset("benchmark", "gsplat", { url: config.url });
  const resource = new Promise((resolve, reject) => {
    asset.once("load", resolve);
    asset.once("error", reject);
  });
  app.assets.add(asset);
  app.assets.load(asset);
  await resource;
  const splats = new pc.Entity("splats");
  splats.addComponent("gsplat", { asset });
  app.root.addChild(splats);
  app.start();
  return {
    canvas,
    pose({ position, target, up }) {
      camera.setPosition(...position);
      camera.lookAt(new pc.Vec3(...target), new pc.Vec3(...up));
    },
    stats: () => ({ ready: true }),
  };
}

export async function mapLibreViewer(config) {
  const { Map, setWorkerUrl } = await import("maplibre-gl");
  setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");
  const map = new Map({
    container: container(),
    interactive: false,
    attributionControl: false,
    style: {
      version: 8,
      sources: {
        data: {
          type: "raster",
          tiles: [
            new URL(config.url, location.href).href
              .replaceAll("%7B", "{")
              .replaceAll("%7D", "}")
              .replaceAll("%20", " "),
          ],
          tileSize: 256,
          maxzoom: config.maxZoom ?? 4,
        },
      },
      layers: [{ id: "data", type: "raster", source: "data" }],
    },
  });
  let error;
  let ready = false;
  map.once("load", () => {
    ready = true;
  });
  map.on("error", (event) => {
    error = event.error;
  });
  return {
    canvas: map.getCanvas(),
    pose({ center, zoom }) {
      if (error) throw error;
      map.jumpTo({ center, zoom });
    },
    stats: () => ({ ready, settled: map.areTilesLoaded() }),
  };
}

export async function cogViewer(config) {
  const [
    { default: Map },
    { default: GeoTIFF },
    { default: WebGLTile },
    { fromLonLat },
    { default: View },
  ] = await Promise.all([
    import("ol/Map.js"),
    import("ol/source/GeoTIFF.js"),
    import("ol/layer/WebGLTile.js"),
    import("ol/proj.js"),
    import("ol/View.js"),
  ]);
  const source = new GeoTIFF({
    sources: [{ url: config.url }],
    convertToRGB: true,
  });
  const view = await source.getView();
  const map = new Map({
    target: container(),
    controls: [],
    interactions: [],
    layers: [new WebGLTile({ source })],
    view: new View({
      ...view,
      constrainResolution: false,
      center: fromLonLat(config.camera.center, view.projection),
      resolution: config.camera.farResolution,
    }),
  });
  let ready = false;
  let tileError = false;
  map.on("rendercomplete", () => {
    ready = true;
  });
  source.on("tileloaderror", () => {
    tileError = true;
  });
  return {
    canvas: () => container().querySelector("canvas"),
    pose({ center, resolution }) {
      if (tileError) throw new Error("COG tile load failed");
      map
        .getView()
        .setCenter(fromLonLat(center, map.getView().getProjection()));
      map.getView().setResolution(resolution);
    },
    stats: () => ({ ready }),
  };
}

export function tiledVtkViewer(config) {
  const scene = globalThis.pointCloudScene;
  if (!scene.benchmarkCamera)
    throw new Error(
      "The vtk.js build must support the benchmark camera and SSE options",
    );
  if (config.format === "copc") {
    const [{ id }] = scene.datasets();
    if (config.pointSize !== undefined)
      scene.settings.setPointSize(id, "fixed", config.pointSize);
    if (config.pointBudget !== undefined) {
      scene.settings.setFixedPointBudget(id, config.pointBudget);
      scene.settings.setBudgetMode(id, "fixed");
    }
  }
  return {
    canvas: document.querySelector("canvas"),
    pose({ position, target, up }) {
      scene.camera.place({ position, focalPoint: target, viewUp: up });
    },
    stats() {
      const snapshot = scene.stats();
      if (snapshot.error) throw new Error(snapshot.error);
      const member = snapshot.members[0]?.stats;
      if (member?.sourceState === "failed")
        throw new Error(member.lastError ?? "tiled-vtkjs source failed");
      const configuredSse =
        config.format === "copc"
          ? member?.controller?.refinementCutoffPx
          : member?.maximumScreenSpaceErrorPx;
      if (config.sse !== undefined && configuredSse !== config.sse)
        throw new Error(
          "Requested SSE " +
            config.sse +
            " but vtk.js configured " +
            configuredSse,
        );
      const stats = member?.renderer ?? member;
      const activity =
        config.settledReference || config.settleInitial
          ? scene.activity()
          : null;
      return {
        ready: (stats?.drawnTiles ?? 0) > 0 || (stats?.drawnPoints ?? 0) > 0,
        ...(activity
          ? {
              settled:
                activity.loading === 0 &&
                activity.datasets.every(({ state }) => state === "settled"),
              settleActivity: activity.datasets,
            }
          : {}),
        drawnTiles: stats?.drawnTiles,
        drawnPoints: stats?.drawnPoints,
        selected: stats?.drawnTileIds?.toSorted(),
        ...(config.measureDemand
          ? {
              desired: member?.desiredTileIds,
              requested: member?.requestedTileIds,
            }
          : {}),
        selectionPasses: member?.selectionPasses,
        residentBytes: snapshot.memory?.rendererBytes ?? stats?.residentBytes,
        residentLimitBytes:
          snapshot.memory?.totalBytes ?? stats?.resourceCeilingBytes,
        cpuRetainedBytes: snapshot.memory?.cpuRetainedBytes,
        cpuReclaimPasses: snapshot.memory?.cpuReclaimPasses,
        cpuReclaimedBytes: snapshot.memory?.cpuReclaimedBytes,
        workReservedBytes: snapshot.memory?.workReservedBytes,
        pressure:
          member?.memoryConstrained ||
          (member?.blockedGroups ?? 0) > 0 ||
          member?.resourceState?.status === "blocked",
        blockedGroups: member?.blockedGroups,
        configuredSse,
        effectiveSse: member?.effectiveScreenSpaceErrorPx,
        cacheHits: member?.queue?.cacheHits,
        cacheEvictions: member?.queue?.cacheEvictions,
        cumulativeUploadBytes: stats?.logicalUploadBytes,
        rendererRefusalReason: member?.resourceState?.reason,
        resourceRefusalDomain: member?.resourceState?.domain,
        resourceRefusalRequestedBytes: member?.resourceState?.requestedBytes,
        resourceRefusalLimitBytes: member?.resourceState?.limitBytes,
        resourceRefusalWake: member?.resourceState?.wake,
        residentSplats: snapshot.gaussian?.residentSplats,
        drawnSplats: snapshot.gaussian?.displayedSplats,
        preparedDrawSplats: snapshot.gaussian?.visibleSplats,
        pendingGaussianSort: snapshot.gaussian?.pendingSort,
        pendingGaussianPreparation: snapshot.gaussian?.pendingPreparation,
        gaussianSorts: snapshot.gaussian?.sortCount,
        lastGaussianSortMs: snapshot.gaussian?.sortMs,
        activeRequests: member?.queue?.active,
        queuedRequests: member?.queue?.queued,
        readyRequests: member?.queue?.ready,
        atlasBytes: snapshot.gaussian?.gpuAttributeBytes,
        packingUploadBytes: snapshot.gaussian?.attributeUploadBytes,
        frontierRebuilds: snapshot.gaussian?.frontierRebuilds,
        qualityFraction: snapshot.coordinator?.viewQualityFraction,
        paints: snapshot.paints,
      };
    },
  };
}

export const viewers = {
  "tiled-vtkjs": tiledVtkViewer,
  cesium: cesiumViewer,
  "deck-tiles": deckTilesViewer,
  "three-tiles": threeViewer,
  "three-mesh": threeViewer,
  spark: threeViewer,
  potree: threeViewer,
  playcanvas: playcanvasViewer,
  maplibre: mapLibreViewer,
  openlayers: cogViewer,
};
