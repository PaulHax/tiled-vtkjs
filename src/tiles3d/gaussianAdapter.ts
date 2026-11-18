/** Shared, renderer-owned Gaussian realization beneath the tile selector. */

import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkPointGaussianMapper, {
  type PreparedPointTextures,
} from "@kitware/vtk.js/Rendering/Core/PointGaussianMapper";
import vtkRenderPass, {
  type vtkRenderPass as RenderPass,
} from "@kitware/vtk.js/Rendering/SceneGraph/RenderPass";

import { IDENTITY, invert, multiply, type Mat16 } from "../core/mat4";
import { safeCall } from "../core/observers";
import { createMemoryPool } from "../core/memoryPool";
import {
  ResourcePolicyError,
  resourcePolicyError,
  type MemoryLease,
} from "../core/memoryResources";
import {
  enqueueSubmissionSteps,
  type Submission,
} from "../core/submissionScheduler";
import type { CameraView } from "../core/camera";
import type {
  DecodedGaussianPrimitive,
  DecodedMeshTileContent,
  DecodedTileContent,
  GaussianAppearanceProfile,
} from "./decode";
import { TileRenderingProfileError } from "./contentAdapter";
import type {
  TileContentAdapter,
  TileContentAdapterFactory,
  TileContentAdapterOptions,
  TileSubmissionOutcome,
  TileSubmissionRefusal,
} from "./contentAdapter";
import {
  createMeshAdapter,
  type MeshAdapter,
  type MeshAdapterStats,
} from "./meshAdapter";
import { collectContentBuffers } from "./decode/transfer";
import {
  createGaussianSortController,
  createGaussianSortWorkerBackend,
  GaussianSortWorkerError,
  type GaussianSortBackend,
  type GaussianSortPresentation,
  type GaussianSortRange,
} from "./gaussianSort";
import {
  packGaussianRanges,
  type GaussianAttributes,
  type GaussianPackedRange,
} from "./gaussianPacking";
import {
  createGaussianGpuAtlas,
  stageGaussianAttributes,
  stageGaussianOrder,
  type GaussianGpuAtlas,
  type GaussianGpuContext,
} from "./gaussianPresentation";
import {
  createGaussianStorage,
  GAUSSIAN_ATTRIBUTE_BYTES_PER_TUPLE,
} from "./gaussianStorage";

export type GaussianRendererStats = {
  displayedSplats: number;
  visibleSplats: number;
  residentSplats: number;
  residentBytes: number;
  pendingSort: boolean;
  pendingPreparation: boolean;
  attributeUploadBytes: number;
  orderUploadBytes: number;
  sortMs: number;
  sortCount: number;
  frontierRebuilds: number;
  presentationRevision: number;
  cpuAttributeBytes: number;
  gpuAttributeBytes: number;
  workerAttributeBytes: number;
  sortWorkspaceBytes: number;
  replacementBytes: number;
};

export type GaussianContentAdapterFactory = TileContentAdapterFactory & {
  stats(renderer: object): GaussianRendererStats | null;
  whenIdle(renderer: object): Promise<void>;
  noteContext(renderer: object): void;
  dispose(renderer: object): void;
};

export type GaussianContentAdapterFactoryOptions = {
  /** Creates a fresh backend for first use and after worker failure. */
  createSortBackend?: (
    onIdleFailure: (error: GaussianSortWorkerError) => void,
  ) => GaussianSortBackend;
  maximumVisibleSplats?: number;
  resourceBudgetBytes?: number;
  getOpenGLRenderWindow?: TileContentAdapterOptions["getOpenGLRenderWindow"];
};

/** Host render-pass hook without requiring the optional VTK peer declarations. */
export type GaussianAlphaPass = {
  traverse(
    viewNode: unknown,
    rendererNode: {
      apply(pass: object, prepass: boolean): void;
      getChildren(): {
        getRenderable(): object | null;
        traverse(pass: object): void;
      }[];
    },
  ): void;
  getShaderReplacement(): null;
  delete(): void;
};

export class GaussianRenderingProfileError extends TileRenderingProfileError {
  constructor(message: string) {
    super(message);
    this.name = "GaussianRenderingProfileError";
  }
}

type Tile = {
  id: string;
  key: string;
  owner: Owner;
  generation: number;
  gaussian: DecodedGaussianPrimitive[];
  hasTranslucentMesh: boolean;
  count: number;
  bytes: number;
  sourceBytes: number;
  meshBytes: number;
  sourceMemory?: MemoryLease;
  meshReady: boolean;
  submitted: boolean;
  retired: boolean;
  retirementRevision: number;
  group: Tile[];
  replacementTiles: readonly Tile[];
  onSubmitted?: () => void;
};

type Owner = {
  key: number;
  options: TileContentAdapterOptions;
  mesh: MeshAdapter;
  tiles: Map<string, Tile>;
  failed: Set<string>;
  drawn: Set<string>;
  presented: Set<Tile>;
  base: number[];
  presentedBase: number[];
  placementRevision: number;
  visible: boolean;
  visibilityRevision: number;
  ceiling: number;
  requiredBytes: number;
  policyBlocked: boolean;
  workRevision: number;
  submissionRevision: number;
  disposed: boolean;
};

type PackedPrimitive = GaussianPackedRange & {
  id: string;
  tile: Tile;
  allocationKey: string;
};

type Storage = GaussianAttributes & {
  arena: ReturnType<typeof createGaussianStorage>;
  bufferRevision: number;
  signature: string;
  allocationKeys: Set<string>;
  tiles: Tile[];
  primitives: PackedPrimitive[];
  origin: [number, number, number];
  count: number;
  bytes: number;
};

/** Captured slot and buffer identities keep obsolete atlas arrays out of running jobs. */
type SortSnapshot = {
  revision: number;
  atlas: symbol;
  bufferRevision: number;
  primitives: readonly PackedPrimitive[];
  tiles: ReadonlyMap<Tile, number>;
  owners: ReadonlyMap<
    Owner,
    {
      placementRevision: number;
      visibilityRevision: number;
      base: number[];
    }
  >;
  count: number;
};

type Realization = {
  storage: Storage;
  memory: MemoryLease | undefined;
};

type SortTarget = Realization & { snapshot: SortSnapshot };

const tileBuffers = (content: DecodedTileContent) => {
  const buffers = collectContentBuffers(content.primitives);
  return {
    geometry: [...buffers.geometry].reduce(
      (sum, buffer) => sum + buffer.byteLength,
      0,
    ),
    textures: [...buffers.textures].reduce(
      (sum, buffer) => sum + buffer.byteLength,
      0,
    ),
  };
};

const meshContent = (content: DecodedTileContent): DecodedMeshTileContent => {
  const primitives = content.primitives.filter(
    (primitive) => primitive.kind === "mesh",
  );
  return {
    ...content,
    primitives,
    byteEstimate: tileBuffers({ ...content, primitives }),
  };
};

const sameIds = (left: Set<string>, right: Set<string>): boolean =>
  left.size === right.size && [...left].every((id) => right.has(id));

const sameSortRanges = (
  left: readonly GaussianSortRange[] | null,
  right: readonly GaussianSortRange[],
): boolean =>
  left !== null &&
  left.length === right.length &&
  left.every((range, index) => {
    const next = right[index]!;
    return (
      range.id === next.id &&
      range.generation === next.generation &&
      range.positions === next.positions &&
      Array.from(range.modelToScene).every(
        (value, axis) => value === next.modelToScene[axis],
      )
    );
  });

const gaussianColorSpaces: readonly GaussianAppearanceProfile["colorSpace"][] =
  ["srgb_rec709_display"];

const validateProfile = (primitive: DecodedGaussianPrimitive): void => {
  if (!gaussianColorSpaces.includes(primitive.profile.colorSpace))
    throw new GaussianRenderingProfileError(
      `unsupported Gaussian reconstruction colorSpace ${primitive.profile.colorSpace}; shared composition currently requires srgb_rec709_display`,
    );
  if (
    primitive.profile.kernel !== "ellipse" ||
    primitive.profile.sortingMethod !== "cameraDistance" ||
    primitive.profile.projection !== "perspective"
  )
    throw new GaussianRenderingProfileError(
      "unsupported Gaussian rendering profile; expected ellipse/perspective/cameraDistance",
    );
  if (!invert(primitive.modelToScene))
    throw new Error("Gaussian placement must be invertible");
};

const tileAllocationSignature = (tile: Tile): string =>
  `${tile.key}:${tile.generation}:${tile.owner.placementRevision}`;

const allocationKey = (tile: Tile, index: number): string =>
  `${tileAllocationSignature(tile)}/${index}`;

const allocationSignature = (tiles: readonly Tile[]): string =>
  tiles.map(tileAllocationSignature).join("|");

const allocationRequests = (tiles: readonly Tile[]) =>
  tiles.flatMap((tile) =>
    tile.gaussian.map((primitive, index) => ({
      key: allocationKey(tile, index),
      count: primitive.positions.length / 3,
    })),
  );

const matchesStorage = (
  value: Storage | null,
  arena: Storage["arena"],
  signature: string,
): boolean =>
  value?.arena === arena &&
  value.signature === signature &&
  value.bufferRevision === arena.bufferRevision &&
  value.primitives.every(({ allocationKey, allocationRevision }) =>
    arena.owns(allocationKey, allocationRevision),
  );

const makeStorage = (
  tiles: Tile[],
  previous: Storage | null,
  arena: ReturnType<typeof createGaussianStorage>,
): Storage => {
  const signature = allocationSignature(tiles);
  if (previous && matchesStorage(previous, arena, signature)) return previous;
  if (!tiles.length) arena.clear();
  const count = tiles.reduce((sum, tile) => sum + tile.count, 0);
  arena.grow(arena.requiredCapacity(allocationRequests(tiles)));
  const primitives: PackedPrimitive[] = [];
  const allocationKeys = new Set<string>();
  for (const tile of tiles)
    for (const [primitiveIndex, primitive] of tile.gaussian.entries()) {
      const matrix = multiply(tile.owner.base, primitive.modelToScene);
      arena.setOrigin([matrix[12]!, matrix[13]!, matrix[14]!]);
      const primitiveCount = primitive.positions.length / 3;
      const key = allocationKey(tile, primitiveIndex);
      allocationKeys.add(key);
      const allocation = arena.allocate(key, primitiveCount);
      const offset = allocation.offset;
      primitives.push({
        id: `${tile.key}/${primitiveIndex}`,
        tile,
        primitive,
        matrix,
        allocationKey: key,
        allocationRevision: allocation.revision,
        offset,
        count: primitiveCount,
      });
    }
  return {
    arena,
    bufferRevision: arena.bufferRevision,
    signature,
    allocationKeys,
    tiles,
    primitives,
    origin: arena.origin,
    positions: arena.positions,
    scales: arena.scales,
    rotations: arena.rotations,
    colors: arena.colors,
    count,
    bytes: arena.bytes,
  };
};

const premultipliedGaussianActors = new WeakSet<object>();

/** Premultiplied alpha after opaque depth, preserving the host's GL state. */
export const createGaussianAlphaPass = (
  getContext: () => WebGL2RenderingContext,
): GaussianAlphaPass => {
  const pass = {} as RenderPass & { getShaderReplacement(): null };
  vtkRenderPass.extend(pass, {});
  pass.getShaderReplacement = () => null;
  pass.traverse = (_viewNode, rendererNode): void => {
    const gl = getContext();
    const blend = gl.isEnabled(gl.BLEND),
      depth = gl.isEnabled(gl.DEPTH_TEST);
    const writeDepth = gl.getParameter(gl.DEPTH_WRITEMASK) as boolean;
    const factors = [
      gl.BLEND_SRC_RGB,
      gl.BLEND_DST_RGB,
      gl.BLEND_SRC_ALPHA,
      gl.BLEND_DST_ALPHA,
    ].map((name) => gl.getParameter(name) as number);
    const equations = [
      gl.getParameter(gl.BLEND_EQUATION_RGB) as number,
      gl.getParameter(gl.BLEND_EQUATION_ALPHA) as number,
    ];
    gl.enable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
    gl.blendFuncSeparate(
      gl.ONE,
      gl.ONE_MINUS_SRC_ALPHA,
      gl.ONE,
      gl.ONE_MINUS_SRC_ALPHA,
    );
    pass.setCurrentOperation("translucentPass");
    try {
      // Keep ordinary VTK actors on straight-alpha blending. Only adapter-owned
      // Gaussian fields produce premultiplied color; changing the
      // whole renderer to ONE also brightens unrelated translucent actors.
      rendererNode.apply(pass, true);
      try {
        for (const child of rendererNode.getChildren()) {
          const actor = child.getRenderable();
          gl.blendFuncSeparate(
            actor && premultipliedGaussianActors.has(actor)
              ? gl.ONE
              : gl.SRC_ALPHA,
            gl.ONE_MINUS_SRC_ALPHA,
            gl.ONE,
            gl.ONE_MINUS_SRC_ALPHA,
          );
          child.traverse(pass);
        }
      } finally {
        rendererNode.apply(pass, false);
      }
    } finally {
      gl.depthMask(writeDepth);
      gl.blendFuncSeparate(factors[0]!, factors[1]!, factors[2]!, factors[3]!);
      gl.blendEquationSeparate(equations[0]!, equations[1]!);
      if (!blend) gl.disable(gl.BLEND);
      if (!depth) gl.disable(gl.DEPTH_TEST);
    }
  };
  return pass as GaussianAlphaPass;
};

export const createGaussianContentAdapterFactory = (
  configuration: GaussianContentAdapterFactoryOptions = {},
): GaussianContentAdapterFactory => {
  const maximum = configuration.maximumVisibleSplats ?? 2_000_000;
  const budget = configuration.resourceBudgetBytes ?? 512 * 1024 * 1024;
  if (
    !Number.isSafeInteger(maximum) ||
    maximum <= 0 ||
    !Number.isSafeInteger(budget) ||
    budget <= 0
  )
    throw new RangeError(
      "Gaussian splat and memory limits must be positive integers",
    );
  const managers = new WeakMap<object, ReturnType<typeof createManager>>();
  const factory = ((options: TileContentAdapterOptions) => {
    const renderer = options.realizationRenderer ?? options.renderer;
    let manager = managers.get(renderer);
    if (!manager) {
      manager = createManager(
        { ...options, renderer },
        maximum,
        budget,
        configuration.createSortBackend,
        configuration.getOpenGLRenderWindow,
      );
      managers.set(renderer, manager);
    }
    return manager.addOwner(options, () => {
      manager!.dispose();
      managers.delete(renderer);
    });
  }) as GaussianContentAdapterFactory;
  factory.stats = (renderer) => managers.get(renderer)?.stats() ?? null;
  factory.whenIdle = (renderer) =>
    managers.get(renderer)?.whenIdle() ?? Promise.resolve();
  factory.noteContext = (renderer) => managers.get(renderer)?.noteContext();
  factory.dispose = (renderer) => {
    managers.get(renderer)?.dispose();
    managers.delete(renderer);
  };
  return factory;
};

const createManager = (
  firstOptions: TileContentAdapterOptions,
  maximum: number,
  budget: number,
  createBackend?: GaussianContentAdapterFactoryOptions["createSortBackend"],
  getOpenGLRenderWindow?: TileContentAdapterOptions["getOpenGLRenderWindow"],
) => {
  const owners = new Set<Owner>();
  const submissions = firstOptions.submissions;
  const maxBytes = submissions.stats().maxBytesPerFrame;
  // Covariance reconstruction can dominate even for a small byte range.
  const tuplesPerStep = (bytes: number) =>
    Math.max(1, Math.min(1024, Math.floor(maxBytes / bytes)));
  const uploadTuplesPerStep = (bytes: number) =>
    Math.max(1, Math.floor(Math.min(maxBytes, 1024 * 1024) / bytes));
  // 40 bytes of geometry plus two 16-byte color and 4-byte order textures.
  // Allocator and driver overhead are outside these logical byte estimates.
  const gpuBytesPerTuple = getOpenGLRenderWindow ? 80 : 60;
  let arena = createGaussianStorage();
  let attached = true;
  let nextOwner = 0,
    generation = 0,
    workRevision = 0;
  const memoryPool =
    firstOptions.memoryPool ?? createMemoryPool({ totalBytes: budget });
  const memory = memoryPool.createOwner();
  // Reserved capacity, allocated arrays and published storage can differ.
  let arenaMemory: MemoryLease | undefined;
  let realizedMemory: MemoryLease | undefined;
  let displayedMemory: MemoryLease | undefined;
  let workerMemory: MemoryLease | undefined;
  let workerSceneRevision = -1;
  let arenaCapacity = 0;
  const sortCancellation = new AbortController();
  let disposed = false,
    dirty = false,
    pendingSort = false;
  let camera: CameraView | null = null;
  let storage: Storage | null = null,
    candidate: Storage | null = null;
  let sortTarget: SortTarget | null = null;
  let sortRanges: readonly GaussianSortRange[] | null = null;
  let sorting: {
    snapshot: SortSnapshot | undefined;
    ranges: readonly GaussianSortRange[] | null;
    started: number;
  } | null = null;
  let compactGroup: Tile[] | null = null;
  let displayed = new Set<Tile>();
  let cachedOrder: GaussianSortPresentation | null = null;
  let cachedOrderMemory: MemoryLease | undefined;
  const clearCachedOrder = (): void => {
    cachedOrder = null;
    cachedOrderMemory?.release();
    cachedOrderMemory = undefined;
  };
  const forgetOrder = (): void => {
    sortRanges = null;
    clearCachedOrder();
  };
  let attributeUploadBytes = 0,
    orderUploadBytes = 0,
    frontierRebuilds = 0;
  let sortMs = 0,
    sortCount = 0,
    presentationRevision = 0,
    acceptedSortRevision = 0,
    sortTargetRevision = 0;
  let latestColorCamera: readonly number[] | null = null;
  let context: GaussianGpuContext | null = null;
  let watchedCanvas: WebGL2RenderingContext["canvas"] | null = null;
  const scheduleContextPaint = () => firstOptions.scheduleRender();
  const watchCanvas = (next: typeof watchedCanvas): void => {
    if (watchedCanvas === next) return;
    watchedCanvas?.removeEventListener(
      "webglcontextlost",
      scheduleContextPaint,
    );
    watchedCanvas?.removeEventListener(
      "webglcontextrestored",
      scheduleContextPaint,
    );
    watchedCanvas = next;
    watchedCanvas?.addEventListener("webglcontextlost", scheduleContextPaint);
    watchedCanvas?.addEventListener(
      "webglcontextrestored",
      scheduleContextPaint,
    );
  };
  let displayedGpu: GaussianGpuAtlas | null = null;
  let stagingGpu: GaussianGpuAtlas | null = null;
  let packing: { storage: Storage; job: Submission } | null = null;
  let allocationJob: Submission | null = null;
  let publication:
    | (SortTarget & {
        job: Submission;
        scratch?: MemoryLease;
      })
    | null = null;
  let newerPublication = false;
  const idleWaiters = new Set<() => void>();
  let sortAvailability = 0;
  let releaseSortCapacity: (() => void) | undefined;
  const freeSortCapacity = (): void => {
    releaseSortCapacity?.();
    releaseSortCapacity = undefined;
  };
  let idleCheckQueued = false;
  const waitingForPreparation = () => dirty || !!allocationJob || !!packing;
  const hasWork = () => waitingForPreparation() || pendingSort || !!publication;
  const waitingForSortTarget = () =>
    dirty ||
    (waitingForPreparation() &&
      (!sortTarget ||
        [...sortTarget.snapshot.owners].some(([owner, { base }]) =>
          base.some((value, index) => value !== owner.base[index]),
        )));
  const wakeIdle = (): void => {
    if (pendingSort && !waitingForSortTarget()) memoryPool.admission.wake();
    if ((!disposed && hasWork()) || !idleWaiters.size || idleCheckQueued)
      return;
    idleCheckQueued = true;
    queueMicrotask(() => {
      idleCheckQueued = false;
      if (!disposed && hasWork()) return;
      for (const resolve of idleWaiters) resolve();
      idleWaiters.clear();
    });
  };
  const polyData = vtkPolyData.newInstance();
  const scaleArray = vtkDataArray.newInstance({
    name: "GaussianScale",
    numberOfComponents: 3,
    values: new Float32Array(),
  });
  const rotationArray = vtkDataArray.newInstance({
    name: "GaussianRotation",
    numberOfComponents: 4,
    values: new Float32Array(),
  });
  const colorArray = vtkDataArray.newInstance({
    name: "GaussianColor",
    numberOfComponents: 4,
    values: new Float32Array(),
  });
  const orderArray = vtkDataArray.newInstance({
    name: "GaussianOrder",
    numberOfComponents: 1,
    values: new Uint32Array(),
  });
  polyData.getPointData().addArray(scaleArray);
  polyData.getPointData().addArray(rotationArray);
  polyData.getPointData().addArray(colorArray);
  const mapper = vtkPointGaussianMapper.newInstance({
    anisotropic: true,
    emissive: false,
    scaleFactor: 1,
    scaleArray: "GaussianScale",
    rotationArray: "GaussianRotation",
    directColorArray: "GaussianColor",
    coordinateShiftAndScaleEnabled: false,
    boundScale: 6,
  });
  mapper.setSplatShaderCode(
    [
      "//VTK::Color::Impl",
      "float gaussianRadius2 = dot(offsetVCVSOutput, offsetVCVSOutput);",
      "float sourceOpacity = max(opacity, 0.0);",
      "float shape = sourceOpacity > 1.0 ? sqrt(1.0 + 2.718281828459045 * log(sourceOpacity)) : 1.0;",
      "float supportRadius = 3.0 + 0.7 * (shape - 1.0);",
      "if (gaussianRadius2 > supportRadius * supportRadius) { discard; }",
      "float gaussian = exp(-0.5 * gaussianRadius2);",
      "opacity = sourceOpacity > 1.0 ? 1.0 - pow(1.0 - gaussian, sourceOpacity) : sourceOpacity * gaussian;",
      "opacity = clamp(opacity, 0.0, 1.0);",
      "ambientColor *= opacity;",
      "diffuseColor *= opacity;",
    ].join("\n"),
  );
  mapper.setPointOrder(orderArray);
  mapper.setInputData(polyData);
  const actor = vtkActor.newInstance();
  premultipliedGaussianActors.add(actor);
  actor.setMapper(mapper);
  actor.setPickable(false);
  actor.setForceTranslucent(true);
  actor.setVisibility(false);
  actor.getProperty().setLighting(false);
  firstOptions.renderer.addActor(actor);

  const report = (error: unknown): void => {
    const policy = resourcePolicyError(error);
    if (policy) {
      for (const owner of owners) {
        owner.policyBlocked = true;
        safeCall(() => owner.options.onPolicyRefusal?.(policy));
      }
    } else
      for (const owner of owners)
        safeCall(() => owner.options.onError?.(error));
  };
  const groupReady = (group: readonly Tile[]): boolean =>
    group.every((tile) => tile.meshReady && !tile.retired);
  const sourceTiles = (): Tile[] => {
    const selected = new Set(displayed);
    for (const owner of owners)
      for (const tile of owner.tiles.values())
        if (
          !tile.retired &&
          (tile.group !== compactGroup || groupReady(tile.group))
        )
          selected.add(tile);
    if (compactGroup && groupReady(compactGroup))
      for (const tile of compactGroup[0]!.replacementTiles)
        selected.delete(tile);
    for (const owner of owners)
      for (const successor of owner.presented)
        for (const predecessor of successor.replacementTiles)
          if (!owner.drawn.has(predecessor.id)) selected.delete(predecessor);
    return [...selected]
      .filter((tile) => tile.count > 0)
      .sort((left, right) => left.generation - right.generation);
  };
  const chargedTiles = (): Tile[] => {
    const tiles = new Set<Tile>(displayed);
    for (const owner of owners)
      for (const tile of owner.tiles.values())
        if (!tile.retired) tiles.add(tile);
    return [...tiles];
  };
  const sourceCharge = (): number =>
    arena.capacity * gpuBytesPerTuple +
    (storage && storage.positions !== arena.positions
      ? (storage.positions.length / 3) * gpuBytesPerTuple
      : 0);
  const releaseRetiredSources = (): void => {
    for (const tile of retiredSources) {
      const presented = displayed.has(tile) || tile.owner.presented.has(tile);
      if (!presented && !tile.owner.tiles.has(tile.id))
        tile.owner.mesh.retireTile(tile.id);
      if (
        !presented &&
        !storage?.tiles.includes(tile) &&
        !candidate?.tiles.includes(tile) &&
        !packing?.storage.tiles.includes(tile)
      ) {
        tile.sourceMemory?.release();
        tile.sourceMemory = undefined;
        retiredSources.delete(tile);
      }
    }
  };
  const retiredSources = new Set<Tile>();
  const hideTiles = (tiles: ReadonlySet<Tile>): void => {
    if (!storage || !tiles.size) return;
    const removed = storage.primitives.filter((packed) =>
      tiles.has(packed.tile),
    );
    const prepared = mapper.getPreparedPointTextures();
    const previous = orderArray.getData() as Uint32Array;
    const ranges: [number, number][] = [];
    let maximumId = -1;
    let index = 0;
    let rangeIndex = 0;
    let rangeOffset = 0;
    const order = previous.filter((id) => {
      const range = prepared?.ranges?.[rangeIndex];
      const offset = range ? range[0] + rangeOffset : index;
      index++;
      if (range && ++rangeOffset === range[1]) {
        rangeIndex++;
        rangeOffset = 0;
      }
      if (
        removed.some(
          (packed) => id >= packed.offset && id < packed.offset + packed.count,
        )
      )
        return false;
      maximumId = Math.max(maximumId, id);
      if (prepared) {
        const last = ranges.at(-1);
        if (last && last[0] + last[1] === offset) last[1]++;
        else ranges.push([offset, 1]);
      }
      return true;
    });
    if (order.length === previous.length) return;
    orderArray.setData(order, 1);
    if (prepared)
      mapper.setPreparedPointTextures({ ...prepared, ranges, maximumId });
    displayedMemory?.setCpu(
      [
        storage.positions.buffer,
        storage.scales.buffer,
        storage.rotations.buffer,
        storage.colors.buffer,
        order.buffer,
      ] as ArrayBuffer[],
      Math.max(0, storage.arena.capacity * 4 - order.byteLength),
    );
    if (!getOpenGLRenderWindow) orderUploadBytes += order.byteLength;
    for (const tile of tiles) displayed.delete(tile);
    actor.setVisibility(order.length > 0);
    presentationRevision++;
    for (const owner of owners) safeCall(owner.options.onPresentationChange);
    notify(true);
  };
  const hideOwner = (owner: Owner): void => {
    const presented = owner.presented;
    owner.presented = new Set();
    owner.mesh.setDrawnTiles([]);
    hideTiles(presented);
  };
  const refuse = (
    error: TileSubmissionRefusal,
    tiles: readonly Tile[],
  ): void => {
    const groups = new Map<Owner, string[]>();
    for (const tile of tiles) {
      const ids = groups.get(tile.owner) ?? [];
      ids.push(tile.id);
      groups.set(tile.owner, ids);
    }
    for (const [owner, ids] of groups) {
      owner.requiredBytes = Math.max(
        owner.requiredBytes,
        ownerChargeFor(owner),
      );
      safeCall(() =>
        owner.options.onAdmissionRefusal
          ? owner.options.onAdmissionRefusal(ids, error)
          : owner.options.onPolicyRefusal?.(error),
      );
    }
  };
  const gaussianShare = (
    owner: Owner,
    tiles: readonly Tile[],
    bytes: number,
  ): number => {
    const total = tiles.reduce((sum, tile) => sum + tile.count, 0);
    const own = tiles
      .filter((tile) => tile.owner === owner)
      .reduce((sum, tile) => sum + tile.count, 0);
    return total ? (bytes * own) / total : 0;
  };
  const ownerChargeFor = (owner: Owner): number =>
    gaussianShare(owner, chargedTiles(), sourceCharge()) +
    owner.mesh.workState().residentBytes;
  const noteProgress = () => {
    workRevision++;
  };
  const notify = (paint = false): void => {
    workRevision++;
    for (const { options } of owners)
      if (paint || !options.onWorkChange) options.scheduleRender();
      else options.onWorkChange();
    wakeIdle();
  };
  const completeAdmissions = (): void => {
    const groups = new Set<Tile[]>();
    for (const owner of owners)
      for (const tile of owner.tiles.values())
        if (!tile.submitted) groups.add(tile.group);
    for (const group of groups) {
      if (
        group.some(
          (tile) =>
            tile.retired ||
            (tile.count > 0 &&
              !(
                candidate?.tiles.includes(tile) &&
                candidate.primitives.every(
                  (packed) =>
                    packed.tile !== tile ||
                    candidate?.arena.isPacked(packed.allocationRevision),
                )
              )) ||
            !tile.meshReady ||
            !tile.owner.tiles.has(tile.id),
        )
      )
        continue;
      for (const tile of group) {
        tile.submitted = true;
        tile.owner.submissionRevision++;
        tile.owner.workRevision++;
      }
      for (const tile of group) safeCall(tile.onSubmitted);
    }
  };
  const makeBackend =
    createBackend ??
    ((onIdleFailure: (error: GaussianSortWorkerError) => void) =>
      createGaussianSortWorkerBackend(
        new Worker(
          new URL(
            /* @vite-ignore */ "./gaussianSortWorker.js",
            import.meta.url,
          ),
          { type: "module" },
        ),
        onIdleFailure,
      ));
  let backend: GaussianSortBackend | undefined;
  let releaseWorker: (() => void) | undefined;
  let workerRestarted = false;
  let sortUnavailable = false;
  let sortAdmissionBlocked = false;
  const retireBackend = (): void => {
    backend?.dispose();
    backend = undefined;
    releaseWorker?.();
    releaseWorker = undefined;
    workerMemory?.release();
    workerMemory = undefined;
    workerSceneRevision = -1;
  };
  const sortWorkBytes = (): number => {
    const count = sortRanges?.reduce(
      (sum, range) => sum + range.positions.length / 3,
      0,
    );
    return count ? count * 32 + 1024 : 0;
  };
  const compatible = (snapshot: SortSnapshot, target: Realization): boolean => {
    const next = target.storage;
    return (
      snapshot.revision >= acceptedSortRevision &&
      snapshot.atlas === next.arena.identity &&
      snapshot.bufferRevision === next.bufferRevision &&
      snapshot.bufferRevision === next.arena.bufferRevision &&
      [...snapshot.owners].every(
        ([owner, placement]) =>
          !owner.disposed &&
          owner.placementRevision === placement.placementRevision &&
          owner.visibilityRevision === placement.visibilityRevision,
      ) &&
      [...snapshot.tiles].every(
        ([tile, revision]) =>
          tile.retirementRevision === revision &&
          (!tile.retired
            ? tile.owner.tiles.get(tile.id) === tile
            : tile.owner.presented.has(tile)),
      ) &&
      snapshot.primitives.every(
        (packed) =>
          next.allocationKeys.has(packed.allocationKey) &&
          next.arena.owns(packed.allocationKey, packed.allocationRevision),
      )
    );
  };
  const sortFailure = (error: unknown): void => {
    const policy =
      error instanceof GaussianSortWorkerError
        ? null
        : resourcePolicyError(error);
    if (policy) {
      sortAdmissionBlocked = true;
      const unsubmitted = [...owners].flatMap((owner) =>
        [...owner.tiles.values()].filter((tile) => !tile.submitted),
      );
      refuse(policy, unsubmitted);
      for (const tile of unsubmitted) dropTile(tile.owner, tile.id);
    } else {
      // A worker failure leaves every prepared tile eligible for presentation.
      completeAdmissions();
    }
    releaseRetiredSources();
    report(error);
    notify();
  };
  const cancelPublication = (): void => {
    if (!publication) return;
    publication.job.cancel();
    publication.scratch?.release();
    publication = null;
    newerPublication = false;
    if (stagingGpu !== displayedGpu) stagingGpu?.release();
    stagingGpu = null;
  };

  const commitPresentation = (
    presentation: Pick<GaussianSortPresentation, "cameraPosition">,
    snapshot: SortSnapshot,
    target: Realization,
    order: Uint32Array,
    colorRanges: [number, number][],
    sortElapsedMs: number | null,
    textures: PreparedPointTextures | null,
    gpu: GaussianGpuAtlas | null,
  ): void => {
    const next = target.storage;
    const nextMemory = target.memory;
    nextMemory?.setCpu(
      [
        next.positions.buffer,
        next.scales.buffer,
        next.rotations.buffer,
        next.colors.buffer,
        order.buffer,
      ] as ArrayBuffer[],
      Math.max(0, next.arena.capacity * 4 - order.byteLength),
    );
    const changedStorage = storage !== next;
    const updates = next.arena.takePackedUpdates();
    const histories: Parameters<
      typeof mapper.setPointAttributeUpdateRanges
    >[0] = {};
    const sources = [
      {
        name: "positions" as const,
        source: polyData.getPoints(),
        values: next.positions,
        components: 3,
        ranges: updates,
      },
      {
        name: "radius" as const,
        source: scaleArray,
        values: next.scales,
        components: 3,
        ranges: updates,
      },
      {
        name: "rotation" as const,
        source: rotationArray,
        values: next.rotations,
        components: 4,
        ranges: updates,
      },
      {
        name: "colors" as const,
        source: colorArray,
        values: next.colors,
        components: 4,
        ranges: [...updates, ...colorRanges],
      },
    ];
    for (const { name, source, values, components, ranges } of sources) {
      if (source.getData() !== values) {
        source.setData(values, components);
        if (!getOpenGLRenderWindow) attributeUploadBytes += values.byteLength;
      } else if (ranges.length) {
        const fromMTime = source.getMTime();
        source.modified();
        histories[name] = {
          source,
          fromMTime,
          toMTime: source.getMTime(),
          ranges,
        };
        if (!getOpenGLRenderWindow)
          attributeUploadBytes += ranges.reduce(
            (sum, range) => sum + range[1] * components * 4,
            0,
          );
      }
    }
    // Prime the point ranges using the work already done in packing. A
    // bounds request during paint otherwise scans the entire backing atlas.
    const points = polyData.getPoints();
    for (let axis = 0; axis < 3; axis++) {
      let min = 0;
      let max = 0;
      for (const packed of next.primitives) {
        const bounds = next.arena.packedBounds(packed.allocationRevision);
        if (bounds) {
          min = Math.min(min, bounds[axis * 2]!);
          max = Math.max(max, bounds[axis * 2 + 1]!);
        }
      }
      points.setRange({ min, max }, axis);
    }
    if (Object.keys(histories).length)
      mapper.setPointAttributeUpdateRanges(histories);
    if (changedStorage) {
      actor.setUserMatrix([
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        ...next.origin,
        1,
      ]);
      frontierRebuilds++;
    }
    if (next.count > 0 && !attached) {
      firstOptions.renderer.addActor(actor);
      attached = true;
    }
    if (next.count === 0 && attached) {
      firstOptions.renderer.removeActor(actor);
      attached = false;
    }
    orderArray.setData(order, 1);
    if (textures && gpu) {
      gpu.activate();
      mapper.setPreparedPointTextures(textures);
      if (displayedGpu !== gpu) displayedGpu?.release();
      displayedGpu = gpu;
      stagingGpu = null;
    } else if (!order.length && displayedGpu?.positions !== next.positions) {
      displayedGpu?.release();
      displayedGpu = null;
      mapper.setPreparedPointTextures(null);
    }
    if (!getOpenGLRenderWindow) orderUploadBytes += order.byteLength;
    actor.setVisibility(order.length > 0);
    storage = next;
    if (displayedMemory !== nextMemory) {
      if (displayedMemory !== arenaMemory && displayedMemory !== realizedMemory)
        displayedMemory?.release();
    }
    displayedMemory = nextMemory;
    if (next === candidate && nextMemory === arenaMemory) compactGroup = null;
    releaseRetiredSources();
    const presentedTiles = [...snapshot.tiles.keys()];
    for (const [owner, { base }] of snapshot.owners) {
      const tiles = presentedTiles.filter((tile) => tile.owner === owner);
      owner.mesh.setBaseMatrix(base);
      owner.presentedBase = [...base];
      owner.mesh.setVisible(owner.visible);
      owner.mesh.setDrawnTiles(tiles.map((tile) => tile.id));
      owner.presented = new Set(tiles);
    }
    displayed = new Set(snapshot.primitives.map((packed) => packed.tile));
    const resident =
      candidate?.arena === next.arena
        ? candidate.allocationKeys
        : snapshot.primitives.map((packed) => packed.allocationKey);
    const live = retainRanges(next.arena, resident);
    next.arena.retireExcept(live);
    if (gpu) {
      const packed =
        candidate?.arena === next.arena
          ? [...next.primitives, ...candidate.primitives]
          : next.primitives;
      gpu.retireExcept(
        new Set(
          packed.flatMap((primitive) =>
            live.has(primitive.allocationKey)
              ? [primitive.allocationRevision]
              : [],
          ),
        ),
      );
    }
    releaseRetiredSources();
    latestColorCamera = [...presentation.cameraPosition];
    if (sortElapsedMs !== null) {
      sortMs = sortElapsedMs;
      sortCount++;
    }
    presentationRevision++;
    acceptedSortRevision = snapshot.revision;
    sortAdmissionBlocked = false;
    completeAdmissions();
    releaseRetiredSources();
    for (const owner of owners) safeCall(owner.options.onPresentationChange);
    notify(!snapshot.count);
    const retained = sourceTiles();
    if (
      next === candidate &&
      allocationSignature(retained) !== storage.signature
    )
      scheduleRefresh();
  };

  const publish = (
    presentation: Pick<GaussianSortPresentation, "cameraPosition" | "order">,
    snapshot: SortSnapshot,
    target: Realization,
    sortElapsedMs: number | null,
  ): boolean => {
    if (!compatible(snapshot, target)) return false;
    if (
      storage?.positions === target.storage.positions &&
      (latestColorCamera?.every(
        (value, axis) => value === presentation.cameraPosition[axis],
      ) ||
        [...snapshot.owners].some(([owner, { base }]) =>
          base.some((value, axis) => value !== owner.base[axis]),
        )) &&
      (!getOpenGLRenderWindow ||
        !snapshot.count ||
        displayedGpu?.positions === storage.positions) &&
      [...snapshot.owners].every(
        ([owner, { base }]) =>
          base.every((value, index) => value === owner.presentedBase[index]) &&
          [...owner.presented].every((tile) => snapshot.tiles.has(tile)),
      ) &&
      snapshot.tiles.size ===
        [...snapshot.owners.keys()].reduce(
          (size, owner) => size + owner.presented.size,
          0,
        )
    ) {
      if (publication) newerPublication = true;
      return true;
    }
    if (publication) {
      if (compatible(publication.snapshot, publication)) {
        newerPublication = true;
        return true;
      }
      cancelPublication();
    }
    if (presentation.order.length !== snapshot.count) {
      sortFailure(new Error("Gaussian sort returned an invalid order length"));
      return true;
    }
    if (snapshot.count && getOpenGLRenderWindow && !context) return true;
    const next = target.storage;
    let scratch: MemoryLease | undefined;
    let order: Uint32Array;
    let drawToStorage: Uint32Array;
    try {
      if (snapshot.count)
        scratch = memory.retain(
          [presentation.order.buffer as ArrayBuffer],
          snapshot.count * 8,
        );
      order = new Uint32Array(snapshot.count);
      drawToStorage = new Uint32Array(snapshot.count);
      scratch?.setCpu(
        [
          presentation.order.buffer,
          order.buffer,
          drawToStorage.buffer,
        ] as ArrayBuffer[],
        0,
      );
    } catch (error) {
      scratch?.release();
      sortFailure(error);
      return true;
    }
    const failPublication = (error: unknown): void => {
      scratch?.release();
      publication = null;
      newerPublication = false;
      if (stagingGpu !== displayedGpu) stagingGpu?.release();
      stagingGpu = null;
      clearCachedOrder();
      sortFailure(error);
      wakeIdle();
    };
    const steps = function* () {
      const maximumId = yield* stageGaussianOrder(
        snapshot.primitives,
        presentation.order,
        order,
        drawToStorage,
        tuplesPerStep(4),
        tuplesPerStep(8),
      );
      const windowContext = snapshot.count ? context : null;
      if (windowContext) {
        stagingGpu =
          displayedGpu?.positions === next.positions
            ? displayedGpu
            : createGaussianGpuAtlas(windowContext, next.positions);
      }
      const gpu = stagingGpu;
      const colorRanges = yield* stageGaussianAttributes({
        attributes: next,
        ranges: snapshot.primitives,
        gpu,
        camera: presentation.cameraPosition,
        recolor: (packed) =>
          storage !== next ||
          !displayed.has(packed.tile) ||
          !latestColorCamera?.every(
            (value, axis) => value === presentation.cameraPosition[axis],
          ),
        order,
        colorTuplesPerStep: tuplesPerStep(16),
        uploadTuplesPerStep,
        onAttributeUpload: (bytes) => {
          attributeUploadBytes += bytes;
        },
        onOrderUpload: (bytes) => {
          orderUploadBytes += bytes;
        },
      });
      yield {
        bytes: 0,
        requiresPaint: true,
        run() {
          if (
            !compatible(snapshot, target) ||
            windowContext?.context.isContextLost()
          ) {
            cancelPublication();
            scheduleRefresh();
            return;
          }
          publication = null;
          try {
            const textures = gpu?.prepared(maximumId) ?? null;
            commitPresentation(
              presentation,
              snapshot,
              target,
              order,
              colorRanges,
              sortElapsedMs,
              textures,
              gpu,
            );
            if (newerPublication) {
              newerPublication = false;
              scheduleRefresh();
            }
          } finally {
            scratch?.release();
            wakeIdle();
          }
        },
      };
    };
    publication = {
      snapshot,
      storage: target.storage,
      memory: target.memory,
      scratch,
      job: enqueueSubmissionSteps(
        submissions,
        steps(),
        failPublication,
        noteProgress,
      ),
    };
    return true;
  };

  const sorter = createGaussianSortController({
    schedule: (run) =>
      memoryPool.admission.run({
        group: firstOptions.renderer,
        signal: sortCancellation.signal,
        availability: () => memory.availability() + sortAvailability,
        canRetry: (error) =>
          error.domain === "work"
            ? memory.canReserveWork(sortWorkBytes())
            : memory.canRetry(error),
        stage: {
          lane: "sort",
          ready: () =>
            !waitingForSortTarget() &&
            (!!backend || memoryPool.admission.canAcquireWorker("sort")),
          async start(signal) {
            const work = memory.work(signal, sortWorkBytes());
            try {
              await run();
              return { value: undefined };
            } finally {
              sorting = null;
              work.release();
              releaseRetiredSources();
            }
          },
        },
      }),
    backend: {
      async sort(job) {
        const snapshot = sortTarget?.snapshot;
        sorting = { snapshot, ranges: sortRanges, started: performance.now() };
        const count = job.ranges.reduce(
          (sum, range) => sum + range.positions.length / 3,
          0,
        );
        const sourceMemory = memory.retain([
          ...collectContentBuffers(
            snapshot?.primitives.map((packed) => packed.primitive) ?? [],
          ).geometry,
          ...job.ranges.flatMap((range) => [
            range.positions.buffer as ArrayBuffer,
            (range.modelToScene as Float64Array).buffer as ArrayBuffer,
          ]),
        ]);
        let nextWorkerMemory: MemoryLease | undefined;
        try {
          if (count && workerSceneRevision !== job.sceneRevision)
            nextWorkerMemory = memory.retain(
              [],
              count * 24 + job.ranges.length * 128 + 1024,
            );
          try {
            if (!backend) {
              releaseWorker = memoryPool.admission.acquireWorker("sort", {
                idle: () => sorting === null,
                retire: retireBackend,
              });
              if (!releaseWorker)
                throw new Error("Page sort worker capacity is occupied");
              backend = makeBackend((error) => {
                retireBackend();
                if (!disposed) sortFailure(error);
              });
            }
          } catch (cause) {
            retireBackend();
            throw new GaussianSortWorkerError(
              "Could not start the Gaussian sort worker",
              { cause },
            );
          }
          try {
            let pending: Promise<Uint32Array>;
            try {
              pending = backend.sort(job);
            } catch (cause) {
              throw new GaussianSortWorkerError(
                "Could not dispatch Gaussian sort work",
                { cause },
              );
            }
            const order = await pending;
            workerRestarted = false;
            return order;
          } catch (error) {
            if (error instanceof GaussianSortWorkerError) retireBackend();
            throw error;
          } finally {
            if (backend && !disposed) {
              if (nextWorkerMemory) {
                workerMemory?.release();
                workerMemory = nextWorkerMemory;
                nextWorkerMemory = undefined;
              }
              if (!count) workerMemory?.setCpu([], 1024);
              workerSceneRevision = job.sceneRevision;
            }
          }
        } finally {
          nextWorkerMemory?.release();
          sourceMemory.release();
        }
      },
      dispose() {
        sortCancellation.abort();
        freeSortCapacity();
        retireBackend();
      },
    },
    canPresent: () => Boolean(sorting?.snapshot?.count),
    onOrder(presentation) {
      const snapshot = sorting?.snapshot;
      if (!snapshot) return;
      if (sorting?.ranges === sortRanges) {
        try {
          const nextMemory = memory.retain([
            presentation.order.buffer as ArrayBuffer,
          ]);
          clearCachedOrder();
          cachedOrderMemory = nextMemory;
          cachedOrder = presentation;
        } catch (error) {
          sortFailure(error);
          return;
        }
      }
      if (!sortTarget) return;
      if (
        !publish(
          presentation,
          snapshot,
          sortTarget,
          performance.now() - sorting!.started,
        ) &&
        cachedOrder === presentation
      ) {
        // Sorting inputs can survive an atlas change. A refresh can reuse the
        // order with its own current mapping, even if this dispatch cannot.
        scheduleRefresh();
      }
    },
    onWorkerError(error) {
      if (!workerRestarted) {
        workerRestarted = true;
        return "retry";
      }
      sortUnavailable = true;
      freeSortCapacity();
      sortFailure(error);
      return "stop";
    },
    onError: sortFailure,
    onIdle() {
      if (!pendingSort) return;
      pendingSort = false;
      wakeIdle();
      notify();
    },
  });
  const setSortScene = (ranges: readonly GaussianSortRange[]): void => {
    if (sameSortRanges(sortRanges, ranges)) return;
    forgetOrder();
    sortRanges = ranges;
    if (ranges.length > 0 && !sortUnavailable)
      releaseSortCapacity ??= memoryPool.admission.reserveSortCapacity();
    else freeSortCapacity();
    sortAvailability += 1;
    memoryPool.admission.wake();
    sorter.setScene(ranges, { defer: true });
  };

  const requestSort = (position: CameraView["position"]): void => {
    if (sortUnavailable) return;
    pendingSort = true;
    sorter.request(position);
  };

  const ensurePacked = (value: Storage): boolean => {
    if (packing && packing.storage !== value) {
      packing.job.cancel();
      packing = null;
    }
    if (
      value.primitives.every(({ allocationRevision }) =>
        value.arena.isPacked(allocationRevision),
      )
    )
      return true;
    if (packing) return false;
    function* steps() {
      const depth = (packed: PackedPrimitive) =>
        packed.tile.id.split("/").length;
      const ranges = [...value.primitives].sort((a, b) => depth(a) - depth(b));
      yield* packGaussianRanges(
        value,
        ranges,
        tuplesPerStep(GAUSSIAN_ATTRIBUTE_BYTES_PER_TUPLE),
        scheduleRefresh,
      );
      yield {
        bytes: 0,
        run() {
          packing = null;
          scheduleRefresh();
        },
      };
    }
    packing = {
      storage: value,
      job: enqueueSubmissionSteps(
        submissions,
        steps(),
        (error) => {
          packing = null;
          sortFailure(error);
        },
        noteProgress,
      ),
    };
    return false;
  };

  const deferAllocation = (): void => {
    if (allocationJob) return;
    allocationJob = submissions.enqueue({
      bytes: 0,
      run() {
        allocationJob = null;
        refresh(true);
      },
    });
  };

  const regionsOverlap = (left: Tile, right: Tile): boolean =>
    left.id === right.id ||
    left.id.startsWith(right.id + "/") ||
    right.id.startsWith(left.id + "/") ||
    left.replacementTiles.includes(right) ||
    right.replacementTiles.includes(left);

  const planPresentation = (value: Storage) => {
    const tiles = new Set<Tile>();
    const placements = new Map<
      Owner,
      {
        placementRevision: number;
        visibilityRevision: number;
        base: number[];
      }
    >();
    const packedByTile = new Map<Tile, PackedPrimitive[]>();
    for (const packed of value.primitives) {
      const entries = packedByTile.get(packed.tile);
      if (entries) entries.push(packed);
      else packedByTile.set(packed.tile, [packed]);
    }
    const covers = (tile: Tile, base: readonly number[]) =>
      !tile.count ||
      !!packedByTile
        .get(tile)
        ?.every(
          (packed) =>
            value.arena.owns(packed.allocationKey, packed.allocationRevision) &&
            value.arena.isPacked(packed.allocationRevision) &&
            multiply(base, packed.primitive.modelToScene).every(
              (component, index) => component === packed.matrix[index],
            ),
        );
    let covered = 0;
    let samePlacement = 0;
    for (const owner of owners) {
      const desired = owner.visible
        ? [...owner.drawn].flatMap((id) => {
            const tile = owner.tiles.get(id);
            return tile?.submitted ? [tile] : [];
          })
        : [];
      const retained = (missing: readonly Tile[]) =>
        [...owner.presented].filter((predecessor) =>
          missing.some((tile) => regionsOverlap(tile, predecessor)),
        );
      let base = owner.base;
      let missing = desired.filter((tile) => !covers(tile, base));
      let keep = retained(missing);
      const acceptedPlacement =
        keep.some((tile) => !covers(tile, base)) &&
        base.some((value, index) => value !== owner.presentedBase[index]);
      if (acceptedPlacement) {
        // Meshes and their pick transform use one placement per owner.
        base = owner.presentedBase;
        missing = desired.filter((tile) => !covers(tile, base));
        keep = retained(missing);
      } else samePlacement++;
      if (keep.some((tile) => !covers(tile, base))) return null;
      for (const tile of keep) tiles.add(tile);
      for (const tile of desired) {
        if (
          !covers(tile, base) ||
          keep.some((old) => regionsOverlap(tile, old))
        )
          continue;
        tiles.add(tile);
        covered++;
      }
      placements.set(owner, {
        placementRevision: owner.placementRevision,
        visibilityRevision: owner.visibilityRevision,
        base: [...base],
      });
    }
    return { tiles, owners: placements, covered, samePlacement };
  };

  // Keep tuple ranges whose draw or order work can still reference them.
  const retainRanges = (target: Storage["arena"], keys: Iterable<string>) => {
    const live = new Set(keys);
    for (const packed of storage?.arena === target ? storage.primitives : [])
      if (displayed.has(packed.tile)) live.add(packed.allocationKey);
    if (packing?.storage.arena === target)
      for (const packed of packing.storage.primitives)
        live.add(packed.allocationKey);
    for (const snapshot of [sorting?.snapshot, publication?.snapshot])
      if (snapshot?.atlas === target.identity)
        for (const packed of snapshot.primitives)
          live.add(packed.allocationKey);
    return live;
  };

  const updatePresentation = (): void => {
    if (!camera) return;
    const activePlan = storage ? planPresentation(storage) : null;
    const candidatePlan = candidate ? planPresentation(candidate) : null;
    const useCandidate =
      candidatePlan !== null &&
      (!activePlan ||
        candidatePlan.covered > activePlan.covered ||
        (candidatePlan.covered === activePlan.covered &&
          (candidatePlan.samePlacement > activePlan.samePlacement ||
            (candidatePlan.samePlacement === activePlan.samePlacement &&
              (!packing || candidate?.arena === storage?.arena)))));
    const planned = useCandidate ? candidatePlan : activePlan;
    const targetStorage = useCandidate ? candidate : storage;
    const targetMemory = useCandidate ? realizedMemory : displayedMemory;
    if (!planned || !targetStorage) return;
    const presentedTiles = planned.tiles;
    const primitives = targetStorage.primitives.filter((primitive) =>
      presentedTiles.has(primitive.tile),
    );
    const visible = primitives.reduce(
      (sum, primitive) => sum + primitive.count,
      0,
    );
    if (visible > maximum) {
      const refusal = {
        domain: "count",
        finalCount: visible,
        peakCount: visible,
        limitCount: maximum,
        reason: "count-limit",
        wake: "selection",
      } as const;
      for (const owner of owners)
        safeCall(() => owner.options.onPolicyRefusal?.(refusal));
      return;
    }
    const ranges: GaussianSortRange[] = primitives.map((packed) => ({
      id: packed.id,
      generation: packed.tile.generation,
      positions: packed.primitive.positions,
      modelToScene: packed.matrix,
    }));
    const snapshot: SortSnapshot = {
      revision: ++sortTargetRevision,
      atlas: targetStorage.arena.identity,
      bufferRevision: targetStorage.bufferRevision,
      primitives,
      tiles: new Map(
        [...presentedTiles].map((tile) => [tile, tile.retirementRevision]),
      ),
      owners: planned.owners,
      count: visible,
    };
    sortTarget = { storage: targetStorage, memory: targetMemory, snapshot };
    if (publication && !compatible(publication.snapshot, publication))
      cancelPublication();
    setSortScene(ranges);
    const order = cachedOrder;
    if (!visible) {
      publish(
        {
          cameraPosition: camera.position,
          order: new Uint32Array(),
        },
        snapshot,
        sortTarget,
        null,
      );
      if (workerMemory || pendingSort) requestSort(camera.position);
    } else if (
      order &&
      camera.position.every(
        (value, axis) => value === order.cameraPosition[axis],
      )
    ) {
      publish(order, snapshot, sortTarget, null);
    } else if (sortUnavailable) {
      completeAdmissions();
    } else requestSort(camera.position);
    notify();
  };

  const refresh = (allocating = false): void => {
    dirty = false;
    if (disposed || (!camera && chargedTiles().length > 0)) {
      wakeIdle();
      return;
    }
    try {
      // A live atlas migration leaves exactly two physical atlases. Keep its
      // eligible upload running and merge new demand after its presentation.
      if (publication && publication.storage.arena !== storage?.arena) {
        newerPublication = true;
        return;
      }
      if (
        compactGroup &&
        !groupReady(compactGroup) &&
        !compactGroup.some((tile) => tile.retired)
      ) {
        updatePresentation();
        return;
      }
      if (compactGroup?.some((tile) => tile.retired)) compactGroup = null;
      sortTarget = null;
      const nextTiles = sourceTiles();
      if (!nextTiles.length && compactGroup && displayed.size) {
        // A mesh-only successor still replaces an accepted Gaussian region
        // atomically. Its empty atlas has to compete with the published plan.
        if (candidate?.count !== 0) {
          arena = createGaussianStorage();
          candidate = makeStorage([], null, arena);
          realizedMemory = arenaMemory;
        }
        completeAdmissions();
        updatePresentation();
        return;
      }
      if (!nextTiles.length) {
        allocationJob?.cancel();
        allocationJob = null;
        cancelPublication();
        packing?.job.cancel();
        packing = null;
        displayedGpu?.release();
        displayedGpu = null;
        mapper.setPreparedPointTextures(null);
        setSortScene([]);
        const empty = makeStorage([], null, arena);
        polyData.getPoints().setData(empty.positions, 3);
        scaleArray.setData(empty.scales, 3);
        rotationArray.setData(empty.rotations, 4);
        colorArray.setData(empty.colors, 4);
        orderArray.setData(new Uint32Array(), 1);
        actor.setVisibility(false);
        if (attached) {
          firstOptions.renderer.removeActor(actor);
          attached = false;
        }
        for (const lease of new Set([
          arenaMemory,
          realizedMemory,
          displayedMemory,
        ]))
          lease?.release();
        arenaMemory = undefined;
        realizedMemory = undefined;
        displayedMemory = undefined;
        arenaCapacity = 0;
        storage = empty;
        candidate = empty;
        displayed.clear();
        releaseRetiredSources();
        for (const owner of owners) {
          owner.mesh.setBaseMatrix(owner.base);
          owner.presentedBase = [...owner.base];
          owner.mesh.setVisible(owner.visible);
          owner.mesh.setDrawnTiles(owner.visible ? [...owner.drawn] : []);
          owner.presented = new Set(
            owner.visible
              ? [...owner.drawn].flatMap((id) =>
                  owner.tiles.has(id) ? [owner.tiles.get(id)!] : [],
                )
              : [],
          );
        }
        completeAdmissions();
        releaseRetiredSources();
        for (const owner of owners)
          safeCall(owner.options.onPresentationChange);
        presentationRevision++;
        if (camera && (workerMemory || pendingSort))
          requestSort(camera.position);
        notify(true);
        return;
      }
      if (!camera) return;
      const requests = allocationRequests(nextTiles);
      const signature = allocationSignature(nextTiles);
      if (
        storage &&
        candidate !== storage &&
        matchesStorage(storage, storage.arena, signature) &&
        displayedMemory
      ) {
        // After cancellation the published arena can serve these ranges.
        // Release the unused migration before resuming it.
        for (const lease of new Set([arenaMemory, realizedMemory]))
          if (lease !== displayedMemory) lease?.release();
        arena = storage.arena;
        arenaMemory = displayedMemory;
        realizedMemory = displayedMemory;
        arenaCapacity = arena.capacity;
        candidate = storage;
        releaseRetiredSources();
      }
      // Compact only when a replacement was admitted with a compact
      // destination. Unused stable ranges can be reused by future demand.
      const compact =
        compactGroup !== null &&
        groupReady(compactGroup) &&
        !matchesStorage(candidate, arena, signature);
      let nextArena = compact ? createGaussianStorage() : arena;
      if (!compact)
        arena.retireExcept(
          retainRanges(
            arena,
            requests.map((request) => request.key),
          ),
        );
      const projectedCapacity = nextArena.requiredCapacity(requests);
      if (
        !compact &&
        arenaMemory !== realizedMemory &&
        projectedCapacity === arena.capacity &&
        arenaCapacity !== arena.capacity
      ) {
        // Retirement can make speculative growth unnecessary.
        arenaMemory?.release();
        arenaMemory = realizedMemory;
        arenaCapacity = arena.capacity;
      }
      const reservedCompaction =
        compact &&
        arenaMemory !== realizedMemory &&
        arenaMemory !== displayedMemory &&
        arenaCapacity === projectedCapacity;
      if (
        !reservedCompaction &&
        (compact || projectedCapacity !== arenaCapacity)
      ) {
        if (
          arenaMemory &&
          arenaMemory !== displayedMemory &&
          arenaMemory !== realizedMemory
        ) {
          throw new ResourcePolicyError(
            "renderer",
            projectedCapacity * gpuBytesPerTuple,
            memoryPool.stats().totalBytes,
            true,
          );
        }
        arenaMemory = memory.reserveRenderer(
          projectedCapacity * gpuBytesPerTuple,
          {
            cpuBytes: projectedCapacity * 60,
            replacing: arenaMemory ? [arenaMemory] : [],
          },
        );
        arenaCapacity = projectedCapacity;
      }
      if (
        projectedCapacity > nextArena.capacity &&
        storage?.arena === nextArena
      )
        nextArena = createGaussianStorage();
      if (projectedCapacity > nextArena.capacity && !allocating) {
        deferAllocation();
        return;
      }
      if (
        packing &&
        (packing.storage.arena !== nextArena ||
          projectedCapacity !== nextArena.capacity)
      ) {
        packing.job.cancel();
        packing = null;
      }
      if (
        realizedMemory !== displayedMemory &&
        realizedMemory !== arenaMemory
      ) {
        realizedMemory?.release();
        realizedMemory = undefined;
        candidate = null;
      }
      arena = nextArena;
      arena.grow(projectedCapacity);
      candidate = makeStorage(nextTiles, compact ? null : candidate, arena);
      const previousRealizedMemory = realizedMemory;
      realizedMemory = arenaMemory;
      realizedMemory?.setCpu(
        [
          arena.positions.buffer,
          arena.scales.buffer,
          arena.rotations.buffer,
          arena.colors.buffer,
          orderArray.getData().buffer,
        ] as ArrayBuffer[],
        Math.max(0, arena.capacity * 4 - orderArray.getData().byteLength),
      );
      if (
        previousRealizedMemory !== realizedMemory &&
        previousRealizedMemory !== displayedMemory
      )
        previousRealizedMemory?.release();
      ensurePacked(candidate);
      completeAdmissions();
      updatePresentation();
    } catch (error) {
      sortTarget = null;
      setSortScene([]);
      const policy = resourcePolicyError(error);
      const incomplete = [...owners].flatMap((owner) =>
        [...owner.tiles.values()].filter((tile) => !tile.submitted),
      );
      if (policy) refuse(policy, incomplete);
      for (const tile of incomplete)
        if (tile.owner.tiles.get(tile.id) === tile)
          dropTile(tile.owner, tile.id);
      if (!policy)
        for (const tile of incomplete) tile.owner.failed.add(tile.id);
      report(error);
      if (policy && storage) updatePresentation();
    } finally {
      wakeIdle();
    }
  };
  const scheduleRefresh = (): void => {
    if (dirty || disposed) return;
    dirty = true;
    queueMicrotask(() => refresh());
    notify();
  };
  const dropTile = (owner: Owner, id: string, immediate = false): boolean => {
    const tile = owner.tiles.get(id);
    if (!tile) return owner.failed.delete(id);
    if (publication?.snapshot.tiles.has(tile)) cancelPublication();
    if (immediate) {
      owner.presented.delete(tile);
      hideTiles(new Set([tile]));
    }
    retiredSources.add(tile);
    for (const grouped of tile.group)
      if (!grouped.submitted) {
        grouped.retired = true;
        grouped.retirementRevision++;
        retiredSources.add(grouped);
        grouped.owner.tiles.delete(grouped.id);
        grouped.owner.mesh.cancelTile(grouped.id);
        grouped.owner.mesh.retireTile(grouped.id);
      }
    tile.retired = true;
    tile.retirementRevision++;
    owner.tiles.delete(id);
    owner.drawn.delete(id);
    if (immediate || (!displayed.has(tile) && !owner.presented.has(tile)))
      owner.mesh.retireTile(id);
    safeCall(owner.options.onPresentationChange);
    releaseRetiredSources();
    owner.submissionRevision++;
    owner.workRevision++;
    scheduleRefresh();
    return true;
  };
  const dispose = (): void => {
    if (disposed) return;
    disposed = true;
    watchCanvas(null);
    sorter.dispose();
    allocationJob?.cancel();
    allocationJob = null;
    cancelPublication();
    packing?.job.cancel();
    packing = null;
    displayedGpu?.release();
    displayedGpu = null;
    forgetOrder();
    wakeIdle();
    for (const tile of [
      ...retiredSources,
      ...[...owners].flatMap((owner) => [...owner.tiles.values()]),
    ])
      tile.sourceMemory?.release();
    retiredSources.clear();
    for (const lease of new Set([arenaMemory, realizedMemory, displayedMemory]))
      lease?.release();
    memory.close();
    for (const owner of owners) {
      owner.disposed = true;
      owner.mesh.dispose();
      owner.tiles.clear();
    }
    owners.clear();
    displayed.clear();
    if (attached) firstOptions.renderer.removeActor(actor);
    actor.delete();
    mapper.delete();
    polyData.delete();
    scaleArray.delete();
    rotationArray.delete();
    colorArray.delete();
    orderArray.delete();
    storage = null;
    candidate = null;
    sortTarget = null;
    arena.clear();
    firstOptions.scheduleRender();
  };
  return {
    dispose,
    whenIdle(): Promise<void> {
      return disposed || !hasWork()
        ? Promise.resolve()
        : new Promise((resolve) => idleWaiters.add(resolve));
    },
    noteContext() {
      if (!getOpenGLRenderWindow || disposed) return;
      const window = getOpenGLRenderWindow();
      const gl = window?.getContext();
      const generation = window?.getGraphicsResourceGeneration();
      watchCanvas(gl?.canvas ?? null);
      if (window && gl && !gl.isContextLost() && generation !== undefined) {
        if (
          context?.window === window &&
          context.context === gl &&
          context.generation === generation
        )
          return;
        context = {
          window: window as GaussianGpuContext["window"],
          context: gl,
          generation,
        };
      } else {
        if (!context) return;
        context = null;
      }
      cancelPublication();
      displayedGpu?.release();
      displayedGpu = null;
      scheduleRefresh();
    },
    stats(): GaussianRendererStats {
      const current = sourceTiles();
      const count = current.reduce((sum, tile) => sum + tile.count, 0);
      const visible = sortTarget?.snapshot.count ?? 0;
      return {
        displayedSplats: displayed.size
          ? Number(orderArray.getNumberOfTuples())
          : 0,
        visibleSplats: visible,
        residentSplats: count,
        residentBytes:
          sourceCharge() +
          [...owners].reduce(
            (sum, owner) => sum + owner.mesh.workState().residentBytes,
            0,
          ),
        pendingSort: pendingSort || dirty,
        pendingPreparation: !!allocationJob || !!packing || !!publication,
        attributeUploadBytes,
        orderUploadBytes,
        sortMs,
        sortCount,
        frontierRebuilds,
        presentationRevision,
        cpuAttributeBytes:
          arena.bytes +
          (storage && storage.positions !== arena.positions
            ? storage.bytes
            : 0) +
          current.reduce((sum, tile) => sum + tile.sourceBytes, 0),
        gpuAttributeBytes: getOpenGLRenderWindow
          ? ((displayedGpu?.positions.length ?? 0) / 3 +
              (stagingGpu && stagingGpu !== displayedGpu
                ? stagingGpu.positions.length / 3
                : 0)) *
            gpuBytesPerTuple
          : (storage?.bytes ?? 0),
        workerAttributeBytes:
          visible * 12 + (sortTarget?.snapshot.primitives.length ?? 0) * 128,
        sortWorkspaceBytes: visible * 32 + 1024,
        replacementBytes: [...displayed]
          .filter((tile) => tile.retired)
          .reduce((sum, tile) => sum + tile.bytes, 0),
      };
    },
    addOwner(
      options: TileContentAdapterOptions,
      onEmpty: () => void,
    ): TileContentAdapter {
      const owner: Owner = {
        key: ++nextOwner,
        options,
        mesh: createMeshAdapter({
          ...options,
          memory,
          retainReplacements: true,
          onError(error) {
            queueMicrotask(() => {
              if (owner.disposed) return;
              // A failed mesh group also owns Gaussian ranges and source
              // leases in the composite realization.
              const failed = new Set<Tile>();
              for (const tile of owner.tiles.values())
                if (
                  !tile.submitted &&
                  owner.mesh.tileState(tile.id) === "failed"
                )
                  for (const member of tile.group) failed.add(member);
              for (const tile of failed)
                if (owner.tiles.get(tile.id) === tile) dropTile(owner, tile.id);
              for (const tile of failed) owner.failed.add(tile.id);
              safeCall(() => options.onError?.(error));
            });
          },
          ...(getOpenGLRenderWindow ? { getOpenGLRenderWindow } : {}),
        }),
        tiles: new Map(),
        failed: new Set(),
        drawn: new Set(),
        presented: new Set(),
        base: [...IDENTITY],
        presentedBase: [...IDENTITY],
        placementRevision: 0,
        visible: options.visible ?? true,
        visibilityRevision: 0,
        ceiling: budget,
        requiredBytes: 0,
        policyBlocked: false,
        workRevision: 0,
        submissionRevision: 0,
        disposed: false,
      };
      owners.add(owner);
      const ownerCharge = (): number => ownerChargeFor(owner);
      const estimateTileBytes = (content: DecodedTileContent): number =>
        owner.mesh.estimateTileBytes(meshContent(content)) +
        content.primitives.reduce(
          (sum, primitive) =>
            sum +
            (primitive.kind === "gaussian"
              ? (primitive.positions.length / 3) * (2 * gpuBytesPerTuple)
              : 0),
          0,
        );
      const cancelMeshUploads = (bytes: number): readonly string[] => {
        const cancelled = new Set(owner.mesh.setResourceCeilingBytes(bytes));
        for (const id of cancelled) {
          const tile = owner.tiles.get(id);
          if (!tile || tile.submitted) continue;
          for (const grouped of tile.group) cancelled.add(grouped.id);
          dropTile(owner, id);
        }
        return [...cancelled];
      };
      const retainedGeneration = (id: string): Tile | undefined =>
        [...retiredSources].find(
          (entry) =>
            entry.owner === owner &&
            entry.id === id &&
            entry.submitted &&
            (entry.meshBytes === 0 || owner.mesh.tileState(id) === "submitted"),
        );
      const restoreRetired = (id: string): Tile | undefined => {
        const tile = retainedGeneration(id);
        if (!tile) return undefined;
        // IDs name immutable content. Reuse the exact generation that the
        // renderer still owns instead of competing with its internal mesh ID.
        retiredSources.delete(tile);
        tile.retired = false;
        tile.group = [tile];
        tile.replacementTiles = [];
        owner.tiles.set(id, tile);
        owner.failed.delete(id);
        owner.submissionRevision++;
        owner.workRevision++;
        scheduleRefresh();
        return tile;
      };
      const submit = (
        entries: readonly {
          id: string;
          content: DecodedTileContent;
          onSubmitted?: () => void;
        }[],
        replacementIds: readonly string[],
      ): TileSubmissionOutcome => {
        if (disposed || owner.disposed || !entries.length) return "failed";
        const ids = new Set(entries.map((entry) => entry.id));
        if (
          ids.size !== entries.length ||
          entries.some((entry) => !entry.id || owner.failed.has(entry.id))
        )
          return "failed";
        const retained = entries.filter(
          (entry) => !owner.tiles.has(entry.id) && retainedGeneration(entry.id),
        );
        if (retained.length) {
          const fresh = entries.filter((entry) => !retained.includes(entry));
          const outcome = fresh.length
            ? submit(fresh, replacementIds)
            : "queued";
          if (outcome === "queued") {
            for (const entry of retained) restoreRetired(entry.id);
            for (const entry of retained) safeCall(entry.onSubmitted);
          }
          return outcome;
        }
        const group: Tile[] = [];
        const oldArenaMemory = arenaMemory;
        const oldArenaCapacity = arenaCapacity;
        let reserved: readonly MemoryLease[] = [];
        let reservedArena: MemoryLease | undefined;
        const rollbackReservations = (): void => {
          for (const lease of reserved) lease.release();
          if (reservedArena && arenaMemory === reservedArena) {
            arenaMemory = oldArenaMemory;
            arenaCapacity = oldArenaCapacity;
            if (compactGroup === group) compactGroup = null;
          }
        };
        try {
          for (const entry of entries) {
            if (owner.failed.has(entry.id)) return "failed";
            if (owner.tiles.has(entry.id))
              return entries.length === 1 ? "queued" : "failed";
            if (!entry.id || group.some((tile) => tile.id === entry.id))
              throw new Error(
                "Gaussian tile identities must be unique and nonempty",
              );
            const hasTranslucentMesh = entry.content.primitives.some(
              (primitive) =>
                primitive.kind === "mesh" &&
                primitive.material.raw.alphaMode === "BLEND",
            );
            const gaussian = entry.content.primitives.filter(
              (primitive) => primitive.kind === "gaussian",
            );
            gaussian.forEach(validateProfile);
            const count = gaussian.reduce(
              (sum, primitive) => sum + primitive.positions.length / 3,
              0,
            );
            const sourceBytes = tileBuffers({
              ...entry.content,
              primitives: gaussian,
            }).geometry;
            // A tile's demand includes power-of-two attribute capacity and order.
            const bytes = count * (2 * gpuBytesPerTuple);
            group.push({
              id: entry.id,
              key: `${owner.key}:${entry.id}`,
              owner,
              generation: ++generation,
              gaussian,
              hasTranslucentMesh,
              count,
              sourceBytes,
              meshBytes: owner.mesh.estimateTileBytes(
                meshContent(entry.content),
              ),
              bytes,
              meshReady: false,
              submitted: false,
              retired: false,
              retirementRevision: 0,
              group,
              replacementTiles: replacementIds.flatMap((id) =>
                owner.tiles.has(id) ? [owner.tiles.get(id)!] : [],
              ),
              onSubmitted: entry.onSubmitted,
            });
          }
          const additionalCount = group.reduce(
            (sum, tile) => sum + tile.count,
            0,
          );
          const replacements = new Set(replacementIds);
          const replacedCount = [...owner.tiles.values()]
            .filter((tile) => replacements.has(tile.id))
            .reduce((sum, tile) => sum + tile.count, 0);
          const charged = chargedTiles();
          const combined = [...charged, ...group];
          if (
            combined.some((tile) => tile.count > 0) &&
            combined.some((tile) => tile.hasTranslucentMesh)
          )
            throw new GaussianRenderingProfileError(
              "translucent triangle interleaving with Gaussian fields is not supported",
            );
          const residentCount = charged.reduce(
            (sum, tile) => sum + tile.count,
            0,
          );
          const projectedCapacity = arena.requiredCapacity(
            allocationRequests(combined),
          );
          const finalTiles = combined.filter(
            (tile) => tile.owner !== owner || !replacements.has(tile.id),
          );
          const fullGaussianBytes = gaussianShare(
            owner,
            combined,
            projectedCapacity * gpuBytesPerTuple,
          );
          const meshDemand = Math.max(
            owner.mesh.workState().residentBytes,
            owner.mesh.requiredBytes(),
          );
          const fullMeshBytes =
            meshDemand +
            group.reduce((sum, tile) => sum + tile.meshBytes, 0) -
            [...owner.tiles.values()]
              .filter((tile) => replacements.has(tile.id))
              .reduce((sum, tile) => sum + tile.meshBytes, 0);
          const compactReplacement =
            replacementIds.length > 0 &&
            (fullGaussianBytes + fullMeshBytes > owner.ceiling ||
              projectedCapacity * gpuBytesPerTuple + fullMeshBytes >
                memoryPool.stats().totalBytes);
          const finalCapacity = compactReplacement
            ? createGaussianStorage().requiredCapacity(
                allocationRequests(finalTiles),
              )
            : projectedCapacity;
          const gaussianBytes = gaussianShare(
            owner,
            compactReplacement ? finalTiles : combined,
            finalCapacity * gpuBytesPerTuple,
          );
          const finalBytes = gaussianBytes + fullMeshBytes;
          const finalCount = residentCount + additionalCount - replacedCount;
          const peakCount = residentCount + additionalCount;
          if (finalCount > maximum || peakCount > maximum * 2) {
            return {
              status: "blocked",
              refusal: {
                domain: "count",
                finalCount,
                peakCount,
                limitCount: finalCount > maximum ? maximum : maximum * 2,
                reason: "count-limit",
                wake: "selection",
              },
            };
          }
          const peakBytes =
            ownerChargeFor(owner) +
            (compactReplacement || projectedCapacity > arenaCapacity
              ? fullGaussianBytes
              : 0) +
            group.reduce((sum, tile) => sum + tile.meshBytes, 0);
          if (finalBytes > owner.ceiling) {
            owner.requiredBytes = Math.max(owner.requiredBytes, finalBytes);
            return {
              status: "blocked",
              refusal: new ResourcePolicyError(
                "renderer",
                finalBytes,
                owner.ceiling,
                false,
                undefined,
                {
                  reason: "configured-limit",
                  finalBytes,
                  peakBytes,
                  wake: "policy",
                },
              ),
            };
          }
          const grows = compactReplacement || projectedCapacity > arenaCapacity;
          if (
            grows &&
            (compactGroup ||
              (realizedMemory &&
                realizedMemory !== displayedMemory &&
                (packing || publication)))
          ) {
            return {
              status: "blocked",
              refusal: new ResourcePolicyError(
                "renderer",
                peakBytes,
                memoryPool.stats().totalBytes,
                true,
                undefined,
                {
                  reason: "transition-unavailable",
                  finalBytes,
                  peakBytes,
                  wake: "presentation",
                },
              ),
            };
          }
          const cancelled = cancelMeshUploads(
            Math.max(0, owner.ceiling - gaussianBytes),
          );
          if (cancelled.length) {
            const required = Math.max(finalBytes, gaussianBytes + meshDemand);
            owner.requiredBytes = Math.max(owner.requiredBytes, required);
            const error = new ResourcePolicyError(
              "renderer",
              required,
              owner.ceiling,
              true,
              undefined,
              {
                reason: "contention",
                finalBytes,
                peakBytes,
                wake: "availability",
              },
            );
            safeCall(() => options.onAdmissionRefusal?.(cancelled, error));
            return { status: "blocked", refusal: error };
          }
          for (const tile of group) {
            tile.sourceMemory = (options.memory ?? memory).retain([
              ...collectContentBuffers(tile.gaussian).geometry,
            ]);
            owner.tiles.set(tile.id, tile);
          }
          const meshEntries = group.flatMap((tile, index) => {
            const content = meshContent(entries[index]!.content);
            if (!content.primitives.length) {
              tile.meshReady = true;
              return [];
            }
            return [
              {
                id: tile.id,
                content,
                onSubmitted: () => {
                  tile.meshReady = true;
                  scheduleRefresh();
                },
              },
            ];
          });
          // Reserve the final compact atlas atomically with mesh resources;
          // displayed storage remains charged until the new order is accepted.
          const amounts = meshEntries.map((entry) =>
            owner.mesh.resourcePlan(entry.content),
          );
          if (grows)
            amounts.push({
              rendererBytes: finalCapacity * gpuBytesPerTuple,
              cpuBytes: finalCapacity * 60,
            });
          reserved = memory.reserveRendererGroup(amounts, [
            ...owner.mesh.resourceLeases(replacementIds),
            ...(grows && arenaMemory ? [arenaMemory] : []),
          ]);
          if (grows) {
            reservedArena = reserved[meshEntries.length];
            arenaMemory = reservedArena;
            arenaCapacity = finalCapacity;
            if (compactReplacement) compactGroup = group;
          }
          if (meshEntries.length) {
            const outcome = owner.mesh.submitTileGroup(
              meshEntries,
              replacementIds,
              reserved.slice(0, meshEntries.length),
            );
            if (outcome !== "queued") {
              for (const admitted of group) {
                if (owner.tiles.get(admitted.id) === admitted)
                  dropTile(owner, admitted.id);
                if (outcome === "failed") owner.failed.add(admitted.id);
              }
              rollbackReservations();
              return outcome;
            }
          }
          scheduleRefresh();
          if (
            grows &&
            oldArenaMemory &&
            oldArenaMemory !== displayedMemory &&
            oldArenaMemory !== realizedMemory
          )
            oldArenaMemory.release();
          owner.workRevision++;
          notify();
          return "queued";
        } catch (error) {
          rollbackReservations();
          const policy = resourcePolicyError(error);
          if (!policy) for (const entry of entries) owner.failed.add(entry.id);
          for (const tile of group)
            if (owner.tiles.has(tile.id)) dropTile(owner, tile.id);
          if (policy) return { status: "blocked", refusal: policy };
          safeCall(() => options.onError?.(error));
          return "failed";
        }
      };
      return {
        gaussianColorSpaces,
        estimateTileBytes,
        presentationMatrix: () => owner.presentedBase,
        requiredBytes: () =>
          Math.max(
            owner.requiredBytes,
            gaussianShare(
              owner,
              chargedTiles(),
              arenaCapacity * gpuBytesPerTuple,
            ) + owner.mesh.requiredBytes(),
          ),
        clearRequiredBytes() {
          owner.requiredBytes = 0;
          owner.mesh.clearRequiredBytes();
          if (owner.policyBlocked) {
            owner.policyBlocked = false;
            if (sortAdmissionBlocked) {
              sortAdmissionBlocked = false;
              sortTarget = null;
              setSortScene([]);
            }
            scheduleRefresh();
          }
        },
        trimReuse: () => owner.mesh.trimReuse(),
        submitTile(id, content, onSubmitted, replacements = []) {
          return submit([{ id, content, onSubmitted }], replacements);
        },
        submitTileGroup: submit,
        cancelTile(id) {
          const tile = owner.tiles.get(id);
          return tile && !tile.submitted ? dropTile(owner, id) : false;
        },
        retireTile: (id) => dropTile(owner, id),
        setDrawnTiles(ids) {
          const next = new Set(
            ids.filter((id) => owner.tiles.get(id)?.submitted),
          );
          if (sameIds(owner.drawn, next)) return;
          owner.drawn = next;
          scheduleRefresh();
        },
        setBaseMatrix(matrix: Mat16 | null) {
          const next = matrix === null ? [...IDENTITY] : Array.from(matrix);
          if (next.every((value, index) => value === owner.base[index])) return;
          if (!invert(next)) {
            safeCall(() =>
              options.onError?.(
                new GaussianRenderingProfileError(
                  "Gaussian member placement must be a finite invertible affine matrix",
                ),
              ),
            );
            return;
          }
          owner.base = next;
          owner.placementRevision++;
          cancelPublication();
          scheduleRefresh();
        },
        setCamera(view) {
          if (disposed || owner.disposed) return;
          const moved =
            camera === null ||
            camera.position.some(
              (value, index) => value !== view.position[index],
            );
          camera = { ...view, position: [...view.position] };
          if (!sortTarget) scheduleRefresh();
          else if (moved && sortTarget.snapshot.count && !sortUnavailable) {
            requestSort(view.position);
            notify();
          }
        },
        setVisible(visible) {
          if (owner.visible === visible) return;
          owner.visible = visible;
          owner.visibilityRevision++;
          cancelPublication();
          if (!visible) {
            owner.mesh.setVisible(false);
            hideOwner(owner);
          }
          safeCall(owner.options.onPresentationChange);
          scheduleRefresh();
        },
        setResourceCeilingBytes(bytes) {
          if (!Number.isFinite(bytes) || bytes < 0) return [];
          owner.ceiling = Math.floor(bytes);
          const cancelled: string[] = [];
          const finalTiles = chargedTiles().filter(
            (tile) =>
              ![...owners].some((entry) =>
                [...entry.tiles.values()].some((replacement) =>
                  replacement.replacementTiles.includes(tile),
                ),
              ),
          );
          const finalCharge = () =>
            gaussianShare(owner, finalTiles, arenaCapacity * gpuBytesPerTuple) +
            finalTiles
              .filter((tile) => tile.owner === owner)
              .reduce((sum, tile) => sum + tile.meshBytes, 0);
          for (const tile of [...owner.tiles.values()].reverse())
            if (!tile.submitted && finalCharge() > owner.ceiling) {
              cancelled.push(...tile.group.map((entry) => entry.id));
              dropTile(owner, tile.id);
            }
          const meshCancelled = cancelMeshUploads(
            Math.max(
              0,
              owner.ceiling -
                gaussianShare(
                  owner,
                  chargedTiles(),
                  arenaCapacity * gpuBytesPerTuple,
                ),
            ),
          );
          return [...new Set([...cancelled, ...meshCancelled])];
        },
        clearTiles() {
          hideOwner(owner);
          for (const id of owner.tiles.keys()) dropTile(owner, id, true);
          owner.mesh.clearTiles();
          owner.failed.clear();
          owner.drawn.clear();
          safeCall(owner.options.onPresentationChange);
          scheduleRefresh();
        },
        tileState(id) {
          const tile = owner.tiles.get(id);
          return tile
            ? tile.submitted
              ? "submitted"
              : "queued"
            : owner.failed.has(id)
              ? "failed"
              : "absent";
        },
        restoreTile(id) {
          if (owner.failed.has(id)) return "failed";
          if (owner.tiles.has(id)) return "absent";
          return restoreRetired(id) ? "restored" : "absent";
        },
        submittedTileIds: () =>
          [...owner.tiles.values()]
            .filter((tile) => tile.submitted)
            .map((tile) => tile.id),
        submissionRevision: () => owner.submissionRevision,
        presentedTileIds: () => [...owner.presented].map((tile) => tile.id),
        pickableMeshTiles: () => owner.mesh.submittedTiles(),
        workState: () => ({
          workRevision: owner.workRevision + workRevision,
          pendingJobs:
            [...owner.tiles.values()].filter((tile) => !tile.submitted).length +
            (hasWork() ? 1 : 0),
          residentBytes: ownerCharge(),
        }),
        stats(): MeshAdapterStats {
          const mesh = owner.mesh.stats();
          const tiles = [...owner.tiles.values()];
          const submitted = tiles.filter((tile) => tile.submitted);
          const drawn = [...displayed].filter((tile) => tile.owner === owner);
          const drawnIds = [
            ...new Set([...drawn.map((tile) => tile.id), ...mesh.drawnTileIds]),
          ];
          return {
            ...mesh,
            workRevision: owner.workRevision + workRevision,
            pendingTiles: tiles.length - submitted.length,
            pendingJobs: tiles.length - submitted.length + (hasWork() ? 1 : 0),
            submittedTiles: submitted.length,
            submittedTileIds: submitted.map((tile) => tile.id),
            submittedPrimitives:
              mesh.submittedPrimitives +
              submitted.reduce((sum, tile) => sum + tile.gaussian.length, 0),
            submittedActors:
              mesh.submittedActors +
              (submitted.some((tile) => tile.count > 0) ? 1 : 0),
            logicalGeometryUploadBytes:
              mesh.logicalGeometryUploadBytes + attributeUploadBytes,
            logicalUploadBytes:
              mesh.logicalUploadBytes + attributeUploadBytes + orderUploadBytes,
            residentGeometryBytes:
              mesh.residentGeometryBytes +
              tiles.reduce((sum, tile) => sum + tile.bytes, 0),
            residentBytes: ownerCharge(),
            resourceCeilingBytes: owner.ceiling,
            drawnTiles: drawnIds.length,
            drawnTileIds: drawnIds,
            drawnPrimitives:
              mesh.drawnPrimitives +
              drawn.reduce((sum, tile) => sum + tile.gaussian.length, 0),
            drawnActors: mesh.drawnActors + (drawn.length ? 1 : 0),
            visible: owner.visible,
          };
        },
        dispose() {
          if (owner.disposed) return;
          owner.disposed = true;
          hideOwner(owner);
          for (const id of owner.tiles.keys()) dropTile(owner, id, true);
          owner.mesh.dispose();
          owners.delete(owner);
          if (!owners.size) onEmpty();
          else scheduleRefresh();
        },
      };
    },
  };
};
