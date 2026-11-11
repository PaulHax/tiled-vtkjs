/** Shared, renderer-owned Gaussian realization beneath the tile selector. */

import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkPointGaussianMapper from "@kitware/vtk.js/Rendering/Core/PointGaussianMapper";
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
import type { Submission } from "../core/submissionScheduler";
import type { CameraView } from "../core/camera";
import type {
  DecodedGaussianPrimitive,
  DecodedMeshTileContent,
  DecodedTileContent,
} from "./decode";
import type {
  TileContentAdapter,
  TileContentAdapterFactory,
  TileContentAdapterOptions,
} from "./contentAdapter";
import {
  createMeshAdapter,
  type MeshAdapter,
  type MeshAdapterStats,
  type MeshSubmitOutcome,
} from "./meshAdapter";
import { collectContentBuffers } from "./decode/transfer";
import {
  createGaussianSortController,
  createGaussianSortWorkerBackend,
  type GaussianSortBackend,
  type GaussianSortRange,
} from "./gaussianSort";
import {
  gaussianModelCamera,
  createGaussianShapeTransform,
  writeGaussianColor,
} from "./gaussianGeometry";
import { createGaussianStorage } from "./gaussianStorage";

export type GaussianRendererStats = {
  displayedSplats: number;
  visibleSplats: number;
  residentSplats: number;
  residentBytes: number;
  pendingSort: boolean;
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
  dispose(renderer: object): void;
};

export type GaussianContentAdapterFactoryOptions = {
  createSortBackend?: () => GaussianSortBackend;
  maximumVisibleSplats?: number;
  resourceBudgetBytes?: number;
};

/** Host render-pass hook without requiring the optional VTK peer declarations. */
export type GaussianAlphaPass = {
  traverse(
    viewNode: unknown,
    rendererNode: { traverse(pass: object): void },
  ): void;
  getShaderReplacement(): null;
  delete(): void;
};

export class GaussianResourceLimitError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GaussianResourceLimitError";
  }
}

export class GaussianRenderingProfileError extends Error {
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
  prepared: boolean;
  meshReady: boolean;
  submitted: boolean;
  retired: boolean;
  job?: Submission;
  group: Tile[];
  onSubmitted?: () => void;
};

type Owner = {
  key: number;
  options: TileContentAdapterOptions;
  mesh: MeshAdapter;
  tiles: Map<string, Tile>;
  failed: Set<string>;
  drawn: Set<string>;
  presented: Set<string>;
  base: number[];
  presentedBase: number[];
  placementRevision: number;
  visible: boolean;
  ceiling: number;
  requiredBytes: number;
  policyBlocked: boolean;
  workRevision: number;
  submissionRevision: number;
  disposed: boolean;
};

type PackedPrimitive = {
  id: string;
  tile: Tile;
  primitive: DecodedGaussianPrimitive;
  matrix: number[];
  offset: number;
  count: number;
};

type Storage = {
  signature: string;
  allocationKeys: Set<string>;
  tiles: Tile[];
  primitives: PackedPrimitive[];
  origin: [number, number, number];
  positions: Float32Array;
  scales: Float32Array;
  rotations: Float32Array;
  colors: Float32Array;
  count: number;
  bytes: number;
};

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

const validateProfile = (primitive: DecodedGaussianPrimitive): void => {
  if (primitive.profile.colorSpace !== "srgb_rec709_display")
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

const makeStorage = (
  tiles: Tile[],
  previous: Storage | null,
  arena: ReturnType<typeof createGaussianStorage>,
): Storage => {
  const signature = tiles
    .map(
      (tile) =>
        `${tile.key}:${tile.generation}:${tile.owner.placementRevision}`,
    )
    .join("|");
  if (previous?.signature === signature) return previous;
  if (!tiles.length) arena.clear();
  const count = tiles.reduce((sum, tile) => sum + tile.count, 0);
  const primitives: PackedPrimitive[] = [];
  const allocationKeys = new Set<string>();
  for (const tile of tiles)
    for (const [primitiveIndex, primitive] of tile.gaussian.entries()) {
      const matrix = multiply(tile.owner.base, primitive.modelToScene);
      arena.setOrigin([matrix[12]!, matrix[13]!, matrix[14]!]);
      const origin = arena.origin;
      const primitiveCount = primitive.positions.length / 3;
      const key = `${tile.key}:${tile.generation}:${tile.owner.placementRevision}/${primitiveIndex}`;
      allocationKeys.add(key);
      const allocation = arena.allocate(key, primitiveCount);
      const offset = allocation.offset;
      primitives.push({
        id: `${tile.key}/${primitiveIndex}`,
        tile,
        primitive,
        matrix,
        offset,
        count: primitiveCount,
      });
      if (!allocation.fresh) continue;
      const transformShape = createGaussianShapeTransform(matrix);
      for (let index = 0; index < primitiveCount; index++) {
        const p = index * 3,
          q = index * 4,
          destination = offset + index;
        const x = primitive.positions[p]!,
          y = primitive.positions[p + 1]!,
          z = primitive.positions[p + 2]!;
        arena.positions[destination * 3] =
          matrix[0]! * x +
          matrix[4]! * y +
          matrix[8]! * z +
          (matrix[12]! - origin[0]);
        arena.positions[destination * 3 + 1] =
          matrix[1]! * x +
          matrix[5]! * y +
          matrix[9]! * z +
          (matrix[13]! - origin[1]);
        arena.positions[destination * 3 + 2] =
          matrix[2]! * x +
          matrix[6]! * y +
          matrix[10]! * z +
          (matrix[14]! - origin[2]);
        const shape = transformShape(
          primitive.scales.subarray(p, p + 3),
          primitive.rotations.subarray(q, q + 4),
        );
        arena.scales.set(shape.scales, destination * 3);
        arena.rotations.set(shape.rotation, destination * 4);
        if (primitive.harmonicDegree === 0)
          writeGaussianColor(
            primitive,
            index,
            [0, 0, 0],
            arena.colors,
            destination * 4,
          );
      }
    }
  return {
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
      rendererNode.traverse(pass);
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
  createBackend?: () => GaussianSortBackend,
) => {
  const owners = new Set<Owner>();
  const arena = createGaussianStorage();
  let attached = true;
  let nextOwner = 0,
    generation = 0,
    workRevision = 0;
  const memoryPool =
    firstOptions.memoryPool ?? createMemoryPool({ totalBytes: budget });
  const memory = memoryPool.createOwner();
  let arenaMemory: MemoryLease | undefined;
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
  let displayed = new Set<Tile>();
  let desiredPrimitives: PackedPrimitive[] = [];
  let attributeUploadBytes = 0,
    orderUploadBytes = 0,
    frontierRebuilds = 0;
  let sortMs = 0,
    sortCount = 0,
    presentationRevision = 0,
    sortStarted = 0;
  let latestColorCamera: readonly number[] | null = null;
  let presentationAllowed = true;
  let sortRequestSerial = 0;
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
  const sourceTiles = (): Tile[] => {
    const selected = new Set<Tile>();
    for (const owner of owners)
      for (const tile of owner.tiles.values())
        if (tile.prepared && !tile.retired) selected.add(tile);
    for (const tile of displayed) selected.add(tile);
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
    arena.capacity * 60 +
    (storage && candidate && storage.positions !== candidate.positions
      ? (storage.positions.length / 3) * 60
      : 0);
  const releaseRetiredSources = (): void => {
    for (const tile of retiredSources) {
      const presented =
        displayed.has(tile) || tile.owner.presented.has(tile.id);
      if (!presented) tile.owner.mesh.retireTile(tile.id);
      if (
        !presented &&
        !storage?.tiles.includes(tile) &&
        !candidate?.tiles.includes(tile)
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
    const previous = orderArray.getData() as Uint32Array;
    const order = previous.filter(
      (id) =>
        !removed.some(
          (packed) => id >= packed.offset && id < packed.offset + packed.count,
        ),
    );
    if (order.length === previous.length) return;
    orderArray.setData(order, 1);
    arenaMemory?.setCpu(
      [
        arena.positions.buffer,
        arena.scales.buffer,
        arena.rotations.buffer,
        arena.colors.buffer,
        order.buffer,
      ] as ArrayBuffer[],
      Math.max(0, arena.capacity * 4 - order.byteLength),
    );
    orderUploadBytes += order.byteLength;
    for (const tile of tiles) displayed.delete(tile);
    actor.setVisibility(order.length > 0);
    presentationRevision++;
    for (const owner of owners) safeCall(owner.options.onPresentationChange);
    notify();
  };
  const refuse = (error: ResourcePolicyError, tiles: readonly Tile[]): void => {
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
  const desiredTiles = (): Set<Tile> => {
    const result = new Set<Tile>();
    for (const owner of owners)
      if (owner.visible)
        for (const id of owner.drawn) {
          const tile = owner.tiles.get(id);
          if (tile?.submitted && !tile.retired && tile.count) result.add(tile);
        }
    return result;
  };
  const notify = (): void => {
    workRevision++;
    for (const owner of owners) owner.options.scheduleRender();
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
            !tile.prepared ||
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
  const backend =
    createBackend?.() ??
    createGaussianSortWorkerBackend(
      new Worker(
        new URL(/* @vite-ignore */ "./gaussianSortWorker.js", import.meta.url),
        { type: "module" },
      ),
    );
  const sorter = createGaussianSortController({
    backend: {
      async sort(job) {
        sortStarted = performance.now();
        pendingSort = true;
        const count = job.ranges.reduce(
          (sum, range) => sum + range.positions.length / 3,
          0,
        );
        const sourceMemory = memory.retain(
          job.ranges.flatMap((range) => [
            range.positions.buffer as ArrayBuffer,
            (range.modelToScene as Float64Array).buffer as ArrayBuffer,
          ]),
        );
        let work;
        let nextWorkerMemory: MemoryLease | undefined;
        try {
          work = await memory.work(
            sortCancellation.signal,
            count ? count * 32 + 1024 : 0,
          );
          if (count && workerSceneRevision !== job.sceneRevision)
            nextWorkerMemory = memory.retain(
              [],
              count * 24 + job.ranges.length * 128 + 1024,
            );
          const order = await backend.sort(job);
          if (nextWorkerMemory) {
            workerMemory?.release();
            workerMemory = nextWorkerMemory;
            workerSceneRevision = job.sceneRevision;
            nextWorkerMemory = undefined;
          }
          if (!count) {
            workerMemory?.setCpu([], 1024);
            workerSceneRevision = job.sceneRevision;
          }
          return order;
        } finally {
          nextWorkerMemory?.release();
          sourceMemory.release();
          work?.release();
        }
      },
      dispose() {
        sortCancellation.abort();
        backend.dispose();
        workerMemory?.release();
        workerMemory = undefined;
      },
    },
    onOrder(presentation) {
      if (disposed || !candidate || dirty || !presentationAllowed) {
        if (!presentationAllowed) pendingSort = false;
        return;
      }
      try {
        const next = candidate;
        const changedStorage = storage !== next;
        const changedCamera =
          latestColorCamera === null ||
          latestColorCamera.some(
            (value, index) => value !== presentation.cameraPosition[index],
          );
        const colorRanges: [number, number][] = [];
        if (
          changedStorage ||
          changedCamera ||
          desiredPrimitives.some((packed) => !displayed.has(packed.tile))
        )
          for (const packed of desiredPrimitives)
            if (packed.primitive.harmonicDegree > 0) {
              const modelCamera = gaussianModelCamera(
                packed.matrix,
                presentation.cameraPosition,
              );
              for (let index = 0; index < packed.count; index++)
                writeGaussianColor(
                  packed.primitive,
                  index,
                  modelCamera,
                  next.colors,
                  (packed.offset + index) * 4,
                );
              colorRanges.push([packed.offset, packed.count]);
            }
        const drawToStorage = new Uint32Array(
          desiredPrimitives.reduce(
            (sum, primitive) => sum + primitive.count,
            0,
          ),
        );
        let drawnOffset = 0;
        for (const packed of desiredPrimitives) {
          for (let index = 0; index < packed.count; index++)
            drawToStorage[drawnOffset + index] = packed.offset + index;
          drawnOffset += packed.count;
        }
        const order = Uint32Array.from(
          presentation.order,
          (id) => drawToStorage[id]!,
        );
        const updates = arena.updates();
        const histories: Parameters<
          typeof mapper.setPointAttributeUpdateRanges
        >[0] = {};
        const sources = [
          {
            name: "positions" as const,
            source: polyData.getPoints(),
            values: next.positions,
            components: 3,
            ranges: updates.ranges,
          },
          {
            name: "radius" as const,
            source: scaleArray,
            values: next.scales,
            components: 3,
            ranges: updates.ranges,
          },
          {
            name: "rotation" as const,
            source: rotationArray,
            values: next.rotations,
            components: 4,
            ranges: updates.ranges,
          },
          {
            name: "colors" as const,
            source: colorArray,
            values: next.colors,
            components: 4,
            ranges: [...updates.ranges, ...colorRanges],
          },
        ];
        for (const { name, source, values, components, ranges } of sources) {
          if (source.getData() !== values) {
            source.setData(values, components);
            attributeUploadBytes += values.byteLength;
          } else if (ranges.length) {
            const fromMTime = source.getMTime();
            source.modified();
            histories[name] = {
              source,
              fromMTime,
              toMTime: source.getMTime(),
              ranges,
            };
            attributeUploadBytes += ranges.reduce(
              (sum, range) => sum + range[1] * components * 4,
              0,
            );
          }
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
        arenaMemory?.setCpu(
          [
            next.positions.buffer,
            next.scales.buffer,
            next.rotations.buffer,
            next.colors.buffer,
            order.buffer,
          ] as ArrayBuffer[],
          Math.max(0, arena.capacity * 4 - order.byteLength),
        );
        orderUploadBytes += order.byteLength;
        actor.setVisibility(order.length > 0);
        storage = next;
        if (displayedMemory !== arenaMemory) displayedMemory?.release();
        displayedMemory = arenaMemory;
        releaseRetiredSources();
        for (const owner of owners) {
          owner.mesh.setBaseMatrix(owner.base);
          owner.presentedBase = [...owner.base];
          owner.mesh.setVisible(owner.visible);
          owner.mesh.setDrawnTiles(owner.visible ? [...owner.drawn] : []);
          owner.presented = new Set(owner.visible ? owner.drawn : []);
        }
        arena.retireExcept(next.allocationKeys);
        displayed = new Set(
          desiredPrimitives.map((primitive) => primitive.tile),
        );
        releaseRetiredSources();
        latestColorCamera = [...presentation.cameraPosition];
        sortMs = performance.now() - sortStarted;
        sortCount++;
        presentationRevision++;
        pendingSort = false;
        completeAdmissions();
        releaseRetiredSources();
        for (const owner of owners)
          safeCall(owner.options.onPresentationChange);
        notify();
        const retained = sourceTiles();
        if (
          retained
            .map(
              (tile) =>
                `${tile.key}:${tile.generation}:${tile.owner.placementRevision}`,
            )
            .join("|") !== storage.signature
        )
          scheduleRefresh();
      } catch (error) {
        pendingSort = false;
        const policy = resourcePolicyError(error);
        if (policy)
          refuse(
            policy,
            [...owners].flatMap((owner) =>
              [...owner.tiles.values()].filter((tile) => !tile.submitted),
            ),
          );
        for (const owner of owners)
          for (const tile of owner.tiles.values())
            if (!tile.submitted) {
              if (!resourcePolicyError(error)) owner.failed.add(tile.id);
              dropTile(owner, tile.id);
            }
        report(error);
        notify();
      }
    },
    onError(error) {
      pendingSort = false;
      const policy = resourcePolicyError(error);
      if (policy)
        refuse(
          policy,
          [...owners].flatMap((owner) =>
            [...owner.tiles.values()].filter((tile) => !tile.submitted),
          ),
        );
      for (const owner of owners)
        for (const tile of owner.tiles.values())
          if (!tile.submitted) {
            if (!resourcePolicyError(error)) owner.failed.add(tile.id);
            dropTile(owner, tile.id);
          }
      report(error);
      notify();
    },
  });

  const trackSortIdle = (): void => {
    const serial = ++sortRequestSerial;
    void sorter.whenIdle().then(() => {
      if (!disposed && serial === sortRequestSerial) {
        pendingSort = false;
        notify();
      }
    });
  };

  const refresh = (): void => {
    dirty = false;
    if (disposed || (!camera && chargedTiles().length > 0)) return;
    try {
      presentationAllowed = false;
      sorter.setScene([], { defer: true });
      const nextTiles = sourceTiles();
      if (!nextTiles.length) {
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
        const oldArenaMemory = arenaMemory;
        arenaMemory = undefined;
        oldArenaMemory?.release();
        if (displayedMemory !== oldArenaMemory) displayedMemory?.release();
        displayedMemory = undefined;
        arenaCapacity = 0;
        storage = empty;
        candidate = empty;
        desiredPrimitives = [];
        displayed.clear();
        releaseRetiredSources();
        for (const owner of owners) {
          owner.mesh.setBaseMatrix(owner.base);
          owner.presentedBase = [...owner.base];
          owner.mesh.setVisible(owner.visible);
          owner.mesh.setDrawnTiles(owner.visible ? [...owner.drawn] : []);
          owner.presented = new Set(owner.visible ? owner.drawn : []);
        }
        completeAdmissions();
        releaseRetiredSources();
        for (const owner of owners)
          safeCall(owner.options.onPresentationChange);
        presentationRevision++;
        if (camera && (workerMemory || pendingSort)) {
          pendingSort = true;
          sorter.request(camera.position);
          trackSortIdle();
        }
        notify();
        return;
      }
      if (!camera) return;
      const requests = nextTiles.flatMap((tile) =>
        tile.gaussian.map((primitive, index) => ({
          key: `${tile.key}:${tile.generation}:${tile.owner.placementRevision}/${index}`,
          count: primitive.positions.length / 3,
        })),
      );
      const retainedKeys = new Set([
        ...(storage?.allocationKeys ?? []),
        ...requests.map((request) => request.key),
      ]);
      arena.retireExcept(retainedKeys);
      const projectedCapacity = nextTiles.length
        ? arena.requiredCapacity(requests)
        : 0;
      if (
        arenaMemory !== displayedMemory &&
        projectedCapacity === arena.capacity &&
        arenaCapacity !== arena.capacity
      ) {
        arenaMemory?.release();
        arenaMemory = displayedMemory;
        arenaCapacity = arena.capacity;
      }
      if (projectedCapacity !== arenaCapacity) {
        if (arenaMemory && arenaMemory !== displayedMemory) {
          throw new ResourcePolicyError(
            "renderer",
            projectedCapacity * 60,
            memoryPool.stats().totalBytes,
            true,
          );
        }
        const nextMemory = memory.reserveRenderer(projectedCapacity * 60, {
          cpuBytes: projectedCapacity * 60,
          replacing: arenaMemory ? [arenaMemory] : [],
        });
        arenaMemory = nextMemory;
        arenaCapacity = projectedCapacity;
      }
      candidate = makeStorage(nextTiles, candidate, arena);
      arenaMemory?.setCpu(
        [
          arena.positions.buffer,
          arena.scales.buffer,
          arena.rotations.buffer,
          arena.colors.buffer,
          orderArray.getData().buffer,
        ] as ArrayBuffer[],
        Math.max(0, arena.capacity * 4 - orderArray.getData().byteLength),
      );
      const drawn = desiredTiles();
      desiredPrimitives = candidate.primitives.filter((primitive) =>
        drawn.has(primitive.tile),
      );
      const visible = desiredPrimitives.reduce(
        (sum, primitive) => sum + primitive.count,
        0,
      );
      if (visible > maximum)
        throw new GaussianResourceLimitError(
          `visible Gaussian representation contains ${visible} splats, exceeding the ${maximum}-splat limit; choose a coarser tileset representation`,
        );
      const ranges: GaussianSortRange[] = desiredPrimitives.map((packed) => ({
        id: packed.id,
        generation: packed.tile.generation,
        positions: packed.primitive.positions,
        modelToScene: packed.matrix,
      }));
      pendingSort = true;
      presentationAllowed = true;
      sorter.setScene(ranges);
      sorter.request(camera.position);
      trackSortIdle();
      notify();
    } catch (error) {
      pendingSort = false;
      const policy = resourcePolicyError(error);
      if (policy)
        refuse(
          policy,
          [...owners].flatMap((owner) =>
            [...owner.tiles.values()].filter((tile) => !tile.submitted),
          ),
        );
      for (const owner of owners)
        for (const tile of owner.tiles.values())
          if (!tile.submitted) {
            if (!resourcePolicyError(error)) owner.failed.add(tile.id);
            dropTile(owner, tile.id);
          }
      report(error);
    }
  };
  const scheduleRefresh = (): void => {
    if (dirty || disposed) return;
    dirty = true;
    queueMicrotask(refresh);
    notify();
  };
  const dropTile = (owner: Owner, id: string, immediate = false): boolean => {
    const tile = owner.tiles.get(id);
    if (!tile) return owner.failed.delete(id);
    if (immediate) {
      owner.presented.delete(id);
      hideTiles(new Set([tile]));
    }
    retiredSources.add(tile);
    for (const grouped of tile.group)
      if (!grouped.submitted) {
        grouped.job?.cancel();
        grouped.retired = true;
        retiredSources.add(grouped);
        grouped.owner.tiles.delete(grouped.id);
        grouped.owner.mesh.cancelTile(grouped.id);
        grouped.owner.mesh.retireTile(grouped.id);
      }
    tile.job?.cancel();
    tile.retired = true;
    owner.tiles.delete(id);
    owner.drawn.delete(id);
    if (immediate || (!displayed.has(tile) && !owner.presented.has(id)))
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
    sorter.dispose();
    for (const tile of [
      ...retiredSources,
      ...[...owners].flatMap((owner) => [...owner.tiles.values()]),
    ])
      tile.sourceMemory?.release();
    retiredSources.clear();
    arenaMemory?.release();
    if (displayedMemory !== arenaMemory) displayedMemory?.release();
    memory.close();
    for (const owner of owners) {
      owner.disposed = true;
      for (const tile of owner.tiles.values()) tile.job?.cancel();
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
    arena.clear();
    desiredPrimitives = [];
    firstOptions.scheduleRender();
  };
  return {
    dispose,
    async whenIdle(): Promise<void> {
      while (!disposed && (dirty || pendingSort)) {
        await Promise.resolve();
        await sorter.whenIdle();
      }
    },
    stats(): GaussianRendererStats {
      const current = sourceTiles();
      const count = current.reduce((sum, tile) => sum + tile.count, 0);
      const visible = desiredPrimitives.reduce(
        (sum, packed) => sum + packed.count,
        0,
      );
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
        attributeUploadBytes,
        orderUploadBytes,
        sortMs,
        sortCount,
        frontierRebuilds,
        presentationRevision,
        cpuAttributeBytes:
          arena.bytes +
          (candidate && storage && candidate.positions !== storage.positions
            ? storage.bytes
            : 0) +
          current.reduce((sum, tile) => sum + tile.sourceBytes, 0),
        gpuAttributeBytes: storage?.bytes ?? 0,
        workerAttributeBytes: visible * 12 + desiredPrimitives.length * 128,
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
        }),
        tiles: new Map(),
        failed: new Set(),
        drawn: new Set(),
        presented: new Set(),
        base: [...IDENTITY],
        presentedBase: [...IDENTITY],
        placementRevision: 0,
        visible: options.visible ?? true,
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
              ? (primitive.positions.length / 3) * 120
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
      const submit = (
        entries: readonly {
          id: string;
          content: DecodedTileContent;
          onSubmitted?: () => void;
        }[],
        replacementIds: readonly string[],
      ): MeshSubmitOutcome => {
        if (disposed || owner.disposed || !entries.length) return "failed";
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
            const bytes = count * 120;
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
              prepared: false,
              meshReady: false,
              submitted: false,
              retired: false,
              group,
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
          const storageRequests = combined.flatMap((tile) =>
            tile.gaussian.map((primitive, index) => ({
              key: `${tile.key}:${tile.generation}:${tile.owner.placementRevision}/${index}`,
              count: primitive.positions.length / 3,
            })),
          );
          const projectedCapacity = arena.requiredCapacity(storageRequests);
          const gaussianBytes = gaussianShare(
            owner,
            combined,
            projectedCapacity * 60,
          );
          const meshDemand = Math.max(
            owner.mesh.workState().residentBytes,
            owner.mesh.requiredBytes(),
          );
          const finalBytes =
            gaussianBytes +
            meshDemand +
            group.reduce((sum, tile) => sum + tile.meshBytes, 0) -
            [...owner.tiles.values()]
              .filter((tile) => replacements.has(tile.id))
              .reduce((sum, tile) => sum + tile.meshBytes, 0);
          if (
            finalBytes > owner.ceiling ||
            residentCount + additionalCount - replacedCount > maximum ||
            residentCount + additionalCount > maximum * 2
          ) {
            owner.requiredBytes = Math.max(owner.requiredBytes, finalBytes);
            safeCall(() =>
              options.onPolicyRefusal?.(
                new ResourcePolicyError("renderer", finalBytes, owner.ceiling),
              ),
            );
            return "budget-blocked";
          }
          if (arenaMemory && arenaMemory !== displayedMemory && pendingSort) {
            safeCall(() =>
              options.onPolicyRefusal?.(
                new ResourcePolicyError(
                  "renderer",
                  finalBytes,
                  memoryPool.stats().totalBytes,
                  true,
                ),
              ),
            );
            return "budget-blocked";
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
            );
            safeCall(() =>
              options.onAdmissionRefusal
                ? options.onAdmissionRefusal(cancelled, error)
                : options.onPolicyRefusal?.(error),
            );
            return "budget-blocked";
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
          const grows = projectedCapacity > arenaCapacity;
          const amounts = meshEntries.map((entry) =>
            owner.mesh.resourcePlan(entry.content),
          );
          if (grows)
            amounts.push({
              rendererBytes: projectedCapacity * 60,
              cpuBytes: projectedCapacity * 60,
            });
          reserved = memory.reserveRendererGroup(amounts, [
            ...owner.mesh.resourceLeases(replacementIds),
            ...(grows && arenaMemory ? [arenaMemory] : []),
          ]);
          if (grows) {
            reservedArena = reserved[meshEntries.length];
            arenaMemory = reservedArena;
            arenaCapacity = projectedCapacity;
          }
          if (meshEntries.length) {
            const outcome = owner.mesh.submitTileGroup(
              meshEntries,
              replacementIds,
              reserved.slice(0, meshEntries.length),
            );
            if (outcome !== "queued") {
              for (const admitted of group) dropTile(owner, admitted.id);
              rollbackReservations();
              return outcome;
            }
          }
          for (const tile of group) {
            tile.job = options.submissions.enqueue({
              bytes: tile.bytes,
              atomic: true,
              run() {
                if (!tile.retired && !owner.disposed) {
                  tile.prepared = true;
                  scheduleRefresh();
                }
              },
              onError(error) {
                owner.failed.add(tile.id);
                for (const admitted of group) dropTile(owner, admitted.id);
                safeCall(() => options.onError?.(error));
              },
            });
          }
          if (
            grows &&
            oldArenaMemory &&
            oldArenaMemory !== displayedMemory &&
            arena.capacity !== oldArenaCapacity
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
          safeCall(() =>
            policy
              ? options.onPolicyRefusal?.(policy)
              : options.onError?.(error),
          );
          return policy ? "budget-blocked" : "failed";
        }
      };
      return {
        estimateTileBytes,
        presentationMatrix: () => owner.presentedBase,
        requiredBytes: () =>
          Math.max(
            owner.requiredBytes,
            gaussianShare(owner, chargedTiles(), arenaCapacity * 60) +
              owner.mesh.requiredBytes(),
          ),
        clearRequiredBytes() {
          owner.requiredBytes = 0;
          owner.mesh.clearRequiredBytes();
          if (owner.policyBlocked) {
            owner.policyBlocked = false;
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
          if (!candidate) scheduleRefresh();
          else if (!presentationAllowed) scheduleRefresh();
          else if (moved) {
            pendingSort = true;
            sorter.request(view.position);
            trackSortIdle();
            notify();
          }
        },
        setVisible(visible) {
          if (owner.visible === visible) return;
          owner.visible = visible;
          if (!visible)
            hideTiles(
              new Set([...displayed].filter((tile) => tile.owner === owner)),
            );
          if (!visible) {
            owner.presented.clear();
            owner.mesh.setVisible(false);
            owner.mesh.setDrawnTiles([]);
          }
          safeCall(owner.options.onPresentationChange);
          scheduleRefresh();
        },
        setResourceCeilingBytes(bytes) {
          if (!Number.isFinite(bytes) || bytes < 0) return [];
          owner.ceiling = Math.floor(bytes);
          const cancelled: string[] = [];
          for (const tile of [...owner.tiles.values()].reverse())
            if (!tile.submitted && ownerCharge() > owner.ceiling) {
              cancelled.push(...tile.group.map((entry) => entry.id));
              dropTile(owner, tile.id);
            }
          const meshCancelled = cancelMeshUploads(
            Math.max(
              0,
              owner.ceiling -
                gaussianShare(owner, chargedTiles(), arenaCapacity * 60),
            ),
          );
          return [...new Set([...cancelled, ...meshCancelled])];
        },
        clearTiles() {
          for (const id of owner.tiles.keys()) dropTile(owner, id, true);
          owner.mesh.clearTiles();
          owner.failed.clear();
          owner.drawn.clear();
          owner.presented.clear();
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
        restoreTile: (id) => (owner.failed.has(id) ? "failed" : "absent"),
        submittedTileIds: () =>
          [...owner.tiles.values()]
            .filter((tile) => tile.submitted)
            .map((tile) => tile.id),
        submissionRevision: () => owner.submissionRevision,
        pickableMeshTiles: () => owner.mesh.submittedTiles(),
        workState: () => ({
          workRevision: owner.workRevision + workRevision,
          pendingJobs:
            [...owner.tiles.values()].filter((tile) => !tile.submitted).length +
            (pendingSort || dirty ? 1 : 0),
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
            pendingJobs:
              tiles.length - submitted.length + (pendingSort || dirty ? 1 : 0),
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
          for (const id of owner.tiles.keys()) dropTile(owner, id, true);
          owner.mesh.dispose();
          owner.presented.clear();
          owners.delete(owner);
          if (!owners.size) onEmpty();
          else scheduleRefresh();
        },
      };
    },
  };
};
