/**
 * vtk.js renderer adapter: consumes controller tile batches and realizes
 * contiguous point spans as vtkPolyData + vtkPointSpriteMapper + vtkActor.
 *
 * The format adapter owns its vtk.js imports.
 * The mapper renders each point as one gl.POINTS vertex (no cell topology),
 * so a tile's cost is exactly its point payload.
 *
 * Lifecycle contract — the controller owns residency, the adapter owns actors:
 *
 * - `applyBatch` is the only entry point that changes which tiles exist. A
 *   batch is a disjoint delta against what the controller last submitted, so
 *   an addition for a key already on screen means the payload was replaced.
 * - `setVisible` is a draw switch and nothing else. Hiding keeps every actor
 *   and its GPU resources so showing again is a pure state restore, and
 *   batches keep applying while hidden (new actors are created invisible).
 *   Freeing a hidden cloud is the controller's decision — `setActive(false)`
 *   arrives here as removals — and `setResourceCeilingBytes` bounds whatever
 *   the reuse pool still holds.
 *
 * The adapter never calls `renderWindow.render()`; every visual change goes
 * through the injected `scheduleRender`, which must coalesce (the host owns
 * render pacing — e.g. a shared-context integration).
 */

import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkPolyData, {
  type vtkPolyData as VtkPolyData,
} from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkActor, {
  type vtkActor as VtkActor,
} from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkPointSpriteMapper, {
  type vtkPointSpriteMapper as VtkPointSpriteMapper,
} from "@kitware/vtk.js/Rendering/Core/PointSpriteMapper";

import type { PresentedPointTile, TileDrawPlan } from "./controller";
import type { TileBatch } from "./payloadResidency";
import { IDENTITY, sameMatrix, translatedMatrix } from "../core/mat4";
import { finiteAbove, finiteNonNegative } from "../core/numeric";
import { keyToString } from "./octree";
import { setActorPointSize, setMapperPointCount } from "./drawState";
import { tileBytes, type TileData } from "./tileSource";
import { safeCall } from "../core/observers";
import {
  ResourcePolicyError,
  type MemoryOwner,
  type MemoryLease,
} from "../core/memoryResources";
import {
  enqueueSubmissionSteps,
  type Submission,
  type SubmissionScheduler,
} from "../core/submissionScheduler";

type PointActorView = {
  getContext(): WebGL2RenderingContext | null;
  preparePolygonalGeometry(
    renderer: unknown,
    actor: unknown,
  ): { bytes: number; release(): void };
};

export type RendererAdapterOptions = {
  /** vtk.js renderer the tile actors are added to. */
  renderer: {
    addActor(actor: unknown): void;
    removeActor(actor: unknown): void;
  };
  /** Coalescing render request (see module doc). */
  scheduleRender: () => void;
  /** Initial uniform point diameter in CSS pixels. Default 2. */
  diameterCssPx?: number;
  /** Initial CSS-to-framebuffer scale. Default 1. */
  devicePixelRatio?: number;
  /** Initial visibility. Default true. Tiles added while hidden stay hidden. */
  visible?: boolean;
  memory?: MemoryOwner;
  /** When supplied, construction is bounded and new coverage publishes before paint. */
  submissions?: SubmissionScheduler;
  /** Use this live view for preparing new point buffers while actors are hidden. */
  getOpenGLRenderWindow?: () => PointActorView | null | undefined;
  onWorkChange?: () => void;
  onError?: (error: unknown) => void;
  onTileError?: (addition: TileAddition, error: unknown) => void;
};

export type RendererAdapter = {
  reserveTile(key: string, tile: TileData): MemoryLease | undefined;
  trimReuse(): void;
  /**
   * Apply one controller batch (typically wired as `onTiles`). Batches apply
   * whether or not the adapter is visible; an addition for a key already on
   * screen replaces that tile's payload.
   */
  applyBatch(batch: TileBatch): void;
  /** Apply per-tile progressive prefixes without replacing actors or VBOs. */
  applyDrawPlan(plan: TileDrawPlan): void;
  /** Point prefixes in the accepted presentation, for picking and occlusion. */
  presentedTiles(): Iterable<PresentedPointTile>;
  /** Invalidate every actor and outstanding preparation on a source change. */
  clearTiles(): void;
  /**
   * Anchor transform (column-major 16 floats, e.g. the scene actor's
   * UserMatrix); each tile renders with `base · translate(tile.origin)`.
   */
  setBaseMatrix(matrix: ArrayLike<number> | null): void;
  setPointDiameterCssPx(diameterCssPx: number): void;
  setDevicePixelRatio(devicePixelRatio: number): void;
  /**
   * Draw switch only: hiding keeps every actor and its GPU resources, so
   * showing again restores exactly the submitted set — including tiles that
   * arrived while hidden. Releasing a hidden cloud's residency is the
   * controller's call (`setActive(false)`, which arrives as removals).
   */
  setVisible(visible: boolean): void;
  /** Set this adapter's allocation from the shared GPU-memory pool. */
  setResourceCeilingBytes(bytes: number): void;
  /** Renderer-owned resource and draw accounting. */
  stats(): RendererAdapterStats;
  /** Held counters for coordinator refreshes, without scanning the drawn set. */
  workState(): Pick<
    RendererAdapterStats,
    "workRevision" | "gpuResidentBytes" | "submittedBytes" | "pendingJobs"
  >;
  /**
   * Diagnostics: the keys this adapter holds actors for, and the keys whose
   * actors are pooled awaiting reuse or release. A scheduled adapter reports
   * the accepted set, which can lag the latest controller batch.
   */
  activeKeys(): { readonly submitted: string[]; readonly pooled: string[] };
  /**
   * Cancel preparation and release every tile actor. Idempotent.
   *
   * Disposing the controller is NOT enough on its own. Its teardown hands back
   * the submitted set as removals, which this adapter answers by moving those
   * actors into the reuse pool — so a host that disposes only the controller
   * leaves up to `resourceCeilingBytes` of GPU resources alive with nothing
   * left that could ever ask for them again. Both objects must be disposed,
   * and the pair is what releases a cloud.
   */
  dispose(): void;
};

/**
 * Three disjoint questions, three sets of fields: what the controller has
 * submitted, what the reuse pool still holds, and what actually draws.
 * `gpuResident*` is the sum of the first two — the resources this adapter
 * owns, which is what a memory ceiling has to work from.
 */
export type RendererAdapterStats = {
  /**
   * Monotonic revision of submitted resource changes. The first render after
   * this changes may allocate or upload buffers and is not a capacity sample.
   */
  readonly workRevision: number;
  /** Tiles matching the controller's submitted set; drawn while visible. */
  readonly submittedTiles: number;
  readonly submittedPoints: number;
  readonly submittedBytes: number;
  /** Retired tiles retained for same-key reuse; never drawn. */
  readonly pooledTiles: number;
  readonly pooledPoints: number;
  readonly pooledBytes: number;
  /**
   * Whether the pool is earning its residency: session totals for tiles that
   * came back to the same key holding the same payload, against tiles built
   * from nothing. A pool that never hits is memory spent on nothing.
   */
  readonly reusedTiles: number;
  readonly builtTiles: number;
  /** Outstanding tile preparations and a possible prepaint publication. */
  readonly pendingJobs: number;
  /** Vertex and color buffer bytes allocated during owning-view preparation. */
  readonly gpuGeometryUploadBytes: number;
  /**
   * Pool hits bucketed by how many tiles were pooled after the one that hit:
   * `[0], [1], [2,3], [4,7], ...` with the last bucket open.
   *
   * The count is insertions since the entry was pooled, not the entry's rank
   * in the pool at the moment it hit: insertions that were themselves evicted
   * before the hit still count. So it reads as an upper bound on how deep a
   * newest-N policy would have had to hold the entry, and the hits in the
   * buckets up to N are the ones such a policy is guaranteed to have earned —
   * it may earn some of the deeper ones too.
   */
  readonly reuseDepths: readonly number[];
  /** Submitted, pooled, and in preparation: tiles holding actor state. */
  readonly gpuResidentTiles: number;
  readonly gpuResidentPoints: number;
  /**
   * Managed point realization reservations, including a per-actor estimate.
   * Cumulative uploads are measured separately in `gpuGeometryUploadBytes`.
   */
  readonly gpuResidentBytes: number;
  /**
   * The pool ceiling. Only the pool is trimmed to it, so submitted tiles alone
   * may exceed it — but then the pool is empty.
   */
  readonly resourceCeilingBytes: number;
  /** What participates in drawing: the submitted set, or nothing while hidden. */
  readonly drawnTiles: number;
  readonly drawnPoints: number;
  readonly drawnFraction: number;
  readonly visible: boolean;
  readonly diameterCssPx: number;
  readonly devicePixelRatio: number;
};

type PointSpan = {
  readonly actor: VtkActor;
  readonly mapper: VtkPointSpriteMapper;
  readonly polyData: VtkPolyData;
  readonly firstPoint: number;
  readonly count: number;
  attached: boolean;
  releaseViewResources?: () => void;
};

type TileActors = {
  memory?: MemoryLease;
  readonly spans: PointSpan[];
  tile: TileData;
  resourceBytes: number;
  drawnPointCount: number;
  /** Pool insertions before this one, so a hit's depth in the pool is known. */
  pooledSerial: number;
};

type TileAddition = TileBatch["added"][number];

type PreparedAddition = {
  addition: TileAddition;
  /** Undefined when the currently submitted actor already owns this payload. */
  entry?: TileActors;
  reused?: boolean;
};

type PendingAddition = {
  addition: TileAddition;
  job?: Submission;
  building?: TileActors;
  prepared?: PreparedAddition;
};

/** `[0], [1], [2,3], [4,7], ...`; the last bucket takes everything deeper. */
const REUSE_DEPTH_BUCKETS = 12;

const depthBucket = (depth: number): number =>
  depth <= 0
    ? 0
    : Math.min(REUSE_DEPTH_BUCKETS - 1, Math.floor(Math.log2(depth)) + 1);

export const createRendererAdapter = (
  options: RendererAdapterOptions,
): RendererAdapter => {
  const { renderer, scheduleRender, submissions } = options;
  const maxSpanBytes = submissions?.stats().maxBytesPerFrame;
  let diameterCssPx = finiteAbove(
    "diameterCssPx",
    options.diameterCssPx ?? 2,
    0,
  );
  let devicePixelRatio = finiteAbove(
    "devicePixelRatio",
    options.devicePixelRatio ?? 1,
    0,
  );
  let visible = options.visible ?? true;
  /**
   * The prefix each submitted tile draws, as the controller last planned it.
   * Null until the first plan arrives: a tile submitted before any plan draws
   * whole, while a tile absent from a plan that did arrive draws nothing.
   */
  let pointPrefixes: ReadonlyMap<string, number> | null = null;
  let desiredPrefixes: ReadonlyMap<string, number> | null = null;
  let baseMatrix: ArrayLike<number> = IDENTITY;
  // A key lives in at most one of the two: removal moves its entry from
  // `tiles` to `pendingRelease`, and a re-addition either takes that entry
  // back or releases it. Nothing else may hold a reference to an entry, so
  // teardown releases each actor exactly once.
  const tiles = new Map<string, TileActors>();
  const pendingRelease = new Map<string, TileActors>();
  const pending = new Map<string, PendingAddition>();
  const retirements = new Set<string>();
  let publication: Submission | undefined;
  let resourceCeilingBytes = 256 * 1024 * 1024;
  let disposed = false;

  // Running totals rather than a walk per question. Trimming asks how many
  // bytes are held once per evicted entry, so re-summing both maps there would
  // make shedding a large pool quadratic in its size — and the pool is largest
  // exactly when a cloud has just been deactivated and the ceiling dropped to
  // nothing, which is when the walk is longest and the answer needed soonest.
  // Every mutation of either map goes through the four helpers below.
  let submittedPoints = 0;
  let submittedBytes = 0;
  let pooledPoints = 0;
  let pooledBytes = 0;
  let reusedTiles = 0;
  let builtTiles = 0;
  let workRevision = 0;
  let preparedBytes = 0;
  let gpuGeometryUploadBytes = 0;
  let pooledSerial = 0;
  /**
   * Pool hits by how many tiles were pooled after the one that hit, in
   * power-of-two buckets. The pool is retained by bytes, but what it costs to
   * leave attached is counted in props, and those are not the same question.
   */
  const reuseDepths = Array.from({ length: REUSE_DEPTH_BUCKETS }, () => 0);

  const attach = (entry: PointSpan): void => {
    if (entry.attached) return;
    entry.attached = true;
    renderer.addActor(entry.actor);
  };

  const holdSubmitted = (keyString: string, entry: TileActors): void => {
    tiles.set(keyString, entry);
    submittedPoints += entry.tile.pointCount;
    submittedBytes += entry.resourceBytes;
  };

  const dropSubmitted = (keyString: string, entry: TileActors): void => {
    tiles.delete(keyString);
    submittedPoints -= entry.tile.pointCount;
    submittedBytes -= entry.resourceBytes;
  };

  const holdPooled = (keyString: string, entry: TileActors): void => {
    pendingRelease.set(keyString, entry);
    pooledPoints += entry.tile.pointCount;
    pooledBytes += entry.resourceBytes;
    // Retained attached as well as resident. Detaching would let the renderer
    // delete the prop's view node and release its buffers, so the two tiles in
    // three that come back to the same key (`reusedTiles` against
    // `builtTiles`) would each pay a full upload to be drawn again.
    for (const span of entry.spans) span.actor.setVisibility(false);
    entry.pooledSerial = pooledSerial++;
  };

  const dropPooled = (keyString: string, entry: TileActors): void => {
    pendingRelease.delete(keyString);
    pooledPoints -= entry.tile.pointCount;
    pooledBytes -= entry.resourceBytes;
  };

  /** A live scalar setter accepts only a usable, actually-different value. */
  const acceptsScalar = (next: number, current: number): boolean =>
    !disposed && Number.isFinite(next) && next > 0 && next !== current;

  /** The tile's planned draw count; the whole tile when no plan is set. */
  const prefixFor = (keyString: string, entry: TileActors): number =>
    pointPrefixes === null
      ? entry.tile.pointCount
      : Math.min(
          entry.tile.pointCount,
          Math.max(0, Math.floor(pointPrefixes.get(keyString) ?? 0)),
        );

  const setDrawnPointCount = (entry: TileActors, count: number): void => {
    entry.drawnPointCount = count;
    for (const span of entry.spans)
      setMapperPointCount(
        span.mapper,
        Math.min(span.count, Math.max(0, count - span.firstPoint)),
      );
  };

  /** Re-read every tile's planned prefix; true when any drawn count moved. */
  const refreshPrefixes = (): boolean => {
    let changed = false;
    for (const [keyString, entry] of tiles) {
      const count = prefixFor(keyString, entry);
      if (count === entry.drawnPointCount) continue;
      setDrawnPointCount(entry, count);
      changed = true;
    }
    return changed;
  };

  const applyTileState = (keyString: string, entry: TileActors): void => {
    setDrawnPointCount(entry, prefixFor(keyString, entry));
    const matrix = translatedMatrix(baseMatrix, entry.tile.origin);
    for (const { actor, mapper } of entry.spans) {
      mapper.setPointSizeScale(devicePixelRatio);
      setActorPointSize(actor, diameterCssPx);
      actor.setVisibility(visible);
      actor.setUserMatrix(matrix);
    }
  };

  // Point positions are XYZ floats. The mapper maps source RGB to RGBA.
  const pointBytes = (tile: TileData) => (tile.rgb ? 16 : 12);
  const payloadPoints = (tile: TileData) =>
    Math.floor(tile.positions.length / 3);
  const pointsPerSpan = (tile: TileData): number =>
    maxSpanBytes
      ? Math.max(1, Math.floor(maxSpanBytes / pointBytes(tile)))
      : Math.max(1, payloadPoints(tile));
  const resourceBytes = (tile: TileData): number =>
    tileBytes(tile) +
    64 * Math.max(0, Math.ceil(payloadPoints(tile) / pointsPerSpan(tile)) - 1);

  const createTile = ({ tile, memory }: TileAddition): TileActors => {
    builtTiles += 1;
    memory?.setCpu([
      tile.positions.buffer as ArrayBuffer,
      ...(tile.rgb ? [tile.rgb.buffer as ArrayBuffer] : []),
    ]);
    return {
      tile,
      spans: [],
      resourceBytes: resourceBytes(tile),
      drawnPointCount: 0,
      pooledSerial: 0,
      memory,
    };
  };

  const releaseSpan = (entry: PointSpan): void => {
    safeCall(entry.releaseViewResources);
    if (entry.attached) safeCall(() => renderer.removeActor(entry.actor));
    safeCall(() => entry.actor.delete?.());
    safeCall(() => entry.mapper.delete?.());
    safeCall(() => entry.polyData.delete?.());
  };

  const releaseTile = (entry: TileActors): void => {
    for (const span of entry.spans) releaseSpan(span);
    entry.memory?.release();
  };

  const createSpan = (
    { tile, spans }: TileActors,
    firstPoint: number,
    count: number,
  ): void => {
    let polyData: VtkPolyData | undefined;
    let mapper: VtkPointSpriteMapper | undefined;
    let actor: VtkActor | undefined;
    let span: PointSpan | undefined;
    try {
      polyData = vtkPolyData.newInstance();
      const first = 3 * firstPoint;
      const end = first + 3 * count;
      const wholeTile = first === 0 && count === payloadPoints(tile);
      polyData
        .getPoints()
        .setData(
          wholeTile ? tile.positions : tile.positions.subarray(first, end),
          3,
        );
      if (tile.rgb) {
        polyData.getPointData().setScalars(
          vtkDataArray.newInstance({
            name: "RGB",
            values: wholeTile ? tile.rgb : tile.rgb.subarray(first, end),
            numberOfComponents: 3,
          }),
        );
      }
      mapper = vtkPointSpriteMapper.newInstance({ scaleFactor: 0 });
      mapper.setInputData(polyData);
      mapper.setStatic?.(true);
      actor = vtkActor.newInstance();
      actor.setMapper(mapper);
      actor.setForceOpaque(true);
      actor.setVisibility(false);
      span = { actor, mapper, polyData, firstPoint, count, attached: false };
      const window = submissions ? options.getOpenGLRenderWindow?.() : null;
      const gl = window?.getContext();
      if (window && gl && !gl.isContextLost() && count > 0) {
        attach(span);
        const prepared = window.preparePolygonalGeometry(renderer, actor);
        span.releaseViewResources = prepared.release;
        gpuGeometryUploadBytes += prepared.bytes;
      }
      spans.push(span);
    } catch (error) {
      if (span) releaseSpan(span);
      else {
        safeCall(() => actor?.delete?.());
        safeCall(() => mapper?.delete?.());
        safeCall(() => polyData?.delete?.());
      }
      throw error;
    }
  };

  /** Reuse is only sound while an entry still holds the payload being added. */
  const holdsPayload = (entry: TileActors, tile: TileData): boolean =>
    entry.tile.positions === tile.positions && entry.tile.rgb === tile.rgb;

  // Only the pool is trimmable: the submitted set is what the controller
  // decided fits its own memory ceiling, and dropping an actor from it would
  // punch a hole nothing would ever refill (the controller sees no change).
  const trimPool = (additionalBytes = 0): void => {
    // Deleting the entry just yielded is well-defined for a Map iterator, and
    // iteration stays in insertion order — so this sheds oldest-first without
    // rebuilding an iterator per eviction.
    for (const [key, entry] of pendingRelease) {
      if (
        submittedBytes + pooledBytes + preparedBytes + additionalBytes <=
        resourceCeilingBytes
      ) {
        return;
      }
      dropPooled(key, entry);
      releaseTile(entry);
    }
  };

  const takeReusable = (
    addition: TileAddition,
  ): PreparedAddition | undefined => {
    const { key, tile, memory } = addition;
    const keyString = keyToString(key);
    const current = tiles.get(keyString);
    if (current && holdsPayload(current, tile)) {
      memory?.release();
      return { addition };
    }
    const pooled = pendingRelease.get(keyString);
    if (pooled) {
      dropPooled(keyString, pooled);
      if (holdsPayload(pooled, tile)) {
        memory?.release();
        return { addition, entry: pooled, reused: true };
      }
      releaseTile(pooled);
    }
  };

  const acceptAdded = (prepared: PreparedAddition): boolean => {
    const {
      addition: { key, tile },
      entry,
      reused,
    } = prepared;
    const keyString = keyToString(key);
    const current = tiles.get(keyString);
    if (!entry) {
      if (
        !current ||
        (current.tile.pointCount === tile.pointCount &&
          current.tile.origin.every(
            (value, index) => value === tile.origin[index],
          ))
      )
        return false;
      const previous = current.tile;
      current.tile = tile;
      try {
        applyTileState(keyString, current);
      } catch (error) {
        current.tile = previous;
        throw error;
      }
      submittedPoints += tile.pointCount - previous.pointCount;
      return true;
    }

    // Establish the new actor while the previous presentation is still live.
    entry.tile = tile;
    applyTileState(keyString, entry);
    for (const span of entry.spans) attach(span);
    if (current) {
      dropSubmitted(keyString, current);
      releaseTile(current);
    }
    holdSubmitted(keyString, entry);
    if (reused) {
      reusedTiles += 1;
      const bucket = depthBucket(pooledSerial - 1 - entry.pooledSerial);
      reuseDepths[bucket] = (reuseDepths[bucket] ?? 0) + 1;
    }
    return true;
  };

  const retire = (key: string): boolean => {
    const entry = tiles.get(key);
    if (!entry) return false;
    dropSubmitted(key, entry);
    if (
      entry.memory &&
      options.memory &&
      !options.memory.replacementAvailable(entry.memory)
    )
      releaseTile(entry);
    else holdPooled(key, entry);
    return true;
  };

  const changedWork = (): void => {
    workRevision += 1;
    options.onWorkChange?.();
  };

  const withdraw = (key: string, entry: PendingAddition): void => {
    pending.delete(key);
    entry.job?.cancel();
    const realized = entry.prepared?.entry ?? entry.building;
    if (realized) {
      preparedBytes -= realized.resourceBytes;
      releaseTile(realized);
    } else entry.addition.memory?.release();
  };

  const commitPending = (): void => {
    const next = desiredPrefixes;
    pointPrefixes = next;
    for (const [key, item] of pending) {
      if (!item.prepared) continue;
      pending.delete(key);
      const entry = item.prepared.entry;
      if (entry) preparedBytes -= entry.resourceBytes;
      try {
        acceptAdded(item.prepared);
      } catch (error) {
        if (entry) releaseTile(entry);
        options.onTileError?.(item.addition, error);
      }
    }
    // Retain accepted coverage while a replacement is preparing or unavailable.
    const wantPoints =
      next === null || [...next.values()].some((count) => count > 0);
    const hasCoverage =
      next === null ||
      [...next].some(
        ([key, count]) => count > 0 && tiles.has(key) && !retirements.has(key),
      );
    if (!visible || !wantPoints || (!pending.size && hasCoverage)) {
      for (const key of retirements) retire(key);
      retirements.clear();
    } else if (next) {
      const retained = new Map(next);
      for (const [key, entry] of tiles)
        if (!next.has(key)) retained.set(key, entry.drawnPointCount);
      pointPrefixes = retained;
    }
    refreshPrefixes();
    trimPool();
    changedWork();
  };

  const requestPublication = (): void => {
    if (!submissions || publication || disposed) return;
    publication = submissions.enqueue({
      bytes: 0,
      requiresPaint: true,
      run: () => {
        publication = undefined;
        commitPending();
      },
      onError: options.onError,
    });
  };

  const clearTiles = (): void => {
    publication?.cancel();
    publication = undefined;
    for (const [key, item] of pending) withdraw(key, item);
    retirements.clear();
    for (const [key, entry] of tiles) {
      dropSubmitted(key, entry);
      releaseTile(entry);
    }
    for (const [key, entry] of pendingRelease) {
      dropPooled(key, entry);
      releaseTile(entry);
    }
    changedWork();
  };

  return {
    reserveTile(key, tile) {
      const held = tiles.get(key) ?? pendingRelease.get(key);
      if (held && holdsPayload(held, tile)) return held.memory?.share();
      const staged = pending.get(key);
      if (staged?.addition.tile === tile)
        return (
          staged.prepared?.entry?.memory ?? staged.addition.memory
        )?.share();
      const memory = options.memory;
      if (!memory) return undefined;
      const bytes = resourceBytes(tile);
      trimPool(bytes);
      try {
        return memory.reserveRenderer(bytes);
      } catch (error) {
        if (
          !submissions ||
          !desiredPrefixes ||
          !(error instanceof ResourcePolicyError) ||
          error.domain !== "renderer"
        )
          throw error;
        // A prefetched replacement may overlap the previously displayed point
        // tiles. The pool owns each allocation's credit and the overlap cap.
        const replacements = [...tiles].flatMap(([id, entry]) =>
          !desiredPrefixes!.has(id) &&
          entry.memory &&
          memory.replacementAvailable(entry.memory)
            ? [entry.memory]
            : [],
        );
        if (!replacements.length) throw error;
        return memory.reserveRenderer(bytes, { replacing: replacements });
      }
    },
    trimReuse() {
      for (const [key, entry] of pendingRelease) {
        dropPooled(key, entry);
        releaseTile(entry);
      }
    },
    clearTiles() {
      if (disposed) return;
      const needsPaint = visible && tiles.size > 0;
      clearTiles();
      if (needsPaint) scheduleRender();
    },
    *presentedTiles() {
      if (!visible || disposed) return;
      for (const [keyString, entry] of tiles)
        if (entry.drawnPointCount > 0)
          yield {
            keyString,
            tile: entry.tile,
            pointCount: entry.drawnPointCount,
          };
    },
    applyDrawPlan(plan) {
      if (disposed) return;
      desiredPrefixes = new Map(
        plan.entries.map(({ key, pointCount }) => [
          keyToString(key),
          finiteNonNegative(pointCount) ? Math.floor(pointCount) : 0,
        ]),
      );
      if (submissions) {
        const next = desiredPrefixes;
        const hasCoverage =
          ![...next.values()].some((count) => count > 0) ||
          [...tiles.keys()].some((key) => (next.get(key) ?? 0) > 0);
        if (
          hasCoverage &&
          [...tiles].some(
            ([key, entry]) =>
              Math.min(entry.tile.pointCount, next.get(key) ?? 0) !==
              entry.drawnPointCount,
          )
        )
          requestPublication();
      } else {
        pointPrefixes = desiredPrefixes;
        if (refreshPrefixes()) scheduleRender();
      }
    },

    applyBatch(batch) {
      if (disposed) {
        for (const addition of batch.added) addition.memory?.release();
        return;
      }
      if (!submissions) {
        let changed = false;
        for (const key of batch.removed)
          changed = retire(keyToString(key)) || changed;
        for (const [index, addition] of batch.added.entries()) {
          let prepared: PreparedAddition | undefined;
          try {
            prepared = takeReusable(addition);
            if (!prepared) {
              trimPool(resourceBytes(addition.tile));
              const entry = createTile(addition);
              prepared = { addition, entry };
              createSpan(entry, 0, payloadPoints(addition.tile));
            }
            changed = acceptAdded(prepared) || changed;
          } catch (error) {
            if (prepared?.entry) releaseTile(prepared.entry);
            else addition.memory?.release();
            for (const remainder of batch.added.slice(index + 1))
              remainder.memory?.release();
            throw error;
          }
        }
        trimPool();
        if (changed) {
          changedWork();
          scheduleRender();
        }
        return;
      }
      for (const key of batch.removed) {
        const id = keyToString(key);
        const item = pending.get(id);
        if (item) withdraw(id, item);
        if (tiles.has(id)) retirements.add(id);
      }
      for (const addition of batch.added) {
        const key = keyToString(addition.key);
        retirements.delete(key);
        const previous = pending.get(key);
        if (previous?.addition.tile === addition.tile) {
          addition.memory?.release();
          continue;
        }
        if (previous) withdraw(key, previous);
        const item: PendingAddition = { addition };
        pending.set(key, item);
        const fail = (error: unknown): void => {
          if (pending.get(key) !== item) return;
          withdraw(key, item);
          changedWork();
          options.onTileError?.(addition, error);
        };
        try {
          item.prepared = takeReusable(addition);
          if (item.prepared) {
            if (item.prepared.entry)
              preparedBytes += item.prepared.entry.resourceBytes;
            requestPublication();
            continue;
          }

          const { tile } = addition;
          const points = pointsPerSpan(tile);
          const total = payloadPoints(tile);
          // Each actor covers a contiguous span of the original progressive
          // order. A logical tile publishes only after every span is ready.
          const steps = function* () {
            let first = 0;
            do {
              const count = Math.min(points, total - first);
              const start = first;
              first += count;
              yield {
                bytes: count * pointBytes(tile),
                run() {
                  if (pending.get(key) !== item || disposed) return;
                  if (!item.building) {
                    trimPool(resourceBytes(tile));
                    item.building = createTile(addition);
                    preparedBytes += item.building.resourceBytes;
                  }
                  createSpan(item.building, start, count);
                  if (start + count === total) {
                    item.prepared = { addition, entry: item.building };
                    item.building = undefined;
                    requestPublication();
                  }
                },
              };
            } while (first < total);
          };
          item.job = enqueueSubmissionSteps(
            submissions,
            steps(),
            fail,
            changedWork,
          );
        } catch (error) {
          fail(error);
        }
      }
      if (retirements.size && (pending.size === 0 || !visible))
        requestPublication();
      changedWork();
    },

    setBaseMatrix(matrix) {
      if (disposed) return;
      const next = matrix ?? IDENTITY;
      if (sameMatrix(baseMatrix, next)) return;
      // Keep a value snapshot: vtk.js may mutate and reuse the same UserMatrix
      // object, and the next update still needs to detect that visual change.
      baseMatrix = Array.from(next);
      for (const entry of tiles.values()) {
        const translated = translatedMatrix(baseMatrix, entry.tile.origin);
        for (const span of entry.spans) span.actor.setUserMatrix(translated);
      }
      scheduleRender();
    },

    setPointDiameterCssPx(nextDiameterCssPx) {
      if (!acceptsScalar(nextDiameterCssPx, diameterCssPx)) return;
      diameterCssPx = nextDiameterCssPx;
      for (const entry of tiles.values()) {
        for (const span of entry.spans)
          setActorPointSize(span.actor, nextDiameterCssPx);
      }
      scheduleRender();
    },

    setDevicePixelRatio(nextDevicePixelRatio) {
      if (!acceptsScalar(nextDevicePixelRatio, devicePixelRatio)) return;
      devicePixelRatio = nextDevicePixelRatio;
      for (const entry of tiles.values()) {
        for (const span of entry.spans)
          span.mapper.setPointSizeScale(nextDevicePixelRatio);
      }
      scheduleRender();
    },

    setVisible(nextVisible) {
      if (disposed || nextVisible === visible) return;
      visible = nextVisible;
      // Visibility is reversible on its own terms: hiding must not destroy
      // state that only new controller batches could rebuild, or a show with
      // no selection change behind it would leave the cloud blank forever.
      // Pooled entries stay hidden either way — they are not submitted.
      for (const entry of tiles.values())
        for (const span of entry.spans) span.actor.setVisibility(visible);
      scheduleRender();
    },

    setResourceCeilingBytes(bytes) {
      if (disposed || !finiteNonNegative(bytes)) return;
      resourceCeilingBytes = bytes;
      trimPool();
    },

    workState() {
      return {
        workRevision,
        gpuResidentBytes: submittedBytes + pooledBytes + preparedBytes,
        submittedBytes,
        pendingJobs: pending.size + (publication ? 1 : 0),
      };
    },

    stats() {
      let preparedTiles = 0;
      let preparedPoints = 0;
      for (const item of pending.values()) {
        if (!item.building && !item.prepared?.entry) continue;
        preparedTiles += 1;
        preparedPoints += item.addition.tile.pointCount;
      }
      let drawnPoints = 0;
      let drawnTiles = 0;
      if (visible) {
        for (const entry of tiles.values()) {
          drawnPoints += entry.drawnPointCount;
          if (entry.drawnPointCount > 0) drawnTiles += 1;
        }
      }
      return {
        workRevision,
        submittedTiles: tiles.size,
        submittedPoints,
        submittedBytes,
        pooledTiles: pendingRelease.size,
        pooledPoints,
        pooledBytes,
        reusedTiles,
        builtTiles,
        pendingJobs: pending.size + (publication ? 1 : 0),
        gpuGeometryUploadBytes,
        reuseDepths: [...reuseDepths],
        gpuResidentTiles: tiles.size + pendingRelease.size + preparedTiles,
        gpuResidentPoints: submittedPoints + pooledPoints + preparedPoints,
        gpuResidentBytes: submittedBytes + pooledBytes + preparedBytes,
        resourceCeilingBytes,
        drawnTiles,
        drawnPoints,
        drawnFraction:
          visible && submittedPoints > 0 ? drawnPoints / submittedPoints : 0,
        visible,
        diameterCssPx,
        devicePixelRatio,
      };
    },

    activeKeys: () => ({
      submitted: [...tiles.keys()].sort(),
      pooled: [...pendingRelease.keys()].sort(),
    }),

    dispose() {
      if (disposed) return;
      disposed = true;
      clearTiles();
    },
  };
};
