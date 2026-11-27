/** Pure camera-driven traversal for explicit and implicit REPLACE hierarchies. */

import {
  prepareView,
  type CameraView,
  type Plane,
  type PreparedView,
} from "../core/camera";
import {
  IDENTITY,
  multiply,
  transformPoint,
  transformVector,
} from "../core/mat4";
import {
  substituteImplicitTemplate,
  type ImplicitTileAddress,
  type TilesetBox,
  type TilesetImplicitTiling,
  type TilesetTile,
} from "./tilesetSource";
import {
  quadtreeMortonIndex,
  subtreeTileIndex,
  type ParsedSubtree,
} from "./subtree";
import type {
  ContentQueueEntrySnapshot,
  TileContentRequest,
} from "./contentQueue";
import type { GeometricErrorScale } from "./memberTypes";

/** Immutable implicit-hierarchy state sampled at the start of one pass. */
export type SubtreeHierarchyState = {
  readonly revision: string;
  readonly configGeneration: number;
  readonly entries: readonly ContentQueueEntrySnapshot[];
  readonly subtreeById: ReadonlyMap<string, ParsedSubtree>;
};

export type TileReadiness =
  | "unloaded"
  | "loading"
  | "decoded"
  | "submitted"
  | "failed";

export type TilesetTraversalOptions = {
  readonly root: TilesetTile;
  readonly camera: CameraView;
  /** The finest threshold configured for this member. */
  readonly maximumScreenSpaceErrorPx: number;
  /** Normalized allocation; the threshold relaxes by `1 / max(q, 0.05)`. */
  readonly qualityFraction?: number;
  /** Caller-owned anchor × tileset-to-scene transform. */
  readonly modelMatrix?: readonly number[];
  /** Maximum-axis 3D Tiles convention, or local XY for terrain-derived error. */
  readonly geometricErrorScale?: GeometricErrorScale;
  readonly readiness: (tileId: string) => TileReadiness;
  /** Physical publication when it follows logical admission asynchronously. */
  readonly presented?: (tileId: string) => boolean;
  /**
   * Whether a contentful tile may still be refined past, default true.
   *
   * Answers "is there any point asking for this tile's children", which is a
   * different question from whether they have arrived. A caller that knows a
   * tile's replacement set can never be drawn — it does not fit the memory
   * allowance — says so here, and the tile is selected as the frontier instead
   * of being refined past into content that will be requested forever.
   */
  readonly refinable?: (tileId: string) => boolean;
  /** Skip this content representation and seek coverage below it. */
  readonly unavailable?: (tileId: string) => boolean;
  /** Immutable hierarchy state sampled at the beginning of this pass. */
  readonly subtrees?: SubtreeHierarchyState | null;
};

export type TilesetTraversalResult = {
  /** Finest camera-selected content, excluding ancestors held for readiness. */
  readonly desiredTileIds: readonly string[];
  /** Content that should be available: desired tiles plus required fallbacks. */
  readonly requestedTileIds: readonly string[];
  /** Submitted draw set with REPLACE fallback applied. */
  readonly drawnTileIds: readonly string[];
  readonly recoveringCoverage: boolean;
  readonly effectiveScreenSpaceErrorPx: number;
  /**
   * Screen-space error of the root's own content, before refinement.
   *
   * This is the member's demand in the one unit that is comparable across
   * formats: how wrong the coarsest thing it can draw currently looks. Zero
   * when the root is culled.
   */
  readonly rootScreenSpaceErrorPx: number;
  /** Visible unknown boundaries. The member owns fetching and retry policy. */
  readonly neededSubtreeRequests: readonly TileContentRequest[];
  /** Explicit source tiles plus implicit tiles materialized for this pass. */
  readonly tileById: ReadonlyMap<string, TilesetTile>;
};

type Vec3 = readonly [number, number, number];

type WorldBox = {
  readonly center: Vec3;
  readonly axes: readonly [Vec3, Vec3, Vec3];
};

type TilePlacement = {
  transform: readonly number[];
  bounds: WorldBox;
};

const worldBox = (box: TilesetBox, matrix: readonly number[]): WorldBox => {
  const h = box.halfAxes;
  return {
    center: transformPoint(matrix, box.center),
    axes: [
      transformVector(matrix, [h[0], h[1], h[2]]),
      transformVector(matrix, [h[3], h[4], h[5]]),
      transformVector(matrix, [h[6], h[7], h[8]]),
    ],
  };
};

const dot = (left: Vec3, right: Vec3): number =>
  left[0] * right[0] + left[1] * right[1] + left[2] * right[2];

/** Exact plane-vs-oriented-box rejection. */
const boxIntersectsFrustum = (
  box: WorldBox,
  planes: readonly Plane[],
): boolean => {
  for (const plane of planes) {
    const radius =
      Math.abs(dot(plane.normal, box.axes[0])) +
      Math.abs(dot(plane.normal, box.axes[1])) +
      Math.abs(dot(plane.normal, box.axes[2]));
    if (dot(plane.normal, box.center) + plane.d + radius < 0) return false;
  }
  return true;
};

const boxDistance = (box: WorldBox, point: Vec3): number => {
  // A conservative world AABB is sufficient for SSE: it may refine early,
  // but never drops detail because an oriented or sheared box was too near.
  const radius: Vec3 = [
    Math.abs(box.axes[0][0]) +
      Math.abs(box.axes[1][0]) +
      Math.abs(box.axes[2][0]),
    Math.abs(box.axes[0][1]) +
      Math.abs(box.axes[1][1]) +
      Math.abs(box.axes[2][1]),
    Math.abs(box.axes[0][2]) +
      Math.abs(box.axes[1][2]) +
      Math.abs(box.axes[2][2]),
  ];
  const dx = Math.max(Math.abs(point[0] - box.center[0]) - radius[0], 0);
  const dy = Math.max(Math.abs(point[1] - box.center[1]) - radius[1], 0);
  const dz = Math.max(Math.abs(point[2] - box.center[2]) - radius[2], 0);
  return Math.hypot(dx, dy, dz);
};

const maximumScale = (matrix: readonly number[]): number =>
  Math.max(
    Math.hypot(matrix[0]!, matrix[1]!, matrix[2]!),
    Math.hypot(matrix[4]!, matrix[5]!, matrix[6]!),
    Math.hypot(matrix[8]!, matrix[9]!, matrix[10]!),
  );

const horizontalScale = (matrix: readonly number[]): number =>
  Math.max(
    Math.hypot(matrix[0]!, matrix[1]!, matrix[2]!),
    Math.hypot(matrix[4]!, matrix[5]!, matrix[6]!),
  );

const screenSpaceError = (
  tile: TilesetTile,
  matrix: readonly number[],
  box: WorldBox,
  view: PreparedView,
  errorScale: GeometricErrorScale,
): number => {
  const scale =
    errorScale === "horizontal"
      ? horizontalScale(matrix)
      : maximumScale(matrix);
  return view.screenSpaceError(
    tile.geometricError * scale,
    boxDistance(box, view.view.position),
  );
};

type VisitResult = {
  readonly visible: boolean;
  readonly coverageSubmitted: boolean;
  readonly coveragePresented: boolean;
  readonly recoveringCoverage?: boolean;
  /** False when complete coverage requires an excluded representation. */
  readonly coveragePossible: boolean;
  readonly desired: string[];
  readonly requested: string[];
  readonly drawn: string[];
};

type MaterializedHierarchy = {
  readonly root: TilesetTile;
  readonly tileById: ReadonlyMap<string, TilesetTile>;
  readonly subtreeRequestByTileId: ReadonlyMap<string, TileContentRequest>;
  readonly unknownRequestByTileId: ReadonlyMap<string, TileContentRequest>;
};

const implicitTileId = (address: ImplicitTileAddress): string => {
  if (address.level === 0) return "root";
  const parts = ["root"];
  for (let bit = address.level - 1; bit >= 0; bit -= 1) {
    parts.push(String(((address.x >> bit) & 1) + 2 * ((address.y >> bit) & 1)));
  }
  return parts.join("/");
};

const childAddress = (
  parent: ImplicitTileAddress,
  quadrant: number,
): ImplicitTileAddress =>
  Object.freeze({
    level: parent.level + 1,
    x: parent.x * 2 + (quadrant & 1),
    y: parent.y * 2 + ((quadrant >> 1) & 1),
  });

const implicitRequest = (
  descriptor: TilesetImplicitTiling,
  address: ImplicitTileAddress,
): TileContentRequest =>
  Object.freeze({
    id: `subtree/${address.level}/${address.x}/${address.y}`,
    url: substituteImplicitTemplate(descriptor.subtreeUrlTemplate, address),
  });

const implicitContent = (
  descriptor: TilesetImplicitTiling,
  address: ImplicitTileAddress,
): { readonly contentUri: string; readonly contentUrl: string } =>
  Object.freeze({
    contentUri: substituteImplicitTemplate(
      descriptor.contentUriTemplate,
      address,
    ),
    contentUrl: substituteImplicitTemplate(
      descriptor.contentUrlTemplate,
      address,
    ),
  });

const explicitHierarchy = (root: TilesetTile): MaterializedHierarchy => {
  const tileById = new Map<string, TilesetTile>();
  const collect = (tile: TilesetTile): void => {
    tileById.set(tile.id, tile);
    for (const child of tile.children) collect(child);
  };
  collect(root);
  return {
    root,
    tileById,
    subtreeRequestByTileId: new Map(),
    unknownRequestByTileId: new Map(),
  };
};

const implicitHierarchy = (
  documentRoot: TilesetTile,
  snapshot: SubtreeHierarchyState | null | undefined,
): MaterializedHierarchy => {
  const descriptor = documentRoot.implicitTiling!;
  const tileById = new Map<string, TilesetTile>();
  const subtreeRequestByTileId = new Map<string, TileContentRequest>();
  const unknownRequestByTileId = new Map<string, TileContentRequest>();
  const entryById = new Map(
    snapshot?.entries.map((entry) => [entry.id, entry]) ?? [],
  );

  const proxy = (
    address: ImplicitTileAddress,
    boundingVolume: TilesetBox,
  ): TilesetTile => {
    const id = implicitTileId(address);
    unknownRequestByTileId.set(id, implicitRequest(descriptor, address));
    const tile = Object.freeze({
      id,
      geometricError: documentRoot.geometricError / 2 ** address.level,
      boundingVolume,
      transform: address.level === 0 ? documentRoot.transform : IDENTITY,
      worldTransform: documentRoot.worldTransform,
      children: Object.freeze([]),
      implicitAddress: address,
    });
    tileById.set(id, tile);
    return tile;
  };

  const buildSubtree = (
    subtreeRoot: ImplicitTileAddress,
    fallbackBounds: TilesetBox,
  ): TilesetTile => {
    const request = implicitRequest(descriptor, subtreeRoot);
    const parsed = snapshot?.subtreeById.get(request.id);
    if (!parsed) return proxy(subtreeRoot, fallbackBounds);

    const buildTile = (
      localLevel: number,
      localX: number,
      localY: number,
      address: ImplicitTileAddress,
    ): TilesetTile => {
      const tileIndex = subtreeTileIndex(localLevel, localX, localY);
      // parseSubtree gives every available tile its metadata bounds.
      const bounds = parsed.tileBoundingBoxes[tileIndex]!;
      const children: TilesetTile[] = [];
      if (address.level + 1 < descriptor.availableLevels) {
        if (localLevel + 1 < descriptor.subtreeLevels) {
          for (let quadrant = 0; quadrant < 4; quadrant += 1) {
            const childLocalX = localX * 2 + (quadrant & 1);
            const childLocalY = localY * 2 + ((quadrant >> 1) & 1);
            const childIndex = subtreeTileIndex(
              localLevel + 1,
              childLocalX,
              childLocalY,
            );
            if (!parsed.tileAvailability.isAvailable(childIndex)) continue;
            children.push(
              buildTile(
                localLevel + 1,
                childLocalX,
                childLocalY,
                childAddress(address, quadrant),
              ),
            );
          }
        } else {
          for (let quadrant = 0; quadrant < 4; quadrant += 1) {
            const childLocalX = localX * 2 + (quadrant & 1);
            const childLocalY = localY * 2 + ((quadrant >> 1) & 1);
            const childIndex = quadtreeMortonIndex(
              descriptor.subtreeLevels,
              childLocalX,
              childLocalY,
            );
            if (!parsed.childSubtreeAvailability.isAvailable(childIndex))
              continue;
            const nextAddress = childAddress(address, quadrant);
            children.push(buildSubtree(nextAddress, bounds));
          }
        }
      }
      const content = parsed.contentAvailability.isAvailable(tileIndex)
        ? implicitContent(descriptor, address)
        : undefined;
      const id = implicitTileId(address);
      const tile: TilesetTile = Object.freeze({
        id,
        geometricError: documentRoot.geometricError / 2 ** address.level,
        boundingVolume: bounds,
        transform: address.level === 0 ? documentRoot.transform : IDENTITY,
        worldTransform: documentRoot.worldTransform,
        ...content,
        children: Object.freeze(children),
        implicitAddress: address,
        ...(address.level === 0 ? { implicitTiling: descriptor } : {}),
      });
      subtreeRequestByTileId.set(id, request);
      tileById.set(id, tile);
      return tile;
    };

    // A failed selected entry remains an unknown boundary and is deliberately
    // not converted to empty availability.
    if (
      ["failed", "blocked"].includes(entryById.get(request.id)?.status ?? "")
    ) {
      return proxy(subtreeRoot, fallbackBounds);
    }
    return buildTile(0, 0, 0, subtreeRoot);
  };

  const rootAddress = Object.freeze({ level: 0, x: 0, y: 0 });
  return {
    root: buildSubtree(rootAddress, documentRoot.boundingVolume),
    tileById,
    subtreeRequestByTileId,
    unknownRequestByTileId,
  };
};

const hasContent = (tile: TilesetTile): boolean =>
  tile.contentUrl !== undefined;

export const sameSubtreeHierarchy = (
  left: SubtreeHierarchyState | null | undefined,
  right: SubtreeHierarchyState | null | undefined,
): boolean => {
  if (left === right) return true;
  if (
    !left ||
    !right ||
    left.revision !== right.revision ||
    left.configGeneration !== right.configGeneration ||
    left.subtreeById.size !== right.subtreeById.size
  )
    return false;
  for (const [id, subtree] of left.subtreeById) {
    if (right.subtreeById.get(id) !== subtree) return false;
  }
  const failed = (snapshot: SubtreeHierarchyState) =>
    new Set(
      snapshot.entries
        .filter(
          (entry) => entry.status === "failed" || entry.status === "blocked",
        )
        .map((entry) => entry.id),
    );
  const a = failed(left),
    b = failed(right);
  return a.size === b.size && [...a].every((id) => b.has(id));
};

/** Per-member cache of one hierarchy, bounded by the current subtree store.
 * Camera, readiness and quality are still evaluated on every traversal.
 */
export const createTilesetTraversal = () => {
  let root: TilesetTile | undefined;
  let snapshot: SubtreeHierarchyState | null | undefined;
  let hierarchy: MaterializedHierarchy | undefined;
  let modelMatrix: readonly number[] | undefined;
  const placements = new Map<TilesetTile, TilePlacement>();
  return (options: TilesetTraversalOptions): TilesetTraversalResult => {
    if (
      root !== options.root ||
      !sameSubtreeHierarchy(snapshot, options.subtrees)
    ) {
      hierarchy = undefined;
      placements.clear();
    }
    const nextMatrix = options.modelMatrix ?? IDENTITY;
    if (
      !modelMatrix ||
      nextMatrix.some((value, i) => value !== modelMatrix![i])
    ) {
      modelMatrix = Array.from(nextMatrix);
      placements.clear();
    }
    root = options.root;
    snapshot = options.subtrees;
    hierarchy ??= root.implicitTiling
      ? implicitHierarchy(root, snapshot)
      : explicitHierarchy(root);
    return traverseHierarchy(options, hierarchy, placements);
  };
};

const traverseHierarchy = (
  options: TilesetTraversalOptions,
  hierarchy: MaterializedHierarchy,
  placements: Map<TilesetTile, TilePlacement>,
): TilesetTraversalResult => {
  if (
    !Number.isFinite(options.maximumScreenSpaceErrorPx) ||
    options.maximumScreenSpaceErrorPx <= 0
  ) {
    throw new RangeError("maximumScreenSpaceErrorPx must be finite and > 0");
  }
  const quality = options.qualityFraction ?? 1;
  if (!Number.isFinite(quality) || quality < 0 || quality > 1) {
    throw new RangeError("qualityFraction must be finite and in [0, 1]");
  }
  const errorScale = options.geometricErrorScale ?? "maximum";
  if (errorScale !== "maximum" && errorScale !== "horizontal") {
    throw new RangeError(
      "geometricErrorScale must be 'maximum' or 'horizontal'",
    );
  }
  const effective = options.maximumScreenSpaceErrorPx / Math.max(quality, 0.05);
  const view = prepareView(options.camera);
  const neededSubtrees = new Map<string, TileContentRequest>();

  const hasSubmittedDescendant = (tile: TilesetTile): boolean =>
    tile.children.some(
      (child) =>
        (hasContent(child) && options.readiness(child.id) === "submitted") ||
        hasSubmittedDescendant(child),
    );

  type Region = { readonly error: number; readonly offset: number };
  type RequestOrder = {
    readonly depth: number;
    readonly offset: number;
    readonly coveredRegion: Region | undefined;
  };
  const requestOrder = new Map<string, RequestOrder>();

  const visit = (
    tile: TilesetTile,
    parentTransform: readonly number[],
    depth: number,
    coveredRegion?: Region,
    retainSubmitted = false,
  ): VisitResult => {
    let placement = placements.get(tile);
    if (!placement) {
      const transform = multiply(parentTransform, tile.transform);
      placement = {
        transform,
        bounds: worldBox(tile.boundingVolume, transform),
      };
      placements.set(tile, placement);
    }
    const { transform: accumulated, bounds } = placement;
    if (!boxIntersectsFrustum(bounds, view.planes)) {
      return {
        visible: false,
        coverageSubmitted: true,
        coveragePresented: true,
        coveragePossible: true,
        desired: [],
        requested: [],
        drawn: [],
      };
    }
    const offset = view.centerRayOffset(bounds.center);
    requestOrder.set(tile.id, { depth, offset, coveredRegion });
    const ownPresented = options.presented?.(tile.id) ?? false;
    const covered = !!coveredRegion || ownPresented;

    const subtreeRequest = hierarchy.subtreeRequestByTileId.get(tile.id);
    if (subtreeRequest && !retainSubmitted)
      neededSubtrees.set(subtreeRequest.id, subtreeRequest);
    const unknownRequest = hierarchy.unknownRequestByTileId.get(tile.id);
    if (unknownRequest) {
      if (!retainSubmitted)
        neededSubtrees.set(unknownRequest.id, unknownRequest);
      const submitted = options.readiness(tile.id) === "submitted";
      return {
        visible: true,
        coverageSubmitted: submitted,
        coveragePresented: covered,
        coveragePossible: true,
        desired: [],
        requested: submitted ? [tile.id] : [],
        drawn: submitted ? [tile.id] : [],
      };
    }

    const authoredContent = hasContent(tile);
    const available = !options.unavailable?.(tile.id);
    const submitted =
      authoredContent && options.readiness(tile.id) === "submitted";
    const contentful = authoredContent && (available || submitted);
    // Contentless tiles are hierarchy nodes, not drawable levels of detail.
    // Their descendants remain reachable regardless of the node's own SSE, and
    // regardless of `refinable` — such a node has no content of its own to
    // stand as a frontier, so refusing to descend would draw nothing at all.
    const error =
      tile.children.length > 0 && (ownPresented || contentful)
        ? screenSpaceError(tile, accumulated, bounds, view, errorScale)
        : 0;
    const childRegion = ownPresented ? { error, offset } : coveredRegion;
    const refine =
      tile.children.length > 0 &&
      (retainSubmitted
        ? !submitted
        : !contentful ||
          (error > effective && (options.refinable?.(tile.id) ?? true)));
    const childResults = refine
      ? tile.children.map((child) =>
          visit(child, accumulated, depth + 1, childRegion, retainSubmitted),
        )
      : [];
    const visibleChildren = childResults.filter((child) => child.visible);

    if (
      !refine ||
      visibleChildren.length === 0 ||
      (!retainSubmitted &&
        contentful &&
        visibleChildren.some((child) => !child.coveragePossible))
    ) {
      if (!contentful) {
        return {
          visible: true,
          // A contentless leaf (or a hierarchy branch whose descendants are
          // all culled) has no draw obligation of its own.
          coverageSubmitted: !authoredContent,
          coveragePresented: covered || !authoredContent,
          coveragePossible: !authoredContent,
          desired: [],
          requested: [],
          drawn: [],
        };
      }
      // A coarse tile still loading can use finer coverage already on the
      // renderer. Intermediate levels may have been retired after refinement.
      const fallbacks =
        !retainSubmitted && !submitted && hasSubmittedDescendant(tile)
          ? tile.children.map((child) =>
              visit(child, accumulated, depth + 1, childRegion, true),
            )
          : [];
      const visibleFallbacks = fallbacks.filter((child) => child.visible);
      return {
        visible: true,
        coverageSubmitted:
          submitted ||
          (visibleFallbacks.length > 0 &&
            visibleFallbacks.every((child) => child.coverageSubmitted)),
        coveragePossible: true,
        coveragePresented:
          covered ||
          (visibleFallbacks.length > 0 &&
            visibleFallbacks.every((child) => child.coveragePresented)),
        recoveringCoverage: visibleFallbacks.some(
          (child) => child.recoveringCoverage,
        ),
        desired: retainSubmitted ? [] : [tile.id],
        requested: [
          ...(!retainSubmitted || submitted ? [tile.id] : []),
          ...fallbacks.flatMap((child) => child.requested),
        ],
        drawn: submitted
          ? [tile.id]
          : fallbacks.flatMap((child) => child.drawn),
      };
    }

    const desired = visibleChildren.flatMap((child) => child.desired);
    const coveragePresented =
      covered || visibleChildren.every((child) => child.coveragePresented);
    const recoveringCoverage = Boolean(
      options.presented && submitted && !coveragePresented,
    );
    if (visibleChildren.every((child) => child.coverageSubmitted)) {
      return {
        visible: true,
        coverageSubmitted: true,
        coveragePossible: true,
        coveragePresented,
        recoveringCoverage:
          recoveringCoverage ||
          visibleChildren.some((child) => child.recoveringCoverage),
        desired,
        requested: [
          ...(recoveringCoverage ? [tile.id] : []),
          ...visibleChildren.flatMap((child) => child.requested),
        ],
        drawn: recoveringCoverage
          ? [tile.id]
          : visibleChildren.flatMap((child) => child.drawn),
      };
    }

    return {
      visible: true,
      coverageSubmitted: submitted,
      coveragePresented,
      recoveringCoverage: visibleChildren.some(
        (child) => child.recoveringCoverage,
      ),
      coveragePossible:
        contentful || visibleChildren.every((child) => child.coveragePossible),
      desired,
      requested: [
        ...(contentful && !retainSubmitted ? [tile.id] : []),
        ...visibleChildren.flatMap((child) => child.requested),
      ],
      // Never partially replace a submitted parent. If it has not arrived yet,
      // any submitted descendants are still useful while initial coverage loads.
      drawn: submitted
        ? [tile.id]
        : visibleChildren.flatMap((child) => child.drawn),
    };
  };

  const rootParent = options.modelMatrix ?? IDENTITY;
  const result = visit(hierarchy.root, rootParent, 0);
  if (hierarchy.subtreeRequestByTileId.size) {
    // A drawn fallback can cross loaded subtree boundaries that refinement
    // no longer visits. Keep its metadata reachable until coverage switches.
    const visited = new Set<string>();
    for (const drawn of result.drawn) {
      for (
        let id = drawn;
        !visited.has(id);
        id = id.slice(0, id.lastIndexOf("/"))
      ) {
        visited.add(id);
        const request = hierarchy.subtreeRequestByTileId.get(id);
        if (request) neededSubtrees.set(request.id, request);
        if (!id.includes("/")) break;
      }
    }
  }

  const requested = [...result.requested].sort((left, right) => {
    const a = requestOrder.get(left)!;
    const b = requestOrder.get(right)!;
    // First restore regions without coverage. Then let the highest-error
    // visible fallback refine without waiting for a whole breadth level.
    const aRegion = a.coveredRegion;
    const bRegion = b.coveredRegion;
    const coverageOrder =
      aRegion && bRegion
        ? bRegion.error - aRegion.error || aRegion.offset - bRegion.offset
        : Number(!!aRegion) - Number(!!bRegion);
    return coverageOrder || a.depth - b.depth || a.offset - b.offset;
  });
  const rootAccumulated = multiply(rootParent, hierarchy.root.transform);
  const rootSse = result.visible
    ? screenSpaceError(
        hierarchy.root,
        rootAccumulated,
        worldBox(hierarchy.root.boundingVolume, rootAccumulated),
        view,
        errorScale,
      )
    : 0;
  return Object.freeze({
    desiredTileIds: Object.freeze(result.desired),
    requestedTileIds: Object.freeze(requested),
    drawnTileIds: Object.freeze(result.drawn),
    recoveringCoverage: result.recoveringCoverage ?? false,
    effectiveScreenSpaceErrorPx: effective,
    rootScreenSpaceErrorPx: rootSse,
    neededSubtreeRequests: Object.freeze([...neededSubtrees.values()]),
    tileById: hierarchy.tileById,
  });
};
