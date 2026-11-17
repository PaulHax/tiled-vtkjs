/** StreamedMember implementation for the constrained 3D Tiles profile. */

import {
  copyCameraView,
  sameCameraView,
  type CameraView,
} from "../core/camera";
import {
  multiply,
  sameMatrix,
  validateAffineMatrix,
  type Mat16,
} from "../core/mat4";
import { safeCall } from "../core/observers";
import {
  ResourcePolicyError,
  resourcePolicyError,
  type MemoryLease,
  type MemoryDomain,
} from "../core/memoryResources";
import { buildTransferList } from "./decode/transfer";
import {
  CULLED,
  importanceFromRootSseCssPx,
  type Allocation,
  type GovernorInputs,
  type StreamedMember,
  type StreamedMemberContext,
} from "../core/streamedMember";
import {
  createContentQueue,
  type ContentQueue,
  type ContentQueueSnapshot,
} from "./contentQueue";
import {
  TileDecodeError,
  type DecodedTileContent,
  type DecodeWasmUrls,
} from "./decode";
import { createMeshContentAdapter } from "./meshContentAdapter";
import {
  TileRenderingProfileError,
  type TileContentAdapter,
  type TileSubmissionOutcome,
  type TileSubmissionRefusal,
} from "./contentAdapter";
import { createMeshPickSet } from "./meshPicking";
import {
  DEFAULT_TILES3D_SUBTREE_CACHE_BYTES,
  type Tiles3dMemberConfig,
  type Tiles3dMemberStats,
  type Tiles3dMemberServices,
} from "./memberTypes";
import {
  validateTiles3dMemberConfig,
  type ResolvedTiles3dConfig,
} from "./memberConfig";
import { createVerticalExaggerationTransform } from "./rtc";
import {
  loadTileset,
  type TilesetSource,
  type TilesetTile,
} from "./tilesetSource";
import { parseSubtree, type ParsedSubtree } from "./subtree";
import {
  createTilesetTraversal,
  type SubtreeHierarchyState,
  type TilesetTraversalResult,
} from "./traversal";

/**
 * Proper ancestor ids of a tile, deepest first.
 *
 * Tile ids are `/`-joined child indices under the tileset root, so an ancestor
 * is exactly a `/`-prefix. Walking them costs the tile's depth, where asking
 * which of a set of ids is an ancestor costs the size of that set.
 */
const ancestorIds = (id: string): string[] => {
  const ancestors: string[] = [];
  for (let cut = id.indexOf("/"); cut !== -1; cut = id.indexOf("/", cut + 1)) {
    ancestors.push(id.slice(0, cut));
  }
  return ancestors.reverse();
};

const errorMessage = (error: unknown): string => {
  if (!(error instanceof Error)) return String(error);
  const messages = [error.message];
  let cause = error.cause;
  while (cause instanceof Error) {
    if (!messages.includes(cause.message)) messages.push(cause.message);
    cause = cause.cause;
  }
  return messages.join(": ");
};

/** Whether two wasm URL sets decode to the same bytes.
 *
 * A host resolves these URLs against the page on every config push, so the
 * object is fresh each time. Only the URLs it names are decode inputs.
 */
const sameDecodeWasm = (
  left: DecodeWasmUrls | undefined,
  right: DecodeWasmUrls | undefined,
): boolean =>
  left === right ||
  (left?.draco?.wrapperUrl === right?.draco?.wrapperUrl &&
    left?.draco?.wasmUrl === right?.draco?.wasmUrl &&
    left?.basis?.encoderUrl === right?.basis?.encoderUrl &&
    left?.basis?.wasmUrl === right?.basis?.wasmUrl);

const decodedBytes = (content: DecodedTileContent): number =>
  content.byteEstimate.geometry + content.byteEstimate.textures;

const workerUrl = (value: string): string =>
  new URL(value, globalThis.location?.href ?? "http://localhost/").toString();

const contentRequests = (
  tileById: ReadonlyMap<string, TilesetTile>,
  ids: readonly string[],
): { readonly id: string; readonly url: string }[] =>
  ids.flatMap((id) => {
    const url = tileById.get(id)?.contentUrl;
    return url === undefined ? [] : [{ id, url }];
  });

export const createTiles3dMember = (
  context: StreamedMemberContext,
  initialConfig: Tiles3dMemberConfig,
  services: Tiles3dMemberServices,
): StreamedMember => {
  let traverseTileset = createTilesetTraversal();
  let config = validateTiles3dMemberConfig(initialConfig);
  let active = true;
  let disposed = false;
  let sourceState: Tiles3dMemberStats["sourceState"] = "idle";
  let source: TilesetSource | null = null;
  let camera: CameraView | null = null;
  let modelMatrix: readonly number[] | null = null;
  let devicePixelRatio = context.devicePixelRatio;
  let allocation: Allocation = {
    qualityFraction: 1,
    memoryBudgetBytes: 0,
    regime: "stationary",
  };
  let interactionDepth = 0;
  let memoryConstrained = false;
  /** A fallback constrains its own subtree. An unaffordable representation
   * can still have affordable descendants.
   */
  const blockedGroupAncestors = new Set<string>();
  const unavailableRepresentations = new Set<string>();
  let constrainedDesiredSignature: string | null = null;
  let configGeneration = 1;
  let loadGeneration = 0;
  let loadController: AbortController | null = null;
  let queue: ContentQueue<DecodedTileContent> | null = null;
  let queueSnapshot: ContentQueueSnapshot | null = null;
  let subtreeQueue: ContentQueue<ParsedSubtree> | null = null;
  let subtreeSnapshot: ContentQueueSnapshot | null = null;
  let subtreeHierarchy: SubtreeHierarchyState | null = null;
  let materializedTileById = new Map<string, TilesetTile>();
  let traversal: TilesetTraversalResult | null = null;
  let errorCount = 0;
  let lastError: string | null = null;
  let workProgressSerial = 0;
  let selectionPasses = 0;
  type RealizationDemand = {
    blocked?: {
      readonly refusal: TileSubmissionRefusal;
      readonly replacement: string | null;
      readonly group: boolean;
      readonly availability: number;
    };
  };
  // These records are scoped to this adapter, which owns the live resources
  // in this view and validates their placement before presenting them.
  const demand = new Map<string, RealizationDemand>();
  let realizationWakeQueued = false;
  let refreshing = false;
  let refreshAgain = false;
  let drawSetDirty = true;
  let sourceMemory: MemoryLease | undefined;
  let sourceRefusal: ResourcePolicyError | null = null;
  let lastPolicyRevision = context.memory.policyRevision();
  let lastAvailability = context.memory.availability();
  let rendererRefusal:
    | { readonly error: ResourcePolicyError; readonly availability: number }
    | undefined;
  let resourceState: Tiles3dMemberStats["resourceState"] = { status: "ready" };
  const setResourceState = (
    next: Tiles3dMemberStats["resourceState"],
  ): void => {
    resourceState = next;
    safeCall(() => config.onResourceState?.(next));
    context.onWorkChange?.();
  };
  const noteResourceRecovery = (): void => {
    if (
      resourceState.status === "blocked" &&
      !sourceRefusal &&
      !rendererRefusal &&
      !memoryConstrained &&
      !blockedGroupAncestors.size &&
      !unavailableRepresentations.size &&
      !queueSnapshot?.blocked &&
      !subtreeSnapshot?.blocked &&
      ![...demand.values()].some((tile) => tile.blocked)
    )
      setResourceState({ status: "ready" });
  };

  const policyRefused = (error: TileSubmissionRefusal): void => {
    if (!(error instanceof ResourcePolicyError)) {
      setResourceState({ status: "blocked", ...error });
      return;
    }
    setResourceState({
      status: "blocked",
      domain: error.domain,
      requestedBytes: error.requestedBytes,
      limitBytes: error.limitBytes,
      retryable: error.retryable,
      ...error.estimate,
    });
  };

  const retryRealizations = (
    wake: "availability" | "presentation",
  ): boolean => {
    const availability = context.memory.availability();
    let retry = false;
    for (const tile of demand.values()) {
      const blocked = tile.blocked;
      if (
        blocked?.refusal instanceof ResourcePolicyError &&
        blocked.refusal.estimate.wake === wake &&
        (wake === "presentation" || availability !== blocked.availability)
      ) {
        tile.blocked = undefined;
        retry = true;
      }
    }
    return retry;
  };

  const onPresentationChange = (): void => {
    syncPickPresentation();
    if (realizationWakeQueued || !demand.size) return;
    realizationWakeQueued = true;
    queueMicrotask(() => {
      realizationWakeQueued = false;
      if (
        !disposed &&
        active &&
        (retryRealizations("presentation") || traversal?.recoveringCoverage)
      )
        refreshSelection();
    });
  };

  const memory = context.memory.createOwner({
    reclaim(domain: MemoryDomain) {
      if (domain === "cpu") {
        queue?.trimCache(0);
        subtreeQueue?.trimCache(0);
        adapter.trimReuse();
      }
      if (domain === "renderer") adapter.trimReuse();
    },
  });

  const report = (error: unknown): void => {
    errorCount += 1;
    lastError = errorMessage(error);
    const cause = error instanceof Error ? error.cause : undefined;
    const unsupported =
      error instanceof TileRenderingProfileError ||
      (error instanceof TileDecodeError && error.stage === "profile") ||
      (cause instanceof TileDecodeError && cause.stage === "profile");
    setResourceState({
      status: unsupported ? "unsupported" : "error",
      errorName: error instanceof Error ? error.name : "Error",
      message: lastError,
    });
    safeCall(() => config.onError?.(error));
    context.onWorkChange?.();
  };

  const adapter: TileContentAdapter = (
    services.createContentAdapter ?? createMeshContentAdapter
  )({
    renderer: context.renderer as {
      addActor(actor: unknown): void;
      removeActor(actor: unknown): void;
    },
    scheduleRender: context.scheduleRender,
    submissions: context.submissions,
    visible: active,
    onError(error) {
      report(error);
      const generation = configGeneration;
      queueMicrotask(() => {
        if (active && generation === configGeneration) refreshSelection();
      });
    },
    memory,
    memoryPool: context.memory,
    onPresentationChange,
    onAdmissionRefusal(ids, error) {
      const affected = ids.map((id) => ({ id, tile: demand.get(id) }));
      queueMicrotask(() => {
        if (disposed || !active) return;
        const cancelled = affected.flatMap(({ id, tile }) =>
          tile && demand.get(id) === tile && adapter.tileState(id) === "absent"
            ? [id]
            : [],
        );
        if (!cancelled.length) return;
        policyRefused(error);
        for (const id of cancelled) rememberRefusal([id], null, error);
        refreshSelection();
      });
    },
    onPolicyRefusal(error) {
      if (error instanceof ResourcePolicyError)
        rendererRefusal = {
          error,
          availability: context.memory.availability(),
        };
      policyRefused(error);
    },
  });
  const pickSet = createMeshPickSet();

  const verticalExaggeration = (): readonly number[] =>
    createVerticalExaggerationTransform(
      config.verticalExaggeration,
      config.verticalPivotZ,
    );

  const exaggeratedEcefToScene = (): readonly number[] =>
    multiply(verticalExaggeration(), config.tilesetToScene);

  const traversalModelMatrix = (): readonly number[] =>
    modelMatrix
      ? multiply(modelMatrix, exaggeratedEcefToScene())
      : exaggeratedEcefToScene();

  /**
   * Anchor placement composed after each tile's scene-local origin. Decoded
   * vertices stay in unexaggerated scene coordinates, so exaggeration is a matrix the
   * renderer applies rather than a property of the downloaded bytes.
   */
  const placementMatrix = (): readonly number[] =>
    modelMatrix
      ? multiply(modelMatrix, verticalExaggeration())
      : verticalExaggeration();

  let presentedScene: Mat16 | null = null;
  const syncPickPresentation = (): void => {
    const drawn = adapter.presentationMatrix();
    const requested = placementMatrix();
    if (Array.from(drawn).every((value, index) => value === requested[index]))
      presentedScene = modelMatrix === null ? null : Array.from(modelMatrix);
    pickSet.setPlacement({ drawn, scene: presentedScene });
    pickSet.replaceDrawn(adapter.pickableMeshTiles());
  };
  const applyPlacement = (): void => {
    adapter.setBaseMatrix(placementMatrix());
    syncPickPresentation();
  };

  const traversalMaximumSse = (): number =>
    memoryConstrained ? Number.MAX_VALUE : config.maximumScreenSpaceErrorPx;

  const traversalQualityFraction = (): number =>
    memoryConstrained ? 1 : allocation.qualityFraction;

  const setSubtreeSnapshot = (snapshot: ContentQueueSnapshot | null): void => {
    subtreeSnapshot = snapshot;
    subtreeHierarchy =
      snapshot && subtreeQueue
        ? {
            revision: snapshot.revision,
            configGeneration: snapshot.configGeneration,
            entries: snapshot.entries,
            subtreeById: subtreeQueue.contents(),
          }
        : null;
  };

  const select = (
    maximumScreenSpaceErrorPx: number,
    qualityFraction: number,
    honourBlockedGroups = true,
  ): TilesetTraversalResult => {
    const presented = adapter.presentedTileIds?.();
    const published = presented ? new Set(presented) : undefined;
    const result = traverseTileset({
      root: source!.root,
      camera: camera!,
      maximumScreenSpaceErrorPx,
      qualityFraction,
      modelMatrix: traversalModelMatrix(),
      geometricErrorScale: config.geometricErrorScale,
      readiness,
      ...(published ? { presented: (id) => published.has(id) } : {}),
      unavailable: (id) => unavailableRepresentations.has(id),
      ...(honourBlockedGroups && blockedGroupAncestors.size > 0
        ? { refinable: (id: string) => !blockedGroupAncestors.has(id) }
        : {}),
      subtrees: subtreeHierarchy,
    });
    // This hierarchy is derived from the bounded subtree snapshot. Replacing
    // the map keeps camera exploration from retaining every historical tile.
    materializedTileById = new Map(result.tileById);
    return result;
  };

  /**
   * Ids grouped under every ancestor prefix, so "the subtree under x" is one
   * lookup instead of a prefix scan of the whole list.
   */
  const indexByAncestor = (ids: readonly string[]): Map<string, string[]> => {
    const index = new Map<string, string[]>();
    for (const id of ids) {
      for (const ancestor of ancestorIds(id)) {
        const bucket = index.get(ancestor);
        if (bucket) bucket.push(id);
        else index.set(ancestor, [id]);
      }
    }
    return index;
  };

  // The desired set changes only with the selection that produced it, and the
  // admitted set only when the adapter says so, so each index is rebuilt once
  // per change rather than once per admission. Both are read many times per
  // admission and once per blocked tile across a whole selection refresh, so
  // scanning instead would cost O(tiles^2).
  let desiredIndex = new Map<string, string[]>();
  let desiredIndexOf: TilesetTraversalResult | null = null;
  let submittedIndex = new Map<string, string[]>();
  let submittedIndexRevision: number | null = null;

  const desiredDescendants = (id: string): readonly string[] => {
    if (!traversal) return [];
    if (desiredIndexOf !== traversal) {
      desiredIndex = indexByAncestor(traversal.desiredTileIds);
      desiredIndexOf = traversal;
    }
    return desiredIndex.get(id) ?? [];
  };

  /** Submitted descendants a newly admitted tile replaces under REPLACE. */
  const replacedDescendants = (id: string): string[] => {
    const revision = adapter.submissionRevision();
    if (submittedIndexRevision !== revision) {
      submittedIndex = indexByAncestor(adapter.submittedTileIds());
      submittedIndexRevision = revision;
    }
    return [...(submittedIndex.get(id) ?? [])];
  };

  /** Submitted ancestors of a tile, deepest first. */
  const submittedAncestors = (id: string): string[] =>
    ancestorIds(id).filter(
      (candidate) => adapter.tileState(candidate) === "submitted",
    );

  const submittedAncestorReplacement = (id: string): string | null => {
    for (const ancestor of submittedAncestors(id)) {
      const desired = desiredDescendants(ancestor);
      if (
        desired.length > 0 &&
        desired.every(
          (candidate) =>
            candidate === id || adapter.tileState(candidate) === "submitted",
        )
      ) {
        return ancestor;
      }
    }
    return null;
  };

  const replacementsForAdmission = (id: string): string[] => {
    const replacements = replacedDescendants(id);
    const ancestor = submittedAncestorReplacement(id);
    return ancestor === null ? replacements : [...replacements, ancestor];
  };

  const waitingForSiblingBeforeReplacement = (id: string): boolean =>
    submittedAncestors(id).some((ancestor) =>
      desiredDescendants(ancestor).some((candidate) => {
        if (candidate === id) return false;
        const state = adapter.tileState(candidate);
        // A submitted sibling cannot free any more space by arriving later.
        // Counting it as outstanding strands the last unaffordable child.
        if (state === "submitted") return false;
        // Nor is a sibling that was itself refused still coming: two refused
        // siblings counting each other would wait for each other forever.
        if (demand.get(candidate)?.blocked) return false;
        return (
          state === "queued" ||
          queue?.state(candidate)?.status === "ready" ||
          (demand.has(candidate) && readiness(candidate) !== "failed")
        );
      }),
    );

  const coveredSoonByDesiredDescendants = (id: string): boolean => {
    const descendants = desiredDescendants(id);
    return (
      descendants.length > 0 &&
      descendants.every((candidate) => {
        const state = adapter.tileState(candidate);
        return (
          state === "queued" ||
          state === "submitted" ||
          queue?.state(candidate)?.status === "ready"
        );
      })
    );
  };

  const rememberRefusal = (
    ids: readonly string[],
    replacement: string | null,
    refusal: TileSubmissionRefusal,
  ): void => {
    const blocked = {
      refusal,
      replacement,
      group: ids.length > 1,
      availability: context.memory.availability(),
    };
    for (const id of ids) {
      const tile = demand.get(id);
      if (tile) tile.blocked = blocked;
    }
  };

  const recordSubmission = (
    ids: readonly string[],
    replacement: string,
    outcome: TileSubmissionOutcome,
  ): void => {
    if (outcome === "queued") {
      for (const id of ids) {
        const tile = demand.get(id);
        if (tile) tile.blocked = undefined;
      }
      return;
    }
    if (outcome === "failed") return;
    const { refusal } = outcome;
    rememberRefusal(ids, replacement, refusal);
    policyRefused(refusal);
  };

  const trySubmitDesiredGroup = (
    id: string,
  ): TileSubmissionOutcome | undefined => {
    if (!traversal || !queue) return undefined;
    const ancestor = submittedAncestors(id)[0];
    if (!ancestor) return undefined;
    // Already submitted siblings keep their realization and count toward
    // complete replacement coverage.
    const desired = desiredDescendants(ancestor).filter(
      (candidate) => adapter.tileState(candidate) !== "submitted",
    );
    if (desired.length < 2) return undefined;
    const replacement = [ancestor, ...desired].join("\n");
    if (
      desired.some((candidate) => {
        const blocked = demand.get(candidate)?.blocked;
        return (
          blocked?.replacement === null || blocked?.replacement === replacement
        );
      })
    )
      return undefined;
    const entries = [];
    for (const candidate of desired) {
      if (adapter.tileState(candidate) !== "absent") return undefined;
      const content = queue.get(candidate);
      if (!content) return undefined;
      entries.push({
        id: candidate,
        content,
        onSubmitted: () => onAdmitted(candidate),
      });
    }
    const outcome = adapter.submitTileGroup(entries, [ancestor]);
    recordSubmission(desired, replacement, outcome);
    if (
      outcome !== "queued" &&
      outcome !== "failed" &&
      !(
        outcome.refusal instanceof ResourcePolicyError &&
        outcome.refusal.retryable
      )
    ) {
      const oversized = entries.filter(
        ({ content }) =>
          adapter.estimateTileBytes(content) > allocation.memoryBudgetBytes,
      );
      if (oversized.length) {
        for (const { id } of oversized) omitRepresentation(id);
      } else blockGroupUnder(id);
    }
    return outcome;
  };

  const blockGroupUnder = (id: string): void => {
    const ancestor = ancestorIds(id).find(
      (candidate) =>
        !unavailableRepresentations.has(candidate) &&
        materializedTileById.get(candidate)?.contentUrl !== undefined,
    );
    if (ancestor === undefined || blockedGroupAncestors.has(ancestor)) return;
    blockedGroupAncestors.add(ancestor);
    constrainedDesiredSignature = unconstrainedDesiredSignature();
    refreshSelection();
  };

  const unconstrainedDesiredSignature = (): string | null => {
    if (!source || !camera) return null;
    selectionPasses += 1;
    // Deliberately blind to both backoffs: this is the frontier the camera
    // would ask for if nothing were blocked, and it is the only thing that can
    // tell a genuinely new view from the one that was already refused.
    return select(
      config.maximumScreenSpaceErrorPx,
      allocation.qualityFraction,
      false,
    ).desiredTileIds.join("\n");
  };

  const clearBudgetConstraint = (): void => {
    memoryConstrained = false;
    blockedGroupAncestors.clear();
    constrainedDesiredSignature = null;
    for (const tile of demand.values()) tile.blocked = undefined;
    rendererRefusal = undefined;
    adapter.clearRequiredBytes();
    if (resourceState.status === "blocked")
      setResourceState({ status: "ready" });
  };

  const clearAdmissionState = (): void => {
    demand.clear();
    unavailableRepresentations.clear();
  };

  const retryIfDesiredSelectionChanged = (): void => {
    if (!memoryConstrained && blockedGroupAncestors.size === 0) return;
    if (constrainedDesiredSignature === unconstrainedDesiredSignature()) return;
    clearBudgetConstraint();
  };

  const readiness = (id: string) => {
    const state = adapter.tileState(id);
    if (state === "submitted") return "submitted" as const;
    if (state === "failed" || demand.get(id)?.blocked) return "failed" as const;
    if (unavailableRepresentations.has(id)) return "failed" as const;
    if (state === "queued") return "loading" as const;
    const entry = queue?.state(id);
    if (entry?.status === "ready") return "decoded" as const;
    if (entry?.status === "failed" || entry?.status === "blocked")
      return "failed" as const;
    if (entry) return "loading" as const;
    return "unloaded" as const;
  };

  const omitRepresentation = (id: string): void => {
    if (unavailableRepresentations.has(id)) return;
    unavailableRepresentations.add(id);
    blockedGroupAncestors.delete(id);
    refreshSelection();
  };

  const updateDrawSet = (selectionComplete = false): void => {
    // Cached-content callbacks can run synchronously inside setSelection.
    // Publish coverage once that batch has finished, not once per callback.
    if (!drawSetDirty || (refreshing && !selectionComplete)) return;
    drawSetDirty = false;
    if (!source || !camera || !active || disposed) {
      adapter.setDrawnTiles([]);
      pickSet.replaceDrawn([]);
      return;
    }
    traversal = select(traversalMaximumSse(), traversalQualityFraction());
    selectionPasses += 1;
    adapter.setDrawnTiles(traversal.drawnTileIds);
    pickSet.replaceDrawn(adapter.pickableMeshTiles());
  };

  const refreshSelection = (resolvedId?: string): void => {
    if (disposed) return;
    drawSetDirty = true;
    if (refreshing) {
      refreshAgain = true;
      return;
    }
    refreshing = true;
    try {
      if (!active || !source || !camera || !queue) {
        queue?.setSelection([]);
        subtreeQueue?.setSelection([]);
        for (const id of demand.keys()) adapter.cancelTile(id);
        demand.clear();
        updateDrawSet(true);
        return;
      }
      selectionPasses += 1;
      const next = select(traversalMaximumSse(), traversalQualityFraction());
      const previousDesired = traversal?.desiredTileIds;
      if (
        previousDesired?.length !== next.desiredTileIds.length ||
        next.desiredTileIds.some((id, index) => id !== previousDesired[index])
      )
        for (const tile of demand.values()) tile.blocked = undefined;
      traversal = next;
      subtreeQueue?.setSelection(next.neededSubtreeRequests);
      const nextContentRequests = contentRequests(
        next.tileById,
        next.requestedTileIds,
      );
      const nextRequested = new Set(
        nextContentRequests.map((request) => request.id),
      );
      for (const id of demand.keys()) {
        if (nextRequested.has(id)) continue;
        adapter.cancelTile(id);
        if (!next.drawnTileIds.includes(id)) adapter.retireTile(id);
        demand.delete(id);
      }
      for (const id of nextRequested) {
        if (!demand.has(id)) demand.set(id, {});
        adapter.restoreTile(id);
      }
      queue.setSelection(
        nextContentRequests.filter(({ id }) => {
          const state = adapter.tileState(id);
          return (
            state !== "submitted" &&
            state !== "failed" &&
            !unavailableRepresentations.has(id)
          );
        }),
      );
      const admissionOrder =
        resolvedId && nextRequested.has(resolvedId)
          ? [
              resolvedId,
              ...[...nextRequested].filter((id) => id !== resolvedId),
            ]
          : nextRequested;
      for (const id of admissionOrder) {
        if (adapter.tileState(id) !== "absent") continue;
        const content = queue.get(id);
        if (!content || unavailableRepresentations.has(id)) continue;

        const groupOutcome = trySubmitDesiredGroup(id);
        if (groupOutcome && groupOutcome !== "failed") {
          if (refreshAgain) break;
          continue;
        }

        let previous = demand.get(id)?.blocked;
        if (!previous?.group && previous?.replacement !== null) {
          const replacements = replacementsForAdmission(id);
          const replacement = replacements.join(",");
          if (previous?.replacement !== replacement) {
            const outcome = adapter.submitTile(
              id,
              content,
              () => onAdmitted(id),
              replacements,
            );
            recordSubmission([id], replacement, outcome);
            if (outcome === "queued" || outcome === "failed") continue;
            previous = demand.get(id)?.blocked;
          }
        }

        if (
          !previous ||
          (previous.refusal instanceof ResourcePolicyError &&
            previous.refusal.retryable)
        )
          continue;
        if (
          coveredSoonByDesiredDescendants(id) ||
          waitingForSiblingBeforeReplacement(id)
        )
          continue;
        if (
          id === source.root.id ||
          adapter.estimateTileBytes(content) > allocation.memoryBudgetBytes
        )
          omitRepresentation(id);
        else blockGroupUnder(id);
        if (refreshAgain) break;
      }
      updateDrawSet(true);
      noteResourceRecovery();
    } finally {
      refreshing = false;
      context.onWorkChange?.();
      if (refreshAgain) {
        refreshAgain = false;
        refreshSelection();
      }
    }
  };

  const onAdmitted = (id: string): void => {
    const tile = demand.get(id);
    if (disposed || !active || !tile) {
      adapter.retireTile(id);
      return;
    }
    tile.blocked = undefined;
    // Admission changes REPLACE coverage. Re-run selection so an ancestor is
    // removed from both the fetch obligation and renderer only after every
    // required descendant has crossed the shared admission boundary.
    refreshSelection();
    context.scheduleRender();
    context.onWorkChange?.();
  };

  const createQueue = (): void => {
    queue?.dispose();
    queue = createContentQueue({
      revision: config.revision,
      configGeneration,
      maxConcurrency: config.concurrency,
      maxDecodedBytes: config.cacheBytes,
      memory,
      decodedBuffers: buildTransferList,
      fetch: config.fetchContent,
      maxAttempts: config.maxAttempts,
      decodeReady: () => services.workers.hasCapacity?.() ?? true,
      priority: 1,
      decode: async (bytes, request, decodeContext) => {
        const tile = materializedTileById.get(request.id);
        if (!tile) throw new Error(`unknown 3D tile ${request.id}`);
        const job = services.workers.decode(
          {
            content: bytes,
            contentUrl: workerUrl(request.url),
            dependencyRootUrl: workerUrl(`${config.endpoint}/`),
            revision: decodeContext.revision,
            accumulatedTransform: [...tile.worldTransform] as any,
            tilesetToScene: [...config.tilesetToScene] as any,
            textureCapabilities: services.textureCapabilities,
            wasm: config.wasm,
          },
          { memory: decodeContext.work },
        );
        const abort = () => job.cancel();
        decodeContext.signal.addEventListener("abort", abort, { once: true });
        try {
          return await job.promise;
        } finally {
          decodeContext.signal.removeEventListener("abort", abort);
        }
      },
      decodedByteLength: decodedBytes,
      onContent: ({ id }) => {
        workProgressSerial += 1;
        if (!disposed && active && !refreshing) refreshSelection(id);
      },
      onError: (_request, error) => {
        workProgressSerial += 1;
        report(error);
      },
      onPolicyRefusal: (request, error) => {
        workProgressSerial += 1;
        policyRefused(error);
        if (!error.retryable) omitRepresentation(request.id);
      },
      onStateChange: (snapshot) => {
        workProgressSerial += 1;
        queueSnapshot = snapshot;
        context.onWorkChange?.();
      },
    });
  };

  const createHierarchyQueue = (tileset: TilesetSource): void => {
    subtreeQueue?.dispose();
    subtreeQueue = null;
    setSubtreeSnapshot(null);
    const implicit = tileset.root.implicitTiling;
    if (!implicit) return;
    subtreeQueue = createContentQueue<ParsedSubtree>({
      revision: config.revision,
      configGeneration,
      maxConcurrency: config.concurrency,
      maxDecodedBytes: DEFAULT_TILES3D_SUBTREE_CACHE_BYTES,
      memory,
      fetch: config.fetchSubtree,
      maxAttempts: config.maxAttempts,
      decode: (bytes) =>
        parseSubtree(bytes, implicit.subtreeLevels, implicit.metadataSchema),
      decodedByteLength: (subtree) => subtree.byteLength,
      onPolicyRefusal: (_request, error) => {
        workProgressSerial += 1;
        policyRefused(error);
        if (error.retryable) return;
        setSubtreeSnapshot(subtreeQueue?.snapshot() ?? null);
        refreshSelection();
      },
      onContent: () => {
        workProgressSerial += 1;
        if (disposed || !active) return;
        setSubtreeSnapshot(subtreeQueue?.snapshot() ?? null);
        refreshSelection();
        context.scheduleRender();
      },
      onError: (_request, error) => {
        workProgressSerial += 1;
        report(error);
        refreshSelection();
      },
      onStateChange: (snapshot) => {
        workProgressSerial += 1;
        setSubtreeSnapshot(snapshot);
        context.onWorkChange?.();
      },
    });
    setSubtreeSnapshot(subtreeQueue.snapshot());
  };

  const beginLoad = (): void => {
    if (disposed || !active) return;
    const generation = ++loadGeneration;
    loadController?.abort();
    const controller = new AbortController();
    loadController = controller;
    sourceState = "loading";
    sourceRefusal = null;
    sourceMemory?.release();
    sourceMemory = undefined;
    source = null;
    materializedTileById.clear();
    traversal = null;
    queueSnapshot = null;
    context.onWorkChange?.();
    void memory.admission
      .run<{ loaded: TilesetSource; lease: MemoryLease }>({
        group: memory,
        availability: memory.availability,
        canRetry: memory.canRetry,
        signal: controller.signal,
        onDiscarded: ({ lease }) => lease.release(),
        stage: {
          lane: "network",
          async start(signal) {
            const work = memory.work(signal);
            try {
              const loaded = await loadTileset({
                endpoint: config.endpoint,
                fetch: config.fetchTileset,
                signal,
                memory: work,
              });
              return {
                value: {
                  loaded,
                  lease: work.commit([], loaded.tileById.size * 1024),
                },
              };
            } finally {
              work.release();
            }
          },
        },
      })
      .then(
        ({ loaded, lease }) => {
          if (
            disposed ||
            generation !== loadGeneration ||
            controller.signal.aborted
          ) {
            lease.release();
            return;
          }
          sourceMemory = lease;
          source = loaded;
          materializedTileById = new Map(loaded.tileById);
          workProgressSerial += 1;
          sourceState = "ready";
          createQueue();
          createHierarchyQueue(loaded);
          refreshSelection();
          context.scheduleRender();
        },
        (error) => {
          if (
            disposed ||
            generation !== loadGeneration ||
            controller.signal.aborted
          )
            return;
          sourceState = "failed";
          workProgressSerial += 1;
          sourceRefusal = resourcePolicyError(error);
          if (sourceRefusal) policyRefused(sourceRefusal);
          else report(error);
        },
      );
  };

  applyPlacement();
  beginLoad();

  return {
    setCamera(view) {
      if (disposed) return;
      // Hosts restate the camera unconditionally — every pre-paint pass and
      // twice per pick — so acting on an unchanged view costs a full tileset
      // traversal for nothing, and does it inside the span the host times.
      if (camera !== null && sameCameraView(camera, view)) return;
      camera = copyCameraView(view);
      adapter.setCamera(camera);
      retryIfDesiredSelectionChanged();
      refreshSelection();
    },

    setModelMatrix(matrix: Mat16 | null) {
      if (disposed) return;
      if (sameMatrix(modelMatrix, matrix ?? null)) return;
      try {
        modelMatrix =
          matrix === null ? null : validateAffineMatrix(matrix, "model matrix");
      } catch (error) {
        report(error);
        return;
      }
      applyPlacement();
      drawSetDirty = true;
      retryIfDesiredSelectionChanged();
      refreshSelection();
    },

    setDevicePixelRatio(next) {
      if (!disposed && Number.isFinite(next) && next > 0)
        devicePixelRatio = next;
    },

    setActive(next) {
      if (disposed || active === next) return;
      active = next;
      adapter.setVisible(next);
      drawSetDirty = true;
      if (!next) {
        loadController?.abort();
        loadGeneration += 1;
        queue?.setSelection([]);
        subtreeQueue?.setSelection([]);
        for (const id of demand.keys()) adapter.cancelTile(id);
        adapter.clearTiles();
        pickSet.replaceDrawn([]);
        clearAdmissionState();
        clearBudgetConstraint();
        updateDrawSet();
      } else if (!source) beginLoad();
      else {
        retryIfDesiredSelectionChanged();
        refreshSelection();
      }
      context.onWorkChange?.();
    },

    setConfig(kindConfig) {
      if (disposed) return;
      let next: ResolvedTiles3dConfig;
      try {
        next = validateTiles3dMemberConfig(kindConfig as Tiles3dMemberConfig);
      } catch (error) {
        report(error);
        return;
      }
      const sourceChanged =
        next.endpoint !== config.endpoint || next.revision !== config.revision;
      const decodeChanged =
        next.tilesetToScene.some(
          (value, index) => value !== config.tilesetToScene[index],
        ) || !sameDecodeWasm(next.wasm, config.wasm);
      const placementChanged =
        next.verticalExaggeration !== config.verticalExaggeration ||
        next.verticalPivotZ !== config.verticalPivotZ ||
        next.geometricErrorScale !== config.geometricErrorScale;
      config = next;
      if (placementChanged) applyPlacement();
      if (sourceChanged || decodeChanged) {
        traverseTileset = createTilesetTraversal();
        configGeneration += 1;
        queue?.dispose();
        queue = null;
        queueSnapshot = null;
        subtreeQueue?.dispose();
        subtreeQueue = null;
        setSubtreeSnapshot(null);
        adapter.clearTiles();
        pickSet.replaceDrawn([]);
        source = null;
        materializedTileById.clear();
        sourceState = "idle";
        traversal = null;
        clearAdmissionState();
        clearBudgetConstraint();
        if (active) beginLoad();
      } else {
        queue?.configure({
          maxConcurrency: config.concurrency,
          maxDecodedBytes: config.cacheBytes,
        });
        subtreeQueue?.configure({ maxConcurrency: config.concurrency });
        retryIfDesiredSelectionChanged();
        refreshSelection();
      }
    },

    beginInteraction() {
      if (!disposed) interactionDepth += 1;
    },
    endInteraction() {
      if (!disposed && interactionDepth > 0) interactionDepth -= 1;
    },
    prepareFrame() {
      if (!disposed && active) {
        const revision = context.memory.policyRevision();
        const availability = context.memory.availability();
        if (revision !== lastPolicyRevision) {
          lastPolicyRevision = revision;
          unavailableRepresentations.clear();
          clearBudgetConstraint();
          if (sourceRefusal) beginLoad();
          queue?.retryBlocked();
          subtreeQueue?.retryBlocked();
          refreshSelection();
        } else if (availability !== lastAvailability) {
          let retry = retryRealizations("availability");
          if (
            rendererRefusal?.error.estimate.wake === "availability" &&
            rendererRefusal.availability !== availability
          ) {
            rendererRefusal = undefined;
            adapter.clearRequiredBytes();
            retry = true;
          }
          if (retry) refreshSelection();
        }
        lastAvailability = availability;
      }
      if (!disposed && drawSetDirty) updateDrawSet();
    },

    governorInputs(): GovernorInputs {
      const renderer = adapter.workState();
      const hasContent = !!traversal && traversal.requestedTileIds.length > 0;
      return {
        projectedImportance:
          active && hasContent
            ? importanceFromRootSseCssPx(
                traversal!.rootScreenSpaceErrorPx,
                config.maximumScreenSpaceErrorPx,
              )
            : CULLED,
        qualityDemand: active && hasContent ? 1 : 0,
        work: {
          operations:
            active &&
            (sourceState === "loading" ||
              !!queueSnapshot?.workPending ||
              !!subtreeSnapshot?.workPending ||
              renderer.pendingJobs > 0 ||
              (traversal?.requestedTileIds.some((id) => {
                const state = readiness(id);
                return state !== "submitted" && state !== "failed";
              }) ??
                false))
              ? Math.max(
                  1,
                  (queueSnapshot?.active ?? 0) +
                    (subtreeSnapshot?.active ?? 0) +
                    renderer.pendingJobs,
                )
              : 0,
          progressSerial: workProgressSerial + renderer.workRevision,
        },
        physicalTileOperations: active ? (queueSnapshot?.active ?? 0) : 0,
        physicalHierarchyOperations: active
          ? (sourceState === "loading" ? 1 : 0) + (subtreeSnapshot?.active ?? 0)
          : 0,
        residentBytes: renderer.residentBytes,
        memoryDemand: {
          minimumBytes: 0,
          desiredBytes: queueSnapshot?.workPending
            ? Number.POSITIVE_INFINITY
            : Math.max(renderer.residentBytes, adapter.requiredBytes()),
        },
      };
    },

    onStall(error) {
      if (!disposed)
        setResourceState({ status: "stalled", message: errorMessage(error) });
    },

    applyAllocation(next) {
      if (disposed) return;
      const previousMemoryBudgetBytes = allocation.memoryBudgetBytes;
      allocation = next;
      // Trim the reuse pool before judging the new share: pooled tiles are
      // held on speculation, and counting them would call a share that still
      // fits everything drawn a memory constraint.
      adapter.setResourceCeilingBytes(allocation.memoryBudgetBytes);
      if (allocation.memoryBudgetBytes > previousMemoryBudgetBytes) {
        unavailableRepresentations.clear();
        clearBudgetConstraint();
      } else if (
        adapter.workState().residentBytes > allocation.memoryBudgetBytes
      ) {
        memoryConstrained = true;
        constrainedDesiredSignature = unconstrainedDesiredSignature();
      } else {
        retryIfDesiredSelectionChanged();
      }
      refreshSelection();
    },

    pick: (view, cssX, cssY) =>
      active ? pickSet.pick(view, cssX, cssY) : null,
    occlusionDepth: (view, cssX, cssY) =>
      active ? pickSet.occlusionDepth(view, cssX, cssY) : null,

    stats(): Tiles3dMemberStats {
      const submissions = context.submissions.stats();
      const effectiveSse =
        traversal?.effectiveScreenSpaceErrorPx ??
        config.maximumScreenSpaceErrorPx /
          Math.max(allocation.qualityFraction, 0.05);
      return {
        kind: "tiles3d",
        active,
        disposed,
        sourceState,
        irreducibleBudget:
          unavailableRepresentations.size > 0 &&
          (traversal?.drawnTileIds.length ?? 0) === 0 &&
          !queueSnapshot?.workPending,
        revision: config.revision,
        capabilityKey: services.textureCapabilities.capabilityKey,
        devicePixelRatio,
        interactionDepth,
        configGeneration,
        verticalExaggeration: config.verticalExaggeration,
        verticalPivotZ: config.verticalPivotZ,
        geometricErrorScale: config.geometricErrorScale,
        allocation,
        maximumScreenSpaceErrorPx: config.maximumScreenSpaceErrorPx,
        effectiveScreenSpaceErrorPx: effectiveSse,
        sseMultiplier: effectiveSse / config.maximumScreenSpaceErrorPx,
        memoryConstrained,
        resourceState,
        blockedGroups: blockedGroupAncestors.size,
        selectedTiles: traversal?.desiredTileIds.length ?? 0,
        requestedTiles: traversal?.requestedTileIds.length ?? 0,
        selectionPasses,
        errorCount,
        lastError,
        queue: queueSnapshot,
        subtrees: subtreeSnapshot,
        decode: services.workers.stats?.() ?? null,
        renderer: adapter.stats(),
        submissions: {
          queuedJobs: submissions.queuedJobs,
          queuedBytes: submissions.queuedBytes,
          lastFrameAdmittedJobs: submissions.lastFrameAdmittedJobs,
          lastFrameAdmittedBytes: submissions.lastFrameAdmittedBytes,
          peakQueuedJobs: submissions.peakQueuedJobs,
          peakFrameAdmittedBytes: submissions.peakFrameAdmittedBytes,
        },
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      active = false;
      sourceState = "disposed";
      traverseTileset = createTilesetTraversal();
      loadGeneration += 1;
      loadController?.abort();
      queue?.dispose();
      queue = null;
      subtreeQueue?.dispose();
      subtreeQueue = null;
      setSubtreeSnapshot(null);
      clearAdmissionState();
      blockedGroupAncestors.clear();
      adapter.dispose();
      sourceMemory?.release();
      sourceMemory = undefined;
      memory.close();
      pickSet.dispose();
      source = null;
      materializedTileById.clear();
      camera = null;
    },
  };
};
