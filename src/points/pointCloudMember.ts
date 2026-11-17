import { ADAPTIVE_QUALITY_DEFAULTS } from "../core/adaptiveBudget";
import { safeCall } from "../core/observers";
import { createLodController, type LodController } from "./controller";
import type { PointPresentation } from "./pointPresentation";
import { pointBudgets } from "./pointQuality";
import { createRendererAdapter, type RendererAdapter } from "./rendererAdapter";
import {
  occlusionFromPick,
  importanceFromRootSseCssPx,
  type Allocation,
  type GovernorInputs,
  type MemberPickResult,
  type OcclusionResult,
  type StreamedMember,
  type StreamedMemberContext,
} from "../core/streamedMember";
import type { TileSource } from "./tileSource";
import {
  ResourcePolicyError,
  type MemoryDomain,
} from "../core/memoryResources";

export const DEFAULT_FIXED_POINT_BUDGET = 2_000_000;
export const DEFAULT_MIN_POINT_BUDGET = ADAPTIVE_QUALITY_DEFAULTS.minBudget;

export type PointCloudAdaptiveOptions = {
  readonly minBudget?: number;
  readonly maxBudget?: number;
  readonly interactionTargetMs?: number;
  readonly stationaryTargetMs?: number;
};

export type PointCloudMemberConfig = {
  readonly source: TileSource;
  readonly pointCount?: number;
  readonly presentation: PointPresentation;
  readonly adaptive: boolean;
  readonly adaptiveOptions?: PointCloudAdaptiveOptions;
  readonly pointBudget?: number;
  readonly refinementCutoffPx?: number;
  readonly selectionDelayMs?: number;
  readonly onError?: (error: unknown) => void;
  readonly onResourceState?: (
    state: PointCloudMemberStats["resourceState"],
  ) => void;
};

export type PointCloudMemberStats = {
  readonly kind: "pointCloud";
  readonly adaptive: boolean;
  readonly minimumPointBudget: number;
  /** Explicit/default point ceiling, before camera demand and page memory. */
  readonly configuredPointCeiling: number | null;
  readonly fullPointCeiling: number;
  readonly allocation: Allocation;
  readonly controller: ReturnType<LodController["stats"]>;
  readonly renderer: ReturnType<RendererAdapter["stats"]>;
  readonly resourceState:
    | { readonly status: "ready" }
    | {
        readonly status: "blocked";
        readonly domain: MemoryDomain;
        readonly requestedBytes: number;
        readonly limitBytes: number;
        readonly retryable: boolean;
      }
    | {
        readonly status: "error";
        readonly errorName: string;
        readonly message: string;
      };
};

const pointCountOf = (config: PointCloudMemberConfig): number => {
  const stated = config.pointCount ?? config.source.metadata().pointCount;
  return Number.isFinite(stated) && stated >= 0 ? Math.floor(stated) : 0;
};

const adaptiveMinimum = (config: PointCloudMemberConfig): number =>
  Math.max(
    1,
    Math.floor(config.adaptiveOptions?.minBudget ?? DEFAULT_MIN_POINT_BUDGET),
  );

const configuredCeiling = (config: PointCloudMemberConfig): number => {
  const ceiling = config.adaptive
    ? config.adaptiveOptions?.maxBudget
    : (config.pointBudget ?? DEFAULT_FIXED_POINT_BUDGET);
  return ceiling === undefined || !Number.isFinite(ceiling)
    ? Number.POSITIVE_INFINITY
    : Math.max(0, Math.floor(ceiling));
};

export const createPointCloudMember = (
  context: StreamedMemberContext,
  initialConfig: PointCloudMemberConfig,
): StreamedMember => {
  let config = initialConfig;
  let active = true;
  let disposed = false;
  let stationaryQualityFraction = 1;
  let resourceState: PointCloudMemberStats["resourceState"] = {
    status: "ready",
  };
  const setResourceState = (
    next: PointCloudMemberStats["resourceState"],
  ): void => {
    resourceState = next;
    safeCall(() => config.onResourceState?.(next));
    context.onWorkChange?.();
  };
  let lastAvailability = context.memory.stats().availabilityRevision;
  let lastPolicy = context.memory.stats().revision;
  let retryOnAvailability = false;
  const onPolicyRefusal = (error: ResourcePolicyError): void => {
    retryOnAvailability = error.retryable;
    setResourceState({
      status: "blocked",
      domain: error.domain,
      requestedBytes: error.requestedBytes,
      limitBytes: error.limitBytes,
      retryable: error.retryable,
    });
    lastAvailability = context.memory.stats().availabilityRevision;
    queueMicrotask(() => {
      lastAvailability = context.memory.stats().availabilityRevision;
    });
    context.onWorkChange?.();
  };
  const memory = context.memory.createOwner({
    reclaim(domain) {
      if (domain === "cpu") controller.trimCache(0);
      adapter.trimReuse();
    },
  });
  let allocation: Allocation = {
    qualityFraction: 1,
    memoryBudgetBytes: 0,
    regime: "stationary",
  };

  const adapter = createRendererAdapter({
    renderer: context.renderer as {
      addActor(actor: unknown): void;
      removeActor(actor: unknown): void;
    },
    scheduleRender: context.scheduleRender,
    devicePixelRatio: context.devicePixelRatio,
    visible: active,
    memory,
  });
  const report = (error: unknown): void => {
    resourceState = {
      status: "error",
      errorName: error instanceof Error ? error.name : "Error",
      message: error instanceof Error ? error.message : String(error),
    };
    safeCall(() => config.onResourceState?.(resourceState));
    safeCall(() => config.onError?.(error));
    context.onWorkChange?.();
  };
  const controller = createLodController({
    source: config.source,
    onTiles(batch) {
      try {
        adapter.applyBatch(batch);
      } catch (error) {
        report(error);
      }
    },
    onDrawPlan: adapter.applyDrawPlan,
    scheduleRender: context.scheduleRender,
    onWorkChange: context.onWorkChange,
    onPointDiameterCssPx: adapter.setPointDiameterCssPx,
    active,
    pointBudget: config.adaptive
      ? adaptiveMinimum(config)
      : (config.pointBudget ?? DEFAULT_FIXED_POINT_BUDGET),
    memoryBudgetBytes: 0,
    memory,
    reserveRenderer: adapter.reserveTile,
    onPolicyRefusal,
    onError: report,
    presentation: config.presentation,
    refinementCutoffPx: config.refinementCutoffPx,
    selectionDelayMs: config.selectionDelayMs,
  });

  const fullCeiling = (memoryCeilingPoints: number): number =>
    Math.max(
      0,
      Math.min(
        pointCountOf(config),
        configuredCeiling(config),
        memoryCeilingPoints,
      ),
    );

  const applyPointAllocation = (next: Allocation): void => {
    allocation = next;
    adapter.setResourceCeilingBytes(next.memoryBudgetBytes);
    const demand = controller.governorInputs().memoryDemand;
    if (
      active &&
      demand.desiredBytes > 0 &&
      demand.minimumBytes > next.memoryBudgetBytes
    ) {
      onPolicyRefusal(
        new ResourcePolicyError(
          "renderer",
          demand.minimumBytes,
          next.memoryBudgetBytes,
        ),
      );
      if (adapter.workState().gpuResidentBytes > 0) return;
    } else if (
      resourceState.status === "blocked" &&
      resourceState.domain === "renderer"
    ) {
      setResourceState({ status: "ready" });
      controller.retryMemory();
    }
    controller.setMemoryBudgetBytes(next.memoryBudgetBytes);
    if (!config.adaptive) {
      controller.setPointBudget(
        config.pointBudget ?? DEFAULT_FIXED_POINT_BUDGET,
      );
      controller.setDensityFraction(1);
      return;
    }

    if (next.regime === "stationary") {
      stationaryQualityFraction = next.qualityFraction;
    }
    const controllerInputs = controller.governorInputs();
    const budgets = pointBudgets({
      allocation: next,
      stationaryFraction: stationaryQualityFraction,
      fullCeiling: fullCeiling(controllerInputs.memoryCeilingPoints),
      demandPoints: controllerInputs.demandPoints,
      minimum: adaptiveMinimum(config),
    });
    controller.setPointBudget(budgets.pointBudget);
    controller.setDensityFraction(budgets.densityFraction);
  };

  return {
    setCamera: controller.setCamera,

    setModelMatrix(matrix) {
      if (disposed) return;
      controller.setModelMatrix(matrix);
      adapter.setBaseMatrix(matrix);
    },

    setDevicePixelRatio(devicePixelRatio) {
      if (!disposed) adapter.setDevicePixelRatio(devicePixelRatio);
    },

    setActive(nextActive) {
      if (disposed || active === nextActive) return;
      active = nextActive;
      controller.setActive(nextActive);
      adapter.setVisible(nextActive);
    },

    setConfig(kindConfig) {
      if (disposed) return;
      const next = kindConfig as PointCloudMemberConfig;
      if (next.source !== config.source) controller.setSource(next.source);
      config = next;
      controller.setPresentation(config.presentation);
      controller.setRefinementCutoffPx(config.refinementCutoffPx ?? 1);
      applyPointAllocation(allocation);
    },

    beginInteraction: controller.beginInteraction,
    endInteraction: controller.endInteraction,
    prepareFrame() {
      const next = context.memory.stats().availabilityRevision;
      const policy = context.memory.stats().revision;
      const policyChanged = policy !== lastPolicy;
      lastPolicy = policy;
      if (
        !disposed &&
        active &&
        (policyChanged || (retryOnAvailability && next !== lastAvailability)) &&
        resourceState.status === "blocked"
      ) {
        lastAvailability = next;
        setResourceState({ status: "ready" });
        controller.retryMemory();
        lastAvailability = context.memory.stats().availabilityRevision;
      }
    },

    governorInputs(): GovernorInputs {
      const narrow = controller.governorInputs();
      const full = fullCeiling(narrow.memoryCeilingPoints);
      return {
        // The controller reports root SSE in CSS pixels; the governor compares
        // members, so it is normalized against this cloud's own cutoff here.
        projectedImportance: importanceFromRootSseCssPx(
          narrow.projectedImportance,
          config.refinementCutoffPx ?? 1,
        ),
        qualityDemand:
          narrow.projectedImportance > 0 && full > 0
            ? Math.min(1, narrow.demandPoints / full)
            : 0,
        work: {
          // Backoff remains one logical obligation even when physical counts
          // are temporarily zero.
          operations: narrow.workPending
            ? Math.max(
                1,
                narrow.physicalTileOperations +
                  narrow.physicalHierarchyOperations,
              )
            : 0,
          progressSerial: narrow.workRevision,
        },
        physicalTileOperations: narrow.physicalTileOperations,
        physicalHierarchyOperations: narrow.physicalHierarchyOperations,
        residentBytes: adapter.workState().gpuResidentBytes,
        memoryDemand: narrow.memoryDemand,
      };
    },

    applyAllocation(next) {
      if (!disposed) applyPointAllocation(next);
    },

    pick(view, cssX, cssY): MemberPickResult | null {
      return controller.pickPoint(view, cssX, cssY);
    },

    occlusionDepth(view, cssX, cssY): OcclusionResult | null {
      return occlusionFromPick(controller.pickPoint(view, cssX, cssY));
    },

    stats(): PointCloudMemberStats {
      return {
        kind: "pointCloud",
        adaptive: config.adaptive,
        minimumPointBudget: adaptiveMinimum(config),
        configuredPointCeiling: Number.isFinite(configuredCeiling(config))
          ? configuredCeiling(config)
          : null,
        fullPointCeiling: fullCeiling(
          controller.governorInputs().memoryCeilingPoints,
        ),
        allocation,
        controller: controller.stats(),
        renderer: adapter.stats(),
        resourceState,
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      controller.dispose();
      adapter.dispose();
      memory.close();
    },
  };
};
