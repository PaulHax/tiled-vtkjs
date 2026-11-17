import type { TilesetFetch } from "./tilesetSource";
import type { ContentQueueFetch } from "./contentQueue";
import type {
  DecodeWasmUrls,
  DecodeWorkerPoolHandle,
  TextureCapabilities,
} from "./decode";
import type { Allocation } from "../core/streamedMember";
import type { MemoryDomain } from "../core/memoryResources";

export type Tiles3dResourceState =
  | { readonly status: "ready" }
  | {
      readonly status: "blocked";
      readonly domain: MemoryDomain;
      readonly requestedBytes: number;
      readonly limitBytes: number;
      readonly reason: "configured-limit" | "transition-unavailable";
      readonly retryable: boolean;
    }
  | {
      readonly status: "error";
      readonly errorName: string;
      readonly message: string;
    }
  | { readonly status: "stalled"; readonly message: string };
import type {
  TileContentAdapterFactory,
  TileContentAdapterStats,
} from "./contentAdapter";

export const DEFAULT_MAXIMUM_SCREEN_SPACE_ERROR_PX = 16;
export const DEFAULT_TILES3D_CACHE_BYTES = 128 * 1024 * 1024;
export const DEFAULT_TILES3D_CONCURRENCY = 4;
export const DEFAULT_TILES3D_SUBTREE_CACHE_BYTES = 8 * 1024 * 1024;
export const DEFAULT_VERTICAL_EXAGGERATION = 1;
export const DEFAULT_VERTICAL_PIVOT_Z = 0;

export type ContentQueueStats = {
  readonly selected: number;
  readonly active: number;
  readonly queued: number;
  readonly retrying: number;
  readonly ready: number;
  readonly failed: number;
  readonly blocked: number;
  readonly cached: number;
  readonly decodedBytes: number;
  readonly workPending: boolean;
  readonly cacheHits: number;
  readonly cacheMisses: number;
  readonly cacheEvictions: number;
  readonly cacheRevisits: number;
  readonly entries: readonly {
    readonly id: string;
    readonly url: string;
    readonly status:
      | "queued"
      | "fetching"
      | "retrying"
      | "ready"
      | "failed"
      | "blocked";
    readonly attempt: number;
  }[];
};

export type GeometricErrorScale = "maximum" | "horizontal";

/** Format services shared by the host; members cancel their own jobs on disposal. */
export type Tiles3dMemberServices = {
  /** Caller-owned pool. Dispose it after disposing every member that uses it. */
  readonly workers: DecodeWorkerPoolHandle;
  readonly textureCapabilities: TextureCapabilities;
  /** View-owned realization, shared across members when the renderer needs global order. */
  readonly createContentAdapter?: TileContentAdapterFactory;
};

export type Tiles3dMemberConfig = {
  readonly endpoint: string;
  readonly revision: string;
  readonly tilesetToScene: readonly number[];
  /** Finite positive scale applied to the scene Z axis. */
  readonly verticalExaggeration?: number;
  /** Finite scene Z coordinate held fixed by exaggeration. */
  readonly verticalPivotZ?: number;
  /** Scale convention for tile geometric error; terrain error is horizontal. */
  readonly geometricErrorScale?: GeometricErrorScale;
  readonly maximumScreenSpaceErrorPx?: number;
  readonly wasm?: DecodeWasmUrls;
  readonly cacheBytes?: number;
  /** Simultaneous content and subtree fetches. */
  readonly concurrency?: number;
  readonly maxAttempts?: number;
  readonly fetchTileset?: TilesetFetch;
  readonly fetchContent?: ContentQueueFetch;
  readonly fetchSubtree?: ContentQueueFetch;
  readonly onError?: (error: unknown) => void;
  readonly onResourceState?: (state: Tiles3dResourceState) => void;
};

export type Tiles3dMemberStats = {
  readonly kind: "tiles3d";
  readonly active: boolean;
  readonly disposed: boolean;
  readonly sourceState: "idle" | "loading" | "ready" | "failed" | "disposed";
  /**
   * The root does not fit the member's byte allowance, so nothing is drawn
   * and no further tile work is outstanding. Clears when the allowance grows.
   */
  readonly irreducibleBudget: boolean;
  readonly revision: string;
  readonly capabilityKey: string;
  readonly devicePixelRatio: number;
  readonly interactionDepth: number;
  /** Currency of placement-affecting source configuration. */
  readonly configGeneration: number;
  readonly verticalExaggeration: number;
  readonly verticalPivotZ: number;
  readonly geometricErrorScale: GeometricErrorScale;
  readonly allocation: Allocation;
  readonly maximumScreenSpaceErrorPx: number;
  readonly effectiveScreenSpaceErrorPx: number;
  readonly sseMultiplier: number;
  readonly memoryConstrained: boolean;
  readonly resourceState: Tiles3dResourceState;
  /** Submitted tiles whose replacement group did not fit the allowance. */
  readonly blockedGroups: number;
  readonly selectedTiles: number;
  readonly requestedTiles: number;
  /** Traversal passes, exposed so hosts can verify stationary frame idempotence. */
  readonly selectionPasses: number;
  readonly errorCount: number;
  readonly lastError: string | null;
  /** Tile content fetch/decode queue. */
  readonly queue: ContentQueueStats | null;
  /** Implicit-hierarchy subtree fetch/parse queue, null for explicit tilesets. */
  readonly subtrees: ContentQueueStats | null;
  readonly decode: ReturnType<
    NonNullable<import("./decode").DecodeWorkerPoolHandle["stats"]>
  > | null;
  readonly renderer: TileContentAdapterStats;
  readonly submissions: {
    readonly queuedJobs: number;
    readonly queuedBytes: number;
    readonly lastFrameAdmittedJobs: number;
    readonly lastFrameAdmittedBytes: number;
    readonly peakQueuedJobs: number;
    readonly peakFrameAdmittedBytes: number;
  };
};
