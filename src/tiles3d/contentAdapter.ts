/** Content realization contract beneath tile selection and residency. */

import type { MemoryPool } from "../core/memoryPool";
import type { ResourcePolicyError } from "../core/memoryResources";
import type { Mat16 } from "../core/mat4";
import type { CameraView } from "../core/camera";
import type { DecodedTileContent, GaussianAppearanceProfile } from "./decode";
import type {
  MeshAdapter,
  MeshAdapterOptions,
  MeshAdapterStats,
} from "./meshAdapter";
import type { SubmittedMeshTile } from "./meshPicking";

export type TileCountLimit = {
  readonly domain: "count";
  readonly finalCount: number;
  readonly peakCount: number;
  readonly limitCount: number;
  readonly reason: "count-limit";
  readonly wake: "selection";
};

export type TileSubmissionRefusal = ResourcePolicyError | TileCountLimit;
export type TileSubmissionOutcome =
  | "queued"
  | "failed"
  | { readonly status: "blocked"; readonly refusal: TileSubmissionRefusal };

export type TileContentAdapterOptions = MeshAdapterOptions & {
  readonly memoryPool?: MemoryPool;
  readonly onPresentationChange?: () => void;
  readonly onWorkChange?: () => void;
  readonly onPolicyRefusal?: (refusal: TileSubmissionRefusal) => void;
  readonly onAdmissionRefusal?: (
    ids: readonly string[],
    error: TileSubmissionRefusal,
  ) => void;
};
export type TileContentAdapterStats = MeshAdapterStats;

/**
 * A submitted tile is ready to participate in a coherent presentation.
 * Implementations retain the last complete presentation while asynchronous
 * uploads or draw-order jobs are pending. Only triangles enter the pick set.
 * Decoded content and its buffers are read-only and can be shared across views.
 */
export type TileContentAdapter = Omit<
  MeshAdapter,
  | "resourcePlan"
  | "resourceLeases"
  | "estimateTileBytes"
  | "submitTile"
  | "submitTileGroup"
  | "submittedTiles"
  | "retireTile"
> & {
  /** Gaussian reconstruction color spaces this adapter can present. */
  readonly gaussianColorSpaces?: readonly GaussianAppearanceProfile["colorSpace"][];
  presentationMatrix(): Mat16;
  estimateTileBytes(content: DecodedTileContent): number;
  setCamera(view: CameraView): void;
  retireTile(id: string, reuse?: "revisit" | "refining"): boolean;
  submitTile(
    id: string,
    content: DecodedTileContent,
    onSubmitted?: () => void,
    replacementIds?: readonly string[],
  ): TileSubmissionOutcome;
  submitTileGroup(
    entries: readonly {
      readonly id: string;
      readonly content: DecodedTileContent;
      readonly onSubmitted?: () => void;
    }[],
    replacementIds: readonly string[],
  ): TileSubmissionOutcome;
  /** Published coverage when preparation and presentation are asynchronous. */
  presentedTileIds?(): readonly string[];
  pickableMeshTiles(): readonly SubmittedMeshTile[];
};

export type TileContentAdapterFactory = (
  options: TileContentAdapterOptions,
) => TileContentAdapter;

/** The selected renderer cannot realize this decoded content profile. */
export class TileRenderingProfileError extends Error {}

export class TileContentRealizationError extends TileRenderingProfileError {
  readonly tileId: string;
  readonly primitiveKind: string;

  constructor(tileId: string, primitiveKind: string) {
    super(
      `cannot render 3D tile ${tileId}: no realization for ${primitiveKind} primitives`,
    );
    this.name = "TileContentRealizationError";
    this.tileId = tileId;
    this.primitiveKind = primitiveKind;
  }
}
