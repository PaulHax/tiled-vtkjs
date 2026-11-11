/** Content realization contract beneath tile selection and residency. */

import type { MemoryPool } from "../core/memoryPool";
import type { ResourcePolicyError } from "../core/memoryResources";
import type { Mat16 } from "../core/mat4";
import type { CameraView } from "../core/camera";
import type { DecodedTileContent } from "./decode";
import type {
  MeshAdapter,
  MeshAdapterOptions,
  MeshAdapterStats,
  MeshSubmitOutcome,
} from "./meshAdapter";
import type { SubmittedMeshTile } from "./meshPicking";

export type TileContentAdapterOptions = MeshAdapterOptions & {
  /** Physical renderer shared by realizations behind per-layer mesh wrappers. */
  readonly realizationRenderer?: MeshAdapterOptions["renderer"];
  readonly memoryPool?: MemoryPool;
  readonly onPresentationChange?: () => void;
  readonly onAdmissionRefusal?: (
    ids: readonly string[],
    error: ResourcePolicyError,
  ) => void;
};
export type TileContentAdapterStats = MeshAdapterStats;

/**
 * A submitted tile is ready to participate in a coherent presentation.
 * Implementations retain the last complete presentation while asynchronous
 * uploads or draw-order jobs are pending. Only triangles enter the pick set.
 */
export type TileContentAdapter = Omit<
  MeshAdapter,
  | "resourcePlan"
  | "resourceLeases"
  | "estimateTileBytes"
  | "submitTile"
  | "submitTileGroup"
  | "submittedTiles"
> & {
  presentationMatrix(): Mat16;
  estimateTileBytes(content: DecodedTileContent): number;
  setCamera(view: CameraView): void;
  submitTile(
    id: string,
    content: DecodedTileContent,
    onSubmitted?: () => void,
    replacementIds?: readonly string[],
  ): MeshSubmitOutcome;
  submitTileGroup(
    entries: readonly {
      readonly id: string;
      readonly content: DecodedTileContent;
      readonly onSubmitted?: () => void;
    }[],
    replacementIds: readonly string[],
  ): MeshSubmitOutcome;
  pickableMeshTiles(): readonly SubmittedMeshTile[];
};

export type TileContentAdapterFactory = (
  options: TileContentAdapterOptions,
) => TileContentAdapter;

/** The selected renderer cannot realize this decoded content profile. */
export class TileContentRealizationError extends Error {
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
