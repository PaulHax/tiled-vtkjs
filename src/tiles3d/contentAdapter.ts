/** Content realization contract beneath tile selection and residency. */

import type { CameraView } from "../core/camera";
import type { DecodedTileContent } from "./decode";
import type {
  MeshAdapter,
  MeshAdapterOptions,
  MeshAdapterStats,
  MeshSubmitOutcome,
} from "./meshAdapter";
import type { SubmittedMeshTile } from "./meshPicking";

export type TileContentAdapterOptions = MeshAdapterOptions;
export type TileContentAdapterStats = MeshAdapterStats;

/**
 * A submitted tile is ready to participate in a coherent presentation.
 * Implementations retain the last complete presentation while asynchronous
 * uploads or draw-order jobs are pending. Only triangles enter the pick set.
 */
export type TileContentAdapter = Omit<
  MeshAdapter,
  "submitTile" | "submitTileGroup" | "submittedTiles"
> & {
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
