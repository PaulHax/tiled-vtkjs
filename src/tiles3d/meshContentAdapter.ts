/** Mesh realization with explicit rejection of unsupported primitive kinds. */

import { safeCall } from "../core/observers";
import type { DecodedMeshTileContent, DecodedTileContent } from "./decode";
import {
  TileContentRealizationError,
  type TileContentAdapterFactory,
} from "./contentAdapter";
import { createMeshAdapter } from "./meshAdapter";

const isMeshContent = (
  content: DecodedTileContent,
): content is DecodedMeshTileContent =>
  content.primitives.every((primitive) => primitive.kind === "mesh");

export const createMeshContentAdapter: TileContentAdapterFactory = (
  options,
) => {
  const adapter = createMeshAdapter(options);
  const rejected = new Set<string>();
  let disposed = false;
  const accept = (id: string, content: DecodedTileContent): boolean => {
    if (disposed || rejected.has(id)) return false;
    if (isMeshContent(content)) return true;
    if (!rejected.has(id)) {
      rejected.add(id);
      const primitive = content.primitives.find(
        (entry) => entry.kind !== "mesh",
      )!;
      safeCall(() =>
        options.onError?.(new TileContentRealizationError(id, primitive.kind)),
      );
    }
    return false;
  };
  return {
    ...adapter,
    setCamera() {},
    submitTile(id, content, onSubmitted, replacementIds) {
      if (!accept(id, content) || !isMeshContent(content)) return "failed";
      return adapter.submitTile(id, content, onSubmitted, replacementIds);
    },
    submitTileGroup(entries, replacementIds) {
      if (disposed) return "failed";
      // Validate the entire group before admitting any resources.
      let supported = true;
      for (const entry of entries) {
        if (!accept(entry.id, entry.content)) supported = false;
      }
      if (!supported) return "failed";
      return adapter.submitTileGroup(
        entries.map((entry) => {
          if (!isMeshContent(entry.content))
            throw new TileContentRealizationError(entry.id, "gaussian");
          return { ...entry, content: entry.content };
        }),
        replacementIds,
      );
    },
    tileState(id) {
      return rejected.has(id) ? "failed" : adapter.tileState(id);
    },
    restoreTile(id) {
      return rejected.has(id) ? "failed" : adapter.restoreTile(id);
    },
    retireTile(id) {
      const failed = rejected.delete(id);
      return adapter.retireTile(id) || failed;
    },
    clearTiles() {
      rejected.clear();
      adapter.clearTiles();
    },
    pickableMeshTiles: adapter.submittedTiles,
    dispose() {
      disposed = true;
      rejected.clear();
      adapter.dispose();
    },
  };
};
