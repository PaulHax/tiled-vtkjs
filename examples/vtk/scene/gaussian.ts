import { parseTileset } from "../../../src/tiles3d/tilesetSource";
import type { Tiles3dMemberConfig } from "../../../src/tiles3d/memberTypes";
import { DATASETS } from "./gaussianDatasets";
import { tilesetPlacement } from "./tilesetPlacement";

export const GAUSSIAN_CITY = DATASETS[0]!;

export const createGaussianSource = () => {
  const source = {
    label: GAUSSIAN_CITY.label,
    endpoint: "/gaussian-city",
    location: { kind: "tiles-url" as const, value: GAUSSIAN_CITY.url },
    attribution: GAUSSIAN_CITY,
  };
  return {
    ...source,
    async prepare(signal: AbortSignal) {
      const request = new URL("/gaussian-city/tileset.json", location.href);
      const response = await fetch(request, { signal });
      if (!response.ok)
        throw new Error(`Gaussian tileset: HTTP ${response.status}`);
      const document = (await response.json()) as unknown;
      const endpoint = new URL(".", request).href.replace(/\/$/, "");
      const parsed = parseTileset(document, endpoint);
      const placement = tilesetPlacement(parsed.root, true);
      const fetchTileset: Tiles3dMemberConfig["fetchTileset"] = async (
        _url,
        { signal },
      ) => {
        signal?.throwIfAborted();
        return {
          ok: true,
          status: 200,
          statusText: "OK",
          json: async () => document,
        };
      };
      return {
        ...source,
        endpoint,
        tilesetToScene: placement.tilesetToScene,
        view: {
          center: [...placement.center] as [number, number, number],
          distance: placement.radius * 2.8,
          bounds: placement.bounds,
        },
        fetchTileset,
      };
    },
  };
};
