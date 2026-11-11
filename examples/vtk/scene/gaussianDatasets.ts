export type GaussianDataset = {
  id: string;
  label: string;
  url: string;
  splats: number;
  tiles: number;
  levels: number;
  attribution: string;
  source: string;
  license: string;
};

export const DATASETS: readonly GaussianDataset[] = [
  {
    id: "jastrzebia-gora",
    label: "Jastrzębia Góra, Poland · 103.5M splats",
    url: "https://3d-tiles-rendererjs-3dgs-plugin.vercel.app/api/blob/JastrebiaGora/tileset.json",
    splats: 103518942,
    tiles: 8784,
    levels: 13,
    attribution: "Andrii Shramko · tiled version hosted by William Liu",
    source: "https://superspl.at/scene/221ee167",
    license: "https://creativecommons.org/licenses/by/4.0/",
  },
];
