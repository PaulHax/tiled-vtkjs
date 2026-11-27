import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const [source, output] = process.argv.slice(2);
if (!source || !output)
  throw new Error(
    "Usage: node prepare-city.mjs <published tileset.json> <output directory>",
  );

const data = await readFile(source);
const sha256 = createHash("sha256").update(data).digest("hex");
if (
  sha256 !== "510fa249994ee65aa69b408fc92aa5b6e1aca1079c45f97c03d47495f9fbdd65"
)
  throw new Error("Unexpected Jastrzębia Góra source manifest " + sha256);

const tileset = JSON.parse(data);
// The benchmark camera uses the tileset's local coordinates.
delete tileset.root.transform;

const camera = {
  target: [-525, 9, -245],
  poiA: [-1120, 72, -245],
  panOffset: [-258, -155, 0],
  direction: [0.1, -0.64, 0.77],
  farDistance: 6894.858154296875,
  nearDistance: 170,
  transferDistance: 600,
};
const viewers = ["tiled-vtkjs", "cesium", "three-tiles"];
const manifest = {
  assetRoot: "assets",
  assetSource: {
    directory: "city-full",
    url: "https://3d-tiles-rendererjs-3dgs-plugin.vercel.app/api/blob/JastrebiaGora/",
  },
  cases: [256, 24].flatMap((memoryMiB) =>
    viewers.map((viewer) => ({
      id: "city-full-" + viewer + "-" + memoryMiB,
      viewer,
      asset: "city-full/tileset.json",
      format: "gaussian-3d-tiles",
      workload: "full-city-two-poi",
      sse: 12,
      memoryMiB,
      camera,
    })),
  ),
};
const directory = resolve(output, "assets/city-full");
await mkdir(directory, { recursive: true });
for (const [file, json] of [
  [resolve(output, "city-full.json"), manifest],
  [resolve(directory, "tileset.json"), tileset],
]) {
  await writeFile(file, JSON.stringify(json, null, 2) + "\n");
}
