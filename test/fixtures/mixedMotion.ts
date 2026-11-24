import { readFile, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { gaussianTilesFixture } from "./gaussianTiles";

// Reposition the existing COPC fixture using its LAS quantization and COPC
// cube. Point records and hierarchy pages stay intact.
async function nearbyCloud(path: string): Promise<Buffer> {
  const bytes = await readFile(path);
  const headerSize = bytes.readUInt16LE(94);
  const userId = bytes.toString("ascii", headerSize + 2, headerSize + 6);
  const recordId = bytes.readUInt16LE(headerSize + 18);
  if (userId !== "copc" || recordId !== 1)
    throw new Error("The fixture must begin with the COPC info record");
  const info = headerSize + 54;
  const scale = 0.035;
  const target = [-2.6, 0, 0];
  for (const [axis, center] of target.entries()) {
    const sourceCenter = bytes.readDoubleLE(info + axis * 8);
    const shift = (offset: number) =>
      bytes.writeDoubleLE(
        center + (bytes.readDoubleLE(offset) - sourceCenter) * scale,
        offset,
      );
    const lasScale = 131 + axis * 8;
    bytes.writeDoubleLE(bytes.readDoubleLE(lasScale) * scale, lasScale);
    shift(155 + axis * 8); // LAS coordinate offset
    shift(179 + axis * 16); // LAS bounds, maximum then minimum
    shift(187 + axis * 16);
    bytes.writeDoubleLE(center, info + axis * 8);
  }
  for (const offset of [info + 24, info + 32]) // COPC half-size and spacing
    bytes.writeDoubleLE(bytes.readDoubleLE(offset) * scale, offset);
  return bytes;
}

export async function writeMixedMotionFixture(
  directory: string,
  cloud: string,
): Promise<void> {
  const splats = join(directory, "splats");
  const mesh = join(directory, "mesh");
  await Promise.all([mkdir(splats), mkdir(mesh)]);
  const fixture = gaussianTilesFixture();

  // glTF is Y-up. The triangle lies on the scene floor, beside the splats.
  const positions = Float32Array.of(1.3, 0, 0.5, 3.7, 0, 0.5, 2.5, 0, -0.5);
  const triangle = {
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }],
    buffers: [
      {
        byteLength: positions.byteLength,
        uri:
          "data:application/octet-stream;base64," +
          Buffer.from(positions.buffer).toString("base64"),
      },
    ],
    bufferViews: [{ buffer: 0, byteLength: positions.byteLength }],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: 3,
        type: "VEC3",
        min: [1.3, 0, -0.5],
        max: [3.7, 0, 0.5],
      },
    ],
    extensionsUsed: ["KHR_materials_unlit"],
    materials: [
      {
        doubleSided: true,
        pbrMetallicRoughness: { baseColorFactor: [0, 1, 1, 1] },
        extensions: { KHR_materials_unlit: {} },
      },
    ],
  };
  const tileset = {
    asset: { version: "1.1" },
    geometricError: 1,
    root: {
      boundingVolume: { box: [2.5, 0, 0, 1.3, 0, 0, 0, 1, 0, 0, 0, 1] },
      refine: "REPLACE",
      geometricError: 0,
      content: { uri: "triangle.gltf" },
    },
  };
  await Promise.all([
    writeFile(join(splats, "tileset.json"), JSON.stringify(fixture.document)),
    ...Object.entries(fixture.files).map(([name, bytes]) =>
      writeFile(join(splats, name), bytes),
    ),
    writeFile(join(mesh, "tileset.json"), JSON.stringify(tileset)),
    writeFile(join(mesh, "triangle.gltf"), JSON.stringify(triangle)),
    writeFile(join(directory, "points.copc.laz"), await nearbyCloud(cloud)),
  ]);
}
