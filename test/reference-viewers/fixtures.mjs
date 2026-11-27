import { writePointTiles } from "./prepare-point-tiles.mjs";
import { execFileSync } from "node:child_process";
import { copyFile, mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { Document, NodeIO } from "@gltf-transform/core";
import { KHRMaterialsUnlit } from "@gltf-transform/extensions";
import createSpzModule from "@adobe/spz";
import sharp from "sharp";
import { spzGlb } from "../fixtures/spzGlb.ts";

const saveJson = (path, value) =>
  writeFile(path, JSON.stringify(value, null, 2) + "\n");
const splatExtensions = [
  "KHR_gaussian_splatting",
  "KHR_gaussian_splatting_compression_spz_2",
];
const bounds = (x, width) => ({
  box: [x, 0, 0, width, 0, 0, 0, 2, 0, 0, 0, 2],
});

async function makeMesh(path, x, width, tint) {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const material = doc
    .createMaterial()
    .setBaseColorFactor(tint)
    .setDoubleSided(true);
  material.setExtension(
    "KHR_materials_unlit",
    doc.createExtension(KHRMaterialsUnlit).createUnlit(),
  );
  const positions = doc
    .createAccessor()
    .setType("VEC3")
    .setArray(
      Float32Array.of(
        x - width,
        0,
        -2,
        x + width,
        0,
        -2,
        x + width,
        0,
        2,
        x - width,
        0,
        2,
      ),
    )
    .setBuffer(buffer);
  const indices = doc
    .createAccessor()
    .setType("SCALAR")
    .setArray(Uint16Array.of(0, 2, 1, 0, 3, 2))
    .setBuffer(buffer);
  const primitive = doc
    .createPrimitive()
    .setAttribute("POSITION", positions)
    .setIndices(indices)
    .setMaterial(material);
  doc
    .createScene()
    .addChild(
      doc.createNode().setMesh(doc.createMesh().addPrimitive(primitive)),
    );
  await new NodeIO().registerExtensions([KHRMaterialsUnlit]).write(path, doc);
}

function splats(x, width, tint, density) {
  const points = [];
  for (let j = 0; j < density; j++)
    for (let i = 0; i < density; i++) {
      const u = (i + 0.5) / density,
        v = (j + 0.5) / density;
      points.push({
        position: [
          x + width * (2 * u - 1),
          0.18 * Math.sin(u * 10) * Math.sin(v * 10),
          2 * (2 * v - 1),
        ],
        scale: Math.log(Math.max(width, 2) / density),
        color: tint.map(
          (value) =>
            (value * (0.6 + 0.4 * Math.sin((u + v) * 8) ** 2) - 0.5) /
            0.2820947917738781,
        ),
      });
    }
  return points;
}

function writePly(path, points) {
  const properties = [
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
  ];
  const header = Buffer.from(
    [
      "ply",
      "format binary_little_endian 1.0",
      "element vertex " + points.length,
      ...properties.map((name) => "property float " + name),
      "end_header",
      "",
    ].join("\n"),
  );
  const values = points.flatMap(({ position, color, scale }) => [
    ...position,
    ...color,
    1.9,
    scale,
    scale,
    scale,
    1,
    0,
    0,
    0,
  ]);
  const bytes = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
  return writeFile(path, Buffer.concat([header, bytes]));
}

async function writeSplatTile(spz, path, points) {
  const numPoints = points.length;
  const compressedBytes = spz.saveSpzToBuffer(
    {
      numPoints,
      shDegree: 0,
      antialiased: false,
      extensions: [],
      positions: Float32Array.from(
        points.flatMap(({ position: [x, y, z] }) => [x, z, -y]),
      ),
      scales: Float32Array.from(
        points.flatMap(({ scale }) => [scale, scale, scale]),
      ),
      rotations: Float32Array.from(points.flatMap(() => [0, 0, 0, 1])),
      alphas: new Float32Array(numPoints).fill(1.9),
      colors: Float32Array.from(points.flatMap(({ color }) => color)),
      sh: new Float32Array(),
    },
    {
      version: 3,
      from: spz.CoordinateSystem.UNSPECIFIED,
      sh1Bits: 8,
      shRestBits: 5,
    },
  );
  const glb = await spzGlb({
    compressedBytes,
    mutate(json) {
      const attributes = json.meshes[0].primitives[0].attributes;
      for (const [name, type] of [
        ["ROTATION", "VEC4"],
        ["SCALE", "VEC3"],
        ["COLOR_0", "VEC4"],
      ]) {
        const semantic =
          name === "COLOR_0" ? name : "KHR_gaussian_splatting:" + name;
        attributes[semantic] = json.accessors.length;
        json.accessors.push({
          componentType: name === "COLOR_0" ? 5121 : 5126,
          count: numPoints,
          type,
          ...(name === "COLOR_0" ? { normalized: true } : {}),
        });
      }
      json.accessors[0].count = numPoints;
      json.accessors[0].min = [-2, -2, -2];
      json.accessors[0].max = [2, 2, 2];
      json.meshes[0].primitives[0].extensions.KHR_gaussian_splatting.colorSpace =
        "srgb_rec709_display";
    },
  });
  await writeFile(path, new Uint8Array(glb));
}

async function makeTiles(dir, gaussian) {
  await mkdir(dir, { recursive: true });
  const spz = gaussian ? await createSpzModule() : null;
  async function add(name, x, width, error, level) {
    const uri = name + ".glb";
    const tint = [0.18 + level * 0.22, 0.78, x < 0 ? 0.35 : 0.72, 1];
    if (gaussian)
      await writeSplatTile(
        spz,
        join(dir, uri),
        splats(x, width, tint.slice(0, 3), 10 + level * 6),
      );
    else await makeMesh(join(dir, uri), x, width, tint);
    const tile = {
      boundingVolume: bounds(x, width),
      geometricError: error,
      content: { uri },
    };
    if (level < 2)
      tile.children = await Promise.all(
        [-1, 1].map((side, i) =>
          add(
            name + "-" + i,
            x + (side * width) / 2,
            width / 2,
            level === 0 ? 0.028 : 0,
            level + 1,
          ),
        ),
      );
    return tile;
  }
  const root = { ...(await add("root", 0, 2, 0.075, 0)), refine: "REPLACE" };
  const extensions = gaussian
    ? {
        extensionsUsed: ["3DTILES_content_gltf"],
        extensionsRequired: ["3DTILES_content_gltf"],
        extensions: {
          "3DTILES_content_gltf": {
            extensionsUsed: splatExtensions,
            extensionsRequired: splatExtensions,
          },
        },
      }
    : {};
  await saveJson(join(dir, "tileset.json"), {
    asset: { version: "1.1" },
    geometricError: 16,
    root,
    ...extensions,
  });
}

async function makePotree(dir) {
  await mkdir(dir, { recursive: true });
  const count = 512;
  const octree = Buffer.alloc(3 * count * 18);
  const hierarchy = Buffer.alloc(3 * 22);
  for (const [node, x, y, z] of [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [2, 20, 20, 20],
  ]) {
    const nodeStart = node * count * 18;
    hierarchy.writeUInt8(node === 0 ? 0 : 1, node * 22);
    hierarchy.writeUInt8(node === 0 ? 129 : 0, node * 22 + 1);
    hierarchy.writeUInt32LE(count, node * 22 + 2);
    hierarchy.writeBigInt64LE(BigInt(nodeStart), node * 22 + 6);
    hierarchy.writeBigInt64LE(BigInt(count * 18), node * 22 + 14);
    const cell = node === 0 ? 40 : 20;
    for (let i = 0; i < count; i++) {
      const offset = nodeStart + i * 18;
      octree.writeInt32LE(
        Math.round((x + (cell * ((i % 8) + 0.5)) / 8) * 1000),
        offset,
      );
      octree.writeInt32LE(
        Math.round((y + (cell * ((Math.floor(i / 8) % 8) + 0.5)) / 8) * 1000),
        offset + 4,
      );
      octree.writeInt32LE(
        Math.round((z + (cell * (Math.floor(i / 64) + 0.5)) / 8) * 1000),
        offset + 8,
      );
      [13000, 55000, 65000].forEach((value, channel) =>
        octree.writeUInt16LE(value, offset + 12 + 2 * channel),
      );
    }
  }
  const attributes = [
    {
      name: "position",
      size: 12,
      numElements: 3,
      elementSize: 4,
      type: "int32",
      min: [0, 0, 0],
      max: [40, 40, 40],
    },
    {
      name: "rgb",
      size: 6,
      numElements: 3,
      elementSize: 2,
      type: "uint16",
      min: [0, 0, 0],
      max: [65535, 65535, 65535],
    },
  ];
  await Promise.all([
    writeFile(join(dir, "octree.bin"), octree),
    writeFile(join(dir, "hierarchy.bin"), hierarchy),
    saveJson(join(dir, "metadata.json"), {
      version: "2.0",
      name: "synthetic",
      points: count * 3,
      spacing: 5,
      scale: [0.001, 0.001, 0.001],
      offset: [0, 0, 0],
      boundingBox: { min: [0, 0, 0], max: [40, 40, 40] },
      encoding: "DEFAULT",
      hierarchy: { firstChunkSize: hierarchy.length, stepSize: 5, depth: 1 },
      attributes,
    }),
  ]);
}

async function makeRasterTiles(dir) {
  for (let z = 0; z <= 3; z++) {
    const axis = 2 ** z;
    for (let x = 0; x < axis; x++) {
      const column = join(dir, String(z), String(x));
      await mkdir(column, { recursive: true });
      await Promise.all(
        Array.from({ length: axis }, (_, y) =>
          sharp({
            create: {
              width: 256,
              height: 256,
              channels: 3,
              background: { r: 70 + z * 28, g: 110 + x * 12, b: 160 - y * 10 },
            },
          })
            .png()
            .toFile(join(column, y + ".png")),
        ),
      );
    }
  }
}

export async function createFixtures(directory, here) {
  await mkdir(directory, { recursive: true });
  const native = join(directory, "native-splats");
  await mkdir(native, { recursive: true });
  const tint = [0.2, 0.85, 0.55];
  await Promise.all([
    makeTiles(join(directory, "mesh"), false),
    makeTiles(join(directory, "gaussian"), true),
    makePotree(join(directory, "potree")),
    makeRasterTiles(join(directory, "raster")),
    copyFile(
      resolve(here, "../fixtures/multipage.copc.laz"),
      join(directory, "multipage.copc.laz"),
    ),
    writePly(join(native, "fine.ply"), splats(0, 2, tint, 40)),
    writePly(join(native, "coarse.ply"), splats(0, 2, tint, 12)),
  ]);
  const { origin: pointOrigin } = await writePointTiles(
    join(directory, "multipage.copc.laz"),
    join(directory, "point-tiles"),
  );
  const single = join(directory, "single");
  await mkdir(single, { recursive: true });
  await copyFile(join(directory, "mesh", "root.glb"), join(single, "root.glb"));
  await saveJson(join(single, "tileset.json"), {
    asset: { version: "1.1" },
    geometricError: 16,
    root: {
      boundingVolume: bounds(0, 2),
      geometricError: 0,
      refine: "REPLACE",
      content: { uri: "root.glb" },
    },
  });
  execFileSync(
    process.execPath,
    [
      resolve(here, "node_modules/@playcanvas/splat-transform/bin/cli.mjs"),
      "--overwrite",
      "-g",
      "cpu",
      join(native, "fine.ply"),
      "-l",
      "0",
      join(native, "coarse.ply"),
      "-l",
      "1",
      join(native, "lod-meta.json"),
    ],
    { stdio: "pipe" },
  );
  execFileSync(
    process.env.BENCHMARK_PYTHON ?? "python3",
    [resolve(here, "make-cog.py"), join(directory, "raster.tif")],
    { stdio: "pipe" },
  );
  const sceneCamera = { target: [0, 0, 0], farDistance: 24, nearDistance: 3.5 };
  const pointCamera = {
    target: [20, 20, 20],
    farDistance: 180,
    nearDistance: 24,
  };
  const mapCamera = {
    type: "map",
    center: [0, 0],
    farZoom: 0,
    nearZoom: 4,
    panDegrees: 8,
  };
  const tests = [
    [
      "deck-tiles-mesh-tiles",
      "deck-tiles",
      "3d-tiles",
      "mesh/tileset.json",
      sceneCamera,
      "mesh-tiles",
    ],
    ...["tiled-vtkjs", "cesium", "three-tiles"].flatMap((viewer) => [
      [
        viewer + "-mesh-tiles",
        viewer,
        "3d-tiles",
        "mesh/tileset.json",
        sceneCamera,
        "mesh-tiles",
      ],
      [
        viewer + "-gaussian-tiles",
        viewer,
        "gaussian-3d-tiles",
        "gaussian/tileset.json",
        sceneCamera,
        "gaussian-tiles",
      ],
    ]),
    [
      "tiled-vtkjs-mesh",
      "tiled-vtkjs",
      "mesh-in-one-tile",
      "single/tileset.json",
      sceneCamera,
      "single-mesh",
    ],
    [
      "three-mesh",
      "three-mesh",
      "glb",
      "mesh/root.glb",
      sceneCamera,
      "single-mesh",
    ],
    [
      "tiled-vtkjs-points",
      "tiled-vtkjs",
      "copc",
      "multipage.copc.laz",
      { ...pointCamera, target: [120, 220, 70] },
      "copc-points",
    ],
    ...["three-tiles", "cesium", "deck-tiles"].map((viewer) => [
      viewer + "-point-tiles",
      viewer,
      "pnts",
      "point-tiles/tileset.json",
      {
        ...pointCamera,
        target: [120, 220, 70].map((value, i) => value - pointOrigin[i]),
      },
      "copc-points",
    ]),
    [
      "potree-points",
      "potree",
      "potree-2",
      "potree/metadata.json",
      pointCamera,
      "potree-points",
    ],
    [
      "spark-splats",
      "spark",
      "ply-lod",
      "native-splats/fine.ply",
      sceneCamera,
      "native-splats",
    ],
    [
      "playcanvas-splats",
      "playcanvas",
      "streamed-sog",
      "native-splats/lod-meta.json",
      sceneCamera,
      "native-splats",
    ],
    [
      "maplibre-raster",
      "maplibre",
      "xyz",
      "raster/{z}/{x}/{y}.png",
      mapCamera,
      "xyz-raster",
    ],
    [
      "openlayers-cog",
      "openlayers",
      "cog",
      "raster.tif",
      { ...mapCamera, farResolution: 65, nearZoom: 5, panDegrees: 0.018 },
      "cog-overviews",
    ],
  ];
  return {
    assetRoot: "fixtures",
    cases: tests.map(([id, viewer, format, asset, camera, workload]) => ({
      id,
      viewer,
      format,
      asset,
      camera,
      workload,
      ...(["copc", "pnts"].includes(format) ? { pointSize: 2, sse: 1 } : {}),
      ...(viewer === "maplibre" ? { maxZoom: 3 } : {}),
      ...(["3d-tiles", "gaussian-3d-tiles", "mesh-in-one-tile"].includes(format)
        ? { sse: 8 }
        : {}),
    })),
  };
}
