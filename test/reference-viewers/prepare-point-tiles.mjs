import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { Bounds, Copc, Key } from "copc";

export async function writePointTiles(source, output) {
  const copc = await Copc.create(source);
  const origin = copc.header.min.map((v, i) => (v + copc.header.max[i]) / 2);
  const pages = [copc.info.rootHierarchyPage];
  const nodes = new Map();
  while (pages.length) {
    const page = await Copc.loadHierarchyPage(source, pages.pop());
    Object.entries(page.nodes).forEach(([id, node]) => nodes.set(id, node));
    pages.push(...Object.values(page.pages));
  }
  await mkdir(output, { recursive: true });
  const tiles = new Map();
  let shift = 0;
  let total = 0;
  for (const [id, node] of [...nodes].sort(
    ([a], [b]) => Key.parse(a)[0] - Key.parse(b)[0],
  )) {
    const bounds = Bounds.stepTo(copc.info.cube, Key.parse(id));
    const center = origin.map((v, i) => (bounds[i] + bounds[i + 3]) / 2 - v);
    const [x, y, z] = origin.map((_, i) => (bounds[i + 3] - bounds[i]) / 2);
    const tile = {
      boundingVolume: { box: [...center, x, 0, 0, 0, y, 0, 0, 0, z] },
      geometricError: copc.info.spacing / 2 ** Key.parse(id)[0],
      refine: "ADD",
    };
    tiles.set(id, tile);
    if (Key.parse(id)[0]) {
      const parent = tiles.get(Key.toString(Key.up(id)));
      (parent.children ??= []).push(tile);
    }
    if (!node.pointCount) continue;
    const view = await Copc.loadPointDataView(source, copc, node);
    const xyz = ["X", "Y", "Z"].map((axis) => view.getter(axis));
    const rgb = ["Red", "Green", "Blue"].every((k) => k in view.dimensions)
      ? ["Red", "Green", "Blue"].map((k) => view.getter(k))
      : [];
    if (id === "0-0-0-0" && rgb.length) {
      for (let i = 0; i < view.pointCount; i++) {
        if (rgb.some((channel) => channel(i) > 255)) {
          shift = 8;
          break;
        }
      }
    }
    const feature = {
      POINTS_LENGTH: view.pointCount,
      POSITION: { byteOffset: 0 },
      ...(rgb.length ? { RGB: { byteOffset: view.pointCount * 12 } } : {}),
      RTC_CENTER: center,
    };
    const json = Buffer.from(JSON.stringify(feature));
    const start = Math.ceil((28 + json.length) / 8) * 8;
    const length =
      Math.ceil((start + view.pointCount * (12 + rgb.length)) / 8) * 8;
    const bytes = Buffer.alloc(length);
    bytes.write("pnts");
    bytes.writeUInt32LE(1, 4);
    bytes.writeUInt32LE(length, 8);
    bytes.writeUInt32LE(start - 28, 12);
    bytes.writeUInt32LE(length - start, 16);
    bytes.fill(32, 28, start);
    json.copy(bytes, 28);
    for (let i = 0; i < view.pointCount; i++) {
      for (let c = 0; c < 3; c++) {
        bytes.writeFloatLE(
          xyz[c](i) - origin[c] - center[c],
          start + 12 * i + 4 * c,
        );
        if (rgb.length)
          bytes[start + view.pointCount * 12 + 3 * i + c] = Math.min(
            255,
            rgb[c](i) >> shift,
          );
      }
    }
    tile.content = { uri: id + ".pnts" };
    await writeFile(join(output, tile.content.uri), bytes);
    total += view.pointCount;
  }
  if (total !== copc.header.pointCount)
    throw new Error("COPC hierarchy point count mismatch");
  for (const tile of tiles.values())
    if (!tile.children) tile.geometricError = 0;
  const root = tiles.get("0-0-0-0");
  await writeFile(
    join(output, "tileset.json"),
    JSON.stringify({
      asset: { version: "1.0" },
      geometricError: root.geometricError,
      root,
    }),
  );
  return { origin, tiles: tiles.size, points: total };
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  const [, , source, recording, output] = process.argv;
  if (!source || !recording || !output)
    throw new Error(
      "Usage: node prepare-point-tiles.mjs source.copc.laz recording.json output-directory",
    );
  const poses = JSON.parse(await readFile(recording, "utf8"));
  const assets = join(output, "assets");
  const capture = await writePointTiles(source, join(assets, "points"));
  await copyFile(source, join(assets, "source.copc.laz"));
  const frames = [
    { ...poses.startPose, atMs: 0 },
    ...poses.poses.filter((p) => p.atMs > 0),
  ];
  const route = frames.map(
    ({ position, focalPoint, viewUp, viewAngle, parallelProjection, atMs }) => {
      const [x, y, z] = position.map((v, i) => v - focalPoint[i]);
      const radius = Math.hypot(x, y, z);
      const horizontal = Math.hypot(x, y);
      // Z-up routes preserve pitch and heading and cannot represent camera roll.
      const right = (viewUp[0] * y - viewUp[1] * x) / horizontal;
      const up =
        (horizontal * viewUp[2] -
          ((viewUp[0] * x + viewUp[1] * y) * z) / horizontal) /
        radius;
      if (
        parallelProjection ||
        viewAngle !== 30 ||
        !horizontal ||
        !(up > 0) ||
        Math.abs(Math.atan2(right, up)) > 1e-5
      )
        throw new Error(
          "Recorded point-cloud route requires a 30-degree Z-up perspective camera",
        );
      return [
        atMs,
        radius,
        Math.atan2(y, x),
        Math.atan2(z, horizontal),
        ...focalPoint,
      ];
    },
  );
  const cases = ["tiled-vtkjs", "three-tiles", "cesium", "deck-tiles"].map(
    (viewer) => {
      const native = viewer === "tiled-vtkjs";
      const keyframes = native
        ? route
        : route.map(([at, radius, heading, pitch, ...target]) => [
            at,
            radius,
            heading,
            pitch,
            ...target.map((v, i) => v - capture.origin[i]),
          ]);
      return {
        id: viewer + "-points",
        viewer,
        workload: "recorded-lidar",
        asset: native ? "source.copc.laz" : "points/tileset.json",
        format: native ? "copc" : "pnts",
        pointSize: 2,
        sse: 1,
        camera: { target: keyframes[0].slice(4) },
        route: { keyframes, phases: [[0, "route"]], checkpoints: [] },
      };
    },
  );
  await writeFile(
    join(output, "manifest.json"),
    JSON.stringify({ assetRoot: "assets", cases }) + "\n",
  );
  console.log(JSON.stringify(capture));
}
