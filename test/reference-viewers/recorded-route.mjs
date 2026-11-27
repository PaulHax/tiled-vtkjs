import { readFile, writeFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";

const [input, output] = process.argv.slice(2);
if (!input || !output)
  throw new Error(
    "Usage: node recorded-route.mjs <scene manifest> <output manifest>",
  );
const recording = JSON.parse(
  await readFile(
    new URL("../recordings/luxembourg-points.json", import.meta.url),
    "utf8",
  ),
);
const doc = JSON.parse(await readFile(input, "utf8"));
const poses = recording.poses;
const angleAt = ({ position, focalPoint }) => {
  const [x, y, z] = position.map((value, index) => value - focalPoint[index]);
  return [
    Math.hypot(x, y, z),
    Math.atan2(y, x),
    Math.atan2(z, Math.hypot(x, y)),
  ];
};
const mix = (a, b, f) => a + (b - a) * f;
const shortest = (a, b) => Math.atan2(Math.sin(b - a), Math.cos(b - a));
const ease = (t) => t * t * (3 - 2 * t);
const at = (ms) =>
  poses.findLast((pose) => pose.atMs <= ms) ?? recording.startPose;
const slice = (begin, end) => [
  [0, angleAt(at(begin))],
  ...poses
    .filter(({ atMs }) => atMs > begin && atMs < end)
    .map((pose) => [pose.atMs - begin, angleAt(pose)]),
  [end - begin, angleAt(at(end))],
];
const approach = slice(14200, 18400);
const orbits = [slice(27000, 35000), slice(35100, 40500)];
const pullBack = slice(40500, 43500);

function routeFor(camera) {
  if (
    camera.type === "map" ||
    Math.abs(camera.up?.[2] ?? 1) !== 1 ||
    !camera.panOffset
  )
    throw new Error(
      "The recorded route expects a Z-up 3D case with two authored points of interest",
    );
  const [, heading, elevation] = angleAt({
    position: camera.direction,
    focalPoint: [0, 0, 0],
  });
  const initial = [camera.farDistance, heading, elevation, ...camera.target];
  const targetA = camera.poiA ?? camera.target;
  const targetB = camera.target.map((x, i) => x + camera.panOffset[i]);
  const transferDistance =
    camera.transferDistance ??
    Math.sqrt(camera.nearDistance * camera.farDistance);
  let state = initial;
  let time = 0;
  const keyframes = [[0, ...initial]];
  const phases = [];
  const segment = (name, duration, sample) => {
    phases.push([time, name]);
    const previous = state;
    for (const [offset, values] of sample(previous)) {
      keyframes.push([time + offset, ...values]);
      state = values;
    }
    time += duration;
  };
  const hold = (name, ms) => segment(name, ms, (start) => [[ms, start]]);
  const move = (name, ms, end) =>
    segment(name, ms, (start) =>
      Array.from({ length: 30 }, (_, i) => {
        const t = (i + 1) / 30;
        const e = ease(t);
        return [
          ms * t,
          [
            Math.exp(mix(Math.log(start[0]), Math.log(end[0]), e)),
            start[1] + shortest(start[1], end[1]) * e,
            ...start.slice(2).map((x, j) => mix(x, end[j + 2], e)),
          ],
        ];
      }),
    );
  const dolly = (name, track, distance, target) => {
    const duration = track.at(-1)[0];
    const from = Math.log(track[0][1][0]);
    const to = Math.log(track.at(-1)[1][0]);
    segment(name, duration, (start) =>
      track.slice(1).map(([offset, [radius]]) => {
        const travel = Math.max(
          0,
          Math.min(1, (Math.log(radius) - from) / (to - from)),
        );
        return [
          offset,
          [
            Math.exp(mix(Math.log(start[0]), Math.log(distance), travel)),
            start[1],
            start[2],
            ...target.map((x, i) =>
              mix(start[i + 3], x, ease(offset / duration)),
            ),
          ],
        ];
      }),
    );
  };
  const orbit = (name, track) => {
    let previous = track[0][1][1];
    let rotation = 0;
    const pitch = track[0][1][2];
    segment(name, track.at(-1)[0], (start) =>
      track.slice(1).map(([offset, [, azimuth, tilt]]) => {
        rotation += shortest(previous, azimuth);
        previous = azimuth;
        return [
          offset,
          [
            start[0],
            start[1] + rotation,
            Math.min(1.22, Math.max(-0.22, start[2] + tilt - pitch)),
            ...start.slice(3),
          ],
        ];
      }),
    );
  };
  hold("overview", 1500);
  dolly("approach-a", approach, camera.nearDistance, targetA);
  hold("inspect-a", 1600);
  orbit("orbit-a", orbits[0]);
  dolly("pull-back", pullBack, transferDistance, targetA);
  move("transfer", 2300, [
    transferDistance,
    heading + 0.38,
    elevation,
    ...targetB,
  ]);
  dolly("approach-b", approach, camera.nearDistance, targetB);
  hold("inspect-b", 1600);
  orbit("orbit-b", orbits[1]);
  move("return", 3500, initial);
  hold("overview-return", 2000);
  const checkpoints = [
    "inspect-a",
    "orbit-a",
    "transfer",
    "inspect-b",
    "orbit-b",
    "overview-return",
  ].map((label) => {
    const index = phases.findIndex(([, name]) => name === label);
    const middle = (phases[index][0] + (phases[index + 1]?.[0] ?? time)) / 2;
    return [keyframes.find(([at]) => at >= middle)[0], label];
  });
  return { keyframes, phases, checkpoints, source: "luxembourg-points.json" };
}

const cases = doc.cases.map(({ traces: _traces, ...entry }) => ({
  ...entry,
  route: routeFor(entry.camera),
}));
const assetDir = resolve(dirname(input), doc.assetRoot);
const assetRoot =
  resolve(dirname(output), doc.assetRoot) === assetDir
    ? doc.assetRoot
    : assetDir;
await writeFile(
  output,
  JSON.stringify({ ...doc, assetRoot, cases }, null, 2) + "\n",
);
console.log(
  basename(output),
  cases.length,
  cases[0]?.route.keyframes.at(-1)[0],
  "ms",
);
