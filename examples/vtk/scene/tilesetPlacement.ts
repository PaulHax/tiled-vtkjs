import { IDENTITY, multiply, transformPoint } from "../../../src/core/mat4";
import { createEcefToEnuTransform } from "../../../src/tiles3d/rtc";
import type { TilesetTile } from "../../../src/tiles3d/tilesetSource";

type Point = [number, number, number];

/** Bowring latitude followed by ellipsoid-height recovery for an ECEF anchor. */
const geodeticAnchor = ([x, y, z]: readonly [
  number,
  number,
  number,
]): Point => {
  const a = 6378137;
  const e2 = 6.6943799901413165e-3;
  const b = a * Math.sqrt(1 - e2);
  const ep2 = (a * a - b * b) / (b * b);
  const p = Math.hypot(x, y);
  const theta = Math.atan2(z * a, p * b);
  const latitude = Math.atan2(
    z + ep2 * b * Math.sin(theta) ** 3,
    p - e2 * a * Math.cos(theta) ** 3,
  );
  const n = a / Math.sqrt(1 - e2 * Math.sin(latitude) ** 2);
  const height = p > 1e-6 ? p / Math.cos(latitude) - n : Math.abs(z) - b;
  return [
    (Math.atan2(y, x) * 180) / Math.PI,
    (latitude * 180) / Math.PI,
    height,
  ];
};

export const tilesetPlacement = (root: TilesetTile, georeferenced = false) => {
  const ecefCenter = transformPoint(root.worldTransform, [
    ...root.boundingVolume.center,
  ]);
  const location = georeferenced ? geodeticAnchor(ecefCenter) : null;
  const tilesetToScene = location
    ? createEcefToEnuTransform(...location)
    : [...IDENTITY];
  const matrix = multiply(tilesetToScene, root.worldTransform);
  const center = transformPoint(matrix, [...root.boundingVolume.center]);
  const low: Point = [Infinity, Infinity, Infinity];
  const high: Point = [-Infinity, -Infinity, -Infinity];
  for (const x of [-1, 1])
    for (const y of [-1, 1])
      for (const z of [-1, 1]) {
        const local: Point = [...root.boundingVolume.center];
        const axes = root.boundingVolume.halfAxes;
        for (let axis = 0; axis < 3; axis += 1) {
          local[axis] =
            local[axis]! +
            axes[axis]! * x +
            axes[3 + axis]! * y +
            axes[6 + axis]! * z;
        }
        const corner = transformPoint(matrix, local);
        for (let axis = 0; axis < 3; axis += 1) {
          low[axis] = Math.min(low[axis]!, corner[axis]!);
          high[axis] = Math.max(high[axis]!, corner[axis]!);
        }
      }
  const radius =
    Math.hypot(...high.map((value, axis) => value - low[axis]!)) / 2;
  return {
    tilesetToScene,
    location,
    center,
    radius: Math.max(radius, 1),
    bounds: [low[0], high[0], low[1], high[1], low[2], high[2]],
  };
};
