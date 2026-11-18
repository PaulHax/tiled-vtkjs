import type { DecodedGaussianPrimitive } from "./decode/types";
import {
  createGaussianShapeTransform,
  writeGaussianColor,
} from "./gaussianGeometry";
import {
  GAUSSIAN_ATTRIBUTE_BYTES_PER_TUPLE,
  type createGaussianStorage,
  type GaussianBounds,
} from "./gaussianStorage";

export type GaussianTupleRange = { offset: number; count: number };

export type GaussianPackedRange = GaussianTupleRange & {
  primitive: DecodedGaussianPrimitive;
  matrix: readonly number[];
  allocationRevision: number;
};

export type GaussianAttributes = {
  positions: Float32Array;
  scales: Float32Array;
  rotations: Float32Array;
  colors: Float32Array;
};

type PackingStorage = GaussianAttributes & {
  arena: ReturnType<typeof createGaussianStorage>;
  origin: readonly [number, number, number];
};

export function* packGaussianRanges(
  storage: PackingStorage,
  ranges: readonly GaussianPackedRange[],
  tuplesPerStep: number,
  onPacked: () => void,
) {
  for (const packed of ranges) {
    const { primitive, matrix, offset, count, allocationRevision } = packed;
    if (storage.arena.isPacked(allocationRevision)) continue;
    const { origin } = storage;
    const transformShape = createGaussianShapeTransform(matrix);
    const staticCamera = [0, 0, 0] as const;
    const partial = storage.arena.packingProgress(allocationRevision);
    const bounds: GaussianBounds = partial?.bounds ?? [
      Infinity,
      -Infinity,
      Infinity,
      -Infinity,
      Infinity,
      -Infinity,
    ];
    for (
      let first = partial?.count ?? 0;
      first < count;
      first += tuplesPerStep
    ) {
      const end = Math.min(count, first + tuplesPerStep);
      yield {
        bytes: (end - first) * GAUSSIAN_ATTRIBUTE_BYTES_PER_TUPLE,
        run() {
          for (let index = first; index < end; index++) {
            const p = index * 3,
              q = index * 4,
              destination = offset + index;
            const x = primitive.positions[p]!,
              y = primitive.positions[p + 1]!,
              z = primitive.positions[p + 2]!;
            storage.positions[destination * 3] =
              matrix[0]! * x +
              matrix[4]! * y +
              matrix[8]! * z +
              (matrix[12]! - origin[0]);
            storage.positions[destination * 3 + 1] =
              matrix[1]! * x +
              matrix[5]! * y +
              matrix[9]! * z +
              (matrix[13]! - origin[1]);
            storage.positions[destination * 3 + 2] =
              matrix[2]! * x +
              matrix[6]! * y +
              matrix[10]! * z +
              (matrix[14]! - origin[2]);
            for (let axis = 0; axis < 3; axis++) {
              const value = storage.positions[destination * 3 + axis]!;
              bounds[axis * 2] = Math.min(bounds[axis * 2]!, value);
              bounds[axis * 2 + 1] = Math.max(bounds[axis * 2 + 1]!, value);
            }
            const shape = transformShape(
              primitive.scales.subarray(p, p + 3),
              primitive.rotations.subarray(q, q + 4),
            );
            storage.scales.set(shape.scales, destination * 3);
            storage.rotations.set(shape.rotation, destination * 4);
            if (primitive.harmonicDegree === 0)
              writeGaussianColor(
                primitive,
                index,
                staticCamera,
                storage.colors,
                destination * 4,
              );
          }
          if (end === count) {
            storage.arena.markPacked(allocationRevision, bounds, offset, count);
            onPacked();
          } else storage.arena.markPartial(allocationRevision, end, bounds);
        },
      };
    }
  }
}
