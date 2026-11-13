import { describe, expect, it } from "vitest";
import { IDENTITY } from "../core/mat4";
import type { DecodedGaussianPrimitive } from "./decode";
import {
  createGaussianShapeTransform,
  gaussianModelCamera,
  transformedGaussianShape,
  writeGaussianColor,
} from "./gaussianGeometry";

const primitive = (): DecodedGaussianPrimitive => ({
  kind: "gaussian",
  positions: new Float32Array([0, 0, 0]),
  scales: new Float32Array([1, 1, 1]),
  rotations: new Float32Array([0, 0, 0, 1]),
  opacities: new Float32Array([0.8]),
  harmonicDegree: 1,
  sphericalHarmonics: new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
  modelToScene: new Float64Array(IDENTITY),
  profile: {
    kernel: "ellipse",
    projection: "perspective",
    sortingMethod: "cameraDistance",
    colorSpace: "srgb_rec709_display",
  },
});

describe("Gaussian placement and appearance", () => {
  it("retains the covariance of a rotated ellipse under nonuniform scale", () => {
    const shape = transformedGaussianShape(
      [2, 1, 1],
      [0, 0, Math.sin(Math.PI / 8), Math.cos(Math.PI / 8)],
      [3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    );
    const [w, x, y, z] = shape.rotation;
    const first = [
      1 - 2 * (y * y + z * z),
      2 * (x * y + w * z),
      2 * (x * z - w * y),
    ];
    const second = [
      2 * (x * y - w * z),
      1 - 2 * (x * x + z * z),
      2 * (y * z + w * x),
    ];
    const third = [
      2 * (x * z + w * y),
      2 * (y * z - w * x),
      1 - 2 * (x * x + y * y),
    ];
    const covariance = (row: number, column: number) =>
      [first, second, third].reduce(
        (sum, axis, index) =>
          sum + axis[row]! * axis[column]! * shape.scales[index]! ** 2,
        0,
      );
    expect(covariance(0, 0)).toBeCloseTo(22.5, 8);
    expect(covariance(1, 1)).toBeCloseTo(10, 8);
    expect(covariance(0, 1)).toBeCloseTo(9, 8);
    expect(covariance(2, 2)).toBeCloseTo(1, 8);
  });

  it("uses the same shape for a rigid placement through the fast path", () => {
    const matrix = [0, 2, 0, 0, -2, 0, 0, 0, 0, 0, 2, 0, 6000000, 10, 20, 1];
    const transform = createGaussianShapeTransform(matrix);
    const shape = transform([3, 2, 1], [0, 0, 0, 1]);
    expect(shape.scales).toEqual([6, 4, 2]);
    expect(shape.rotation[0]).toBeCloseTo(Math.SQRT1_2, 14);
    expect(shape.rotation[3]).toBeCloseTo(Math.SQRT1_2, 14);
    expect(shape.rotation.slice(1, 3)).toEqual([0, 0]);
  });

  it("evaluates directional appearance in the placed primitive model frame", () => {
    const data = primitive();
    const color = new Float32Array(4);
    const matrix = [
      0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 6000000.125, 10, 20, 1,
    ];
    writeGaussianColor(
      data,
      0,
      gaussianModelCamera(matrix, [6000001.125, 10, 20]),
      color,
      0,
    );
    expect(color[0]).toBeCloseTo(0.5 - 0.4886025119029199, 6);
    expect(color[3]).toBeCloseTo(0.8, 6);
    writeGaussianColor(
      data,
      0,
      gaussianModelCamera(matrix, [5999999.125, 10, 20]),
      color,
      0,
    );
    expect(color[0]).toBeCloseTo(0.5 + 0.4886025119029199, 6);
  });

  it("retains reconstructed color above one and clamps negative channels only", () => {
    const data = primitive();
    data.harmonicDegree = 0;
    data.sphericalHarmonics = new Float32Array([10, -10, 0]);
    const color = new Float32Array(4);
    writeGaussianColor(data, 0, [0, 0, 0], color, 0);
    expect(color[0]).toBeCloseTo(3.320947917738781, 6);
    expect(color[1]).toBe(0);
    expect(color[2]).toBe(0.5);
  });
});
