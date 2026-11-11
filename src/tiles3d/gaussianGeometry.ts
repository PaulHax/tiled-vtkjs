import { invert, type Mat16 } from "../core/mat4";
import type { DecodedGaussianPrimitive } from "./decode";

export type GaussianShape = {
  scales: [number, number, number];
  /** Mapper convention, real component first. */
  rotation: [number, number, number, number];
};

export const createGaussianShapeTransform = (matrix: Mat16) => {
  const lengths = [0, 4, 8].map((offset) =>
    Math.hypot(matrix[offset]!, matrix[offset + 1]!, matrix[offset + 2]!),
  );
  const scale = lengths[0]!;
  const dot = (left: number, right: number) =>
    matrix[left]! * matrix[right]! +
    matrix[left + 1]! * matrix[right + 1]! +
    matrix[left + 2]! * matrix[right + 2]!;
  const similarity =
    scale > 0 &&
    lengths.every((value) => Math.abs(value - scale) <= 1e-8 * scale) &&
    [
      [0, 4],
      [0, 8],
      [4, 8],
    ].every(([a, b]) => Math.abs(dot(a!, b!)) <= 1e-8 * scale * scale);
  const determinant =
    matrix[0]! * (matrix[5]! * matrix[10]! - matrix[6]! * matrix[9]!) -
    matrix[4]! * (matrix[1]! * matrix[10]! - matrix[2]! * matrix[9]!) +
    matrix[8]! * (matrix[1]! * matrix[6]! - matrix[2]! * matrix[5]!);
  if (!similarity || determinant <= 0)
    return (scales: ArrayLike<number>, quaternion: ArrayLike<number>) =>
      transformedGaussianShape(scales, quaternion, matrix);
  const [aw, ax, ay, az] = rotationQuaternion([
    matrix[0]! / scale,
    matrix[4]! / scale,
    matrix[8]! / scale,
    matrix[1]! / scale,
    matrix[5]! / scale,
    matrix[9]! / scale,
    matrix[2]! / scale,
    matrix[6]! / scale,
    matrix[10]! / scale,
  ]);
  return (
    scales: ArrayLike<number>,
    quaternion: ArrayLike<number>,
  ): GaussianShape => {
    const length = Math.hypot(
      quaternion[0]!,
      quaternion[1]!,
      quaternion[2]!,
      quaternion[3]!,
    );
    const x = quaternion[0]! / length,
      y = quaternion[1]! / length,
      z = quaternion[2]! / length,
      w = quaternion[3]! / length;
    return {
      scales: [scales[0]! * scale, scales[1]! * scale, scales[2]! * scale],
      rotation: [
        aw * w - ax * x - ay * y - az * z,
        aw * x + ax * w + ay * z - az * y,
        aw * y - ax * z + ay * w + az * x,
        aw * z + ax * y - ay * x + az * w,
      ],
    };
  };
};

const rotationQuaternion = (
  r: readonly number[],
): [number, number, number, number] => {
  const trace = r[0]! + r[4]! + r[8]!;
  let w: number, x: number, y: number, z: number;
  if (trace > 0) {
    const s = 2 * Math.sqrt(trace + 1);
    w = s / 4;
    x = (r[7]! - r[5]!) / s;
    y = (r[2]! - r[6]!) / s;
    z = (r[3]! - r[1]!) / s;
  } else if (r[0]! > r[4]! && r[0]! > r[8]!) {
    const s = 2 * Math.sqrt(1 + r[0]! - r[4]! - r[8]!);
    w = (r[7]! - r[5]!) / s;
    x = s / 4;
    y = (r[1]! + r[3]!) / s;
    z = (r[2]! + r[6]!) / s;
  } else if (r[4]! > r[8]!) {
    const s = 2 * Math.sqrt(1 + r[4]! - r[0]! - r[8]!);
    w = (r[2]! - r[6]!) / s;
    x = (r[1]! + r[3]!) / s;
    y = s / 4;
    z = (r[5]! + r[7]!) / s;
  } else {
    const s = 2 * Math.sqrt(1 + r[8]! - r[0]! - r[4]!);
    w = (r[3]! - r[1]!) / s;
    x = (r[2]! + r[6]!) / s;
    y = (r[5]! + r[7]!) / s;
    z = s / 4;
  }
  const length = Math.hypot(w, x, y, z);
  return [w / length, x / length, y / length, z / length];
};

/** Transform the full covariance, including nonuniform scale and shear. */
export const transformedGaussianShape = (
  scales: ArrayLike<number>,
  quaternion: ArrayLike<number>,
  matrix: Mat16,
): GaussianShape => {
  const qlength = Math.hypot(
    quaternion[0]!,
    quaternion[1]!,
    quaternion[2]!,
    quaternion[3]!,
  );
  const x = quaternion[0]! / qlength,
    y = quaternion[1]! / qlength;
  const z = quaternion[2]! / qlength,
    w = quaternion[3]! / qlength;
  const r = [
    1 - 2 * (y * y + z * z),
    2 * (x * y - w * z),
    2 * (x * z + w * y),
    2 * (x * y + w * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z - w * x),
    2 * (x * z - w * y),
    2 * (y * z + w * x),
    1 - 2 * (x * x + y * y),
  ];
  const b = Array.from({ length: 9 }, () => 0);
  for (let row = 0; row < 3; row++)
    for (let column = 0; column < 3; column++) {
      b[row * 3 + column] =
        scales[column]! *
        (matrix[row]! * r[column]! +
          matrix[4 + row]! * r[3 + column]! +
          matrix[8 + row]! * r[6 + column]!);
    }
  // Orthogonal transformed axes need no eigensolve (the usual tile placement).
  const lengths = [0, 1, 2].map((c) => Math.hypot(b[c]!, b[3 + c]!, b[6 + c]!));
  const dot = (a: number, c: number) =>
    b[a]! * b[c]! + b[3 + a]! * b[3 + c]! + b[6 + a]! * b[6 + c]!;
  if (
    lengths.every((v) => v > 0) &&
    [
      [0, 1],
      [0, 2],
      [1, 2],
    ].every(
      ([a, c]) => Math.abs(dot(a!, c!)) <= 1e-8 * lengths[a!]! * lengths[c!]!,
    )
  ) {
    const basis = b.map((value, index) => value / lengths[index % 3]!);
    const determinant =
      basis[0]! * (basis[4]! * basis[8]! - basis[5]! * basis[7]!) -
      basis[1]! * (basis[3]! * basis[8]! - basis[5]! * basis[6]!) +
      basis[2]! * (basis[3]! * basis[7]! - basis[4]! * basis[6]!);
    if (determinant < 0)
      for (let row = 0; row < 3; row++) basis[row * 3] = -basis[row * 3]!;
    return {
      scales: lengths as [number, number, number],
      rotation: rotationQuaternion(basis),
    };
  }
  const covariance = Array.from({ length: 9 }, () => 0);
  for (let row = 0; row < 3; row++)
    for (let column = 0; column < 3; column++) {
      covariance[row * 3 + column] =
        b[row * 3]! * b[column * 3]! +
        b[row * 3 + 1]! * b[column * 3 + 1]! +
        b[row * 3 + 2]! * b[column * 3 + 2]!;
    }
  const vectors = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 12; sweep++) {
    let changed = false;
    for (const [p, q] of [
      [0, 1],
      [0, 2],
      [1, 2],
    ] as const) {
      const pq = covariance[p * 3 + q]!;
      const pp = covariance[p * 3 + p]!,
        qq = covariance[q * 3 + q]!;
      if (Math.abs(pq) <= 1e-13 * Math.max(Math.abs(pp), Math.abs(qq), 1e-30))
        continue;
      changed = true;
      const angle = 0.5 * Math.atan2(2 * pq, qq - pp);
      const c = Math.cos(angle),
        s = Math.sin(angle);
      for (let row = 0; row < 3; row++) {
        const a = covariance[row * 3 + p]!,
          d = covariance[row * 3 + q]!;
        covariance[row * 3 + p] = c * a - s * d;
        covariance[row * 3 + q] = s * a + c * d;
        const v = vectors[row * 3 + p]!,
          u = vectors[row * 3 + q]!;
        vectors[row * 3 + p] = c * v - s * u;
        vectors[row * 3 + q] = s * v + c * u;
      }
      for (let column = 0; column < 3; column++) {
        const a = covariance[p * 3 + column]!,
          d = covariance[q * 3 + column]!;
        covariance[p * 3 + column] = c * a - s * d;
        covariance[q * 3 + column] = s * a + c * d;
      }
    }
    if (!changed) break;
  }
  const determinant =
    vectors[0]! * (vectors[4]! * vectors[8]! - vectors[5]! * vectors[7]!) -
    vectors[1]! * (vectors[3]! * vectors[8]! - vectors[5]! * vectors[6]!) +
    vectors[2]! * (vectors[3]! * vectors[7]! - vectors[4]! * vectors[6]!);
  if (determinant < 0)
    for (let row = 0; row < 3; row++) vectors[row * 3] = -vectors[row * 3]!;
  return {
    scales: [
      Math.sqrt(Math.max(0, covariance[0]!)),
      Math.sqrt(Math.max(0, covariance[4]!)),
      Math.sqrt(Math.max(0, covariance[8]!)),
    ],
    rotation: rotationQuaternion(vectors),
  };
};

export const gaussianModelCamera = (
  matrix: Mat16,
  camera: readonly number[],
): [number, number, number] => {
  const inverse = invert(matrix);
  if (!inverse)
    throw new Error(
      "Gaussian placement must be invertible for directional appearance",
    );
  const x = camera[0]! - matrix[12]!,
    y = camera[1]! - matrix[13]!,
    z = camera[2]! - matrix[14]!;
  return [
    inverse[0]! * x + inverse[4]! * y + inverse[8]! * z,
    inverse[1]! * x + inverse[5]! * y + inverse[9]! * z,
    inverse[2]! * x + inverse[6]! * y + inverse[10]! * z,
  ];
};

/** KHR real spherical harmonics, evaluated in the primitive's model frame. */
export const writeGaussianColor = (
  primitive: DecodedGaussianPrimitive,
  splat: number,
  camera: readonly number[],
  target: Float32Array,
  offset: number,
): void => {
  if (primitive.harmonicDegree === 0) {
    for (let channel = 0; channel < 3; channel++)
      target[offset + channel] = Math.max(
        0,
        0.5 +
          0.2820947917738781 *
            primitive.sphericalHarmonics[splat * 3 + channel]!,
      );
    target[offset + 3] = primitive.opacities[splat]!;
    return;
  }
  const position = splat * 3;
  let x = primitive.positions[position]! - camera[0]!;
  let y = primitive.positions[position + 1]! - camera[1]!;
  let z = primitive.positions[position + 2]! - camera[2]!;
  const radius = Math.hypot(x, y, z);
  if (radius > 0) {
    x /= radius;
    y /= radius;
    z /= radius;
  } else {
    x = 0;
    y = 0;
    z = 1;
  }
  const basis = [0.2820947917738781];
  if (primitive.harmonicDegree >= 1)
    basis.push(
      -0.4886025119029199 * y,
      0.4886025119029199 * z,
      -0.4886025119029199 * x,
    );
  if (primitive.harmonicDegree >= 2)
    basis.push(
      1.092548430592079 * x * y,
      -1.092548430592079 * y * z,
      0.31539156525252 * (2 * z * z - x * x - y * y),
      -1.092548430592079 * x * z,
      0.5462742152960395 * (x * x - y * y),
    );
  if (primitive.harmonicDegree >= 3)
    basis.push(
      -0.5900435899266435 * y * (3 * x * x - y * y),
      2.890611442640554 * x * y * z,
      -0.4570457994644657 * y * (4 * z * z - x * x - y * y),
      0.3731763325901154 * z * (2 * z * z - 3 * x * x - 3 * y * y),
      -0.4570457994644657 * x * (4 * z * z - x * x - y * y),
      1.445305721320277 * z * (x * x - y * y),
      -0.5900435899266435 * x * (x * x - 3 * y * y),
    );
  const start = splat * basis.length * 3;
  for (let channel = 0; channel < 3; channel++) {
    let value = 0.5;
    for (let coefficient = 0; coefficient < basis.length; coefficient++)
      value +=
        basis[coefficient]! *
        primitive.sphericalHarmonics[start + coefficient * 3 + channel]!;
    target[offset + channel] = Math.max(0, value);
  }
  target[offset + 3] = primitive.opacities[splat]!;
};
