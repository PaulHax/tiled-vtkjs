import type {
  DecodedGaussianPrimitive,
  GaussianAppearanceProfile,
} from "./types";
import { TileDecodeError, TileUnsupportedGaussianProfileError } from "./types";

export const GAUSSIAN_EXTENSION = "KHR_gaussian_splatting";

type AttributeEncoding = {
  componentType: number;
  normalized: boolean;
};

type GaussianInput = {
  extension: unknown;
  mode: number | undefined;
  indexed: boolean;
  attributeNames: readonly string[];
  readAttribute: (
    semantic: string,
    type: "SCALAR" | "VEC3" | "VEC4",
    encodings: readonly AttributeEncoding[],
  ) => Float32Array;
  modelToScene: readonly number[];
  tileUri: string;
};

const FLOAT = [{ componentType: 5126, normalized: false }];
const UNSIGNED_NORMALIZED = [
  { componentType: 5121, normalized: true },
  { componentType: 5123, normalized: true },
];

export const gaussianProfileFor = (
  input: unknown,
  tileUri: string,
): GaussianAppearanceProfile => {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new TileDecodeError(
      tileUri,
      "profile",
      `${GAUSSIAN_EXTENSION} must be an object`,
    );
  }
  const extension = input as Record<string, unknown>;
  const profile = {
    kernel: extension.kernel,
    colorSpace: extension.colorSpace,
    projection: Object.hasOwn(extension, "projection")
      ? extension.projection
      : "perspective",
    sortingMethod: Object.hasOwn(extension, "sortingMethod")
      ? extension.sortingMethod
      : "cameraDistance",
  };
  const supported: Record<keyof GaussianAppearanceProfile, readonly unknown[]> =
    {
      kernel: ["ellipse"],
      colorSpace: ["srgb_rec709_display", "lin_rec709_display"],
      projection: ["perspective"],
      sortingMethod: ["cameraDistance"],
    };
  for (const property of Object.keys(
    supported,
  ) as (keyof GaussianAppearanceProfile)[]) {
    if (!supported[property].includes(profile[property])) {
      throw new TileUnsupportedGaussianProfileError(
        tileUri,
        property,
        profile[property],
      );
    }
  }
  return profile as GaussianAppearanceProfile;
};

const gaussianSemantic = (attribute: string): string =>
  `${GAUSSIAN_EXTENSION}:${attribute}`;
const harmonicSemantic = (degree: number, coefficient: number): string =>
  gaussianSemantic(`SH_DEGREE_${degree}_COEF_${coefficient}`);

/** Decode the standard ellipse kernel without baking placement into float attributes. */
export const decodeGaussian = (
  input: GaussianInput,
): DecodedGaussianPrimitive => {
  const profile = gaussianProfileFor(input.extension, input.tileUri);
  const fail = (reason: string): never => {
    throw new TileDecodeError(
      input.tileUri,
      "decode",
      `${GAUSSIAN_EXTENSION}: ${reason}`,
    );
  };
  if (input.mode !== 0) fail("ellipse primitive mode must be POINTS (0)");
  // Indexed draws are legal glTF POINTS but do not define a distinct Gaussian
  // field profile here. Reject rather than silently duplicating or dropping splats.
  if (input.indexed) fail("indexed Gaussian primitives are not supported");
  const positions = input.readAttribute("POSITION", "VEC3", [
    ...FLOAT,
    { componentType: 5120, normalized: false },
    { componentType: 5120, normalized: true },
    { componentType: 5121, normalized: false },
    { componentType: 5121, normalized: true },
    { componentType: 5122, normalized: false },
    { componentType: 5122, normalized: true },
    { componentType: 5123, normalized: false },
    { componentType: 5123, normalized: true },
  ]);
  const rotations = input.readAttribute(gaussianSemantic("ROTATION"), "VEC4", [
    ...FLOAT,
    { componentType: 5120, normalized: true },
    { componentType: 5122, normalized: true },
  ]);
  const scales = input.readAttribute(gaussianSemantic("SCALE"), "VEC3", [
    ...FLOAT,
    ...UNSIGNED_NORMALIZED,
    { componentType: 5121, normalized: false },
    { componentType: 5123, normalized: false },
  ]);
  const opacities = input.readAttribute(gaussianSemantic("OPACITY"), "SCALAR", [
    ...FLOAT,
    ...UNSIGNED_NORMALIZED,
  ]);
  const count = positions.length / 3;
  if (!Number.isSafeInteger(count) || count <= 0)
    fail("POSITION must contain splats");
  const checked = (
    array: Float32Array,
    components: number,
    label: string,
  ): Float32Array => {
    if (
      array.length !== count * components ||
      array.some((value) => !Number.isFinite(value))
    ) {
      fail(`${label} must match POSITION and contain finite values`);
    }
    return array;
  };
  checked(positions, 3, "POSITION");
  checked(rotations, 4, "ROTATION");
  checked(scales, 3, "SCALE");
  checked(opacities, 1, "OPACITY");
  if (scales.some((value) => value < 0))
    fail("SCALE must be non-negative linear standard deviations");
  if (opacities.some((value) => value < 0 || value > 1))
    fail("OPACITY must be in [0, 1]");
  for (let offset = 0; offset < rotations.length; offset += 4) {
    const length = Math.hypot(
      rotations[offset]!,
      rotations[offset + 1]!,
      rotations[offset + 2]!,
      rotations[offset + 3]!,
    );
    // snorm8 unit quaternions have quantization error up to roughly 1/127.
    if (Math.abs(length - 1) > 0.02)
      fail("ROTATION must contain unit XYZW quaternions");
  }
  const coefficients: Float32Array[] = [];
  let harmonicDegree: 0 | 1 | 2 | 3 = 0;
  let missingDegree = false;
  const expectedHarmonics = new Set<string>();
  for (let degree = 0; degree <= 3; degree += 1) {
    const semantics = Array.from({ length: 2 * degree + 1 }, (_, index) =>
      harmonicSemantic(degree, index),
    );
    semantics.forEach((semantic) => expectedHarmonics.add(semantic));
    const present = semantics.filter((semantic) =>
      input.attributeNames.includes(semantic),
    ).length;
    if (present === 0 && degree > 0) {
      missingDegree = true;
      continue;
    }
    if (present !== semantics.length || missingDegree)
      fail(
        `SH degree ${degree} must include all coefficients and all lower degrees`,
      );
    harmonicDegree = degree as 0 | 1 | 2 | 3;
    for (const semantic of semantics)
      coefficients.push(
        checked(input.readAttribute(semantic, "VEC3", FLOAT), 3, semantic),
      );
  }
  for (const semantic of input.attributeNames) {
    if (
      semantic.startsWith(gaussianSemantic("SH_")) &&
      !expectedHarmonics.has(semantic)
    ) {
      fail(`unsupported spherical harmonic attribute ${semantic}`);
    }
  }
  const sphericalHarmonics = new Float32Array(count * coefficients.length * 3);
  for (let splat = 0; splat < count; splat += 1) {
    for (
      let coefficient = 0;
      coefficient < coefficients.length;
      coefficient += 1
    ) {
      const source = coefficients[coefficient]!;
      sphericalHarmonics.set(
        source.subarray(splat * 3, splat * 3 + 3),
        (splat * coefficients.length + coefficient) * 3,
      );
    }
  }
  return {
    kind: "gaussian",
    positions,
    rotations,
    scales,
    opacities,
    sphericalHarmonics,
    harmonicDegree,
    profile,
    modelToScene: new Float64Array(input.modelToScene),
  };
};
