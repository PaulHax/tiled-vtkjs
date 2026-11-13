import createSpzModule, { type SpzModule } from "@adobe/spz";

import { gaussianProfileFor } from "./gaussian";
import {
  TileDecodeError,
  TileUnsupportedGaussianProfileError,
  type DecodedGaussianPrimitive,
} from "./types";

export const SPZ_EXTENSION = "KHR_gaussian_splatting_compression_spz_2";
export const SPLAT_OPACITY_EXTENSION = "EXT_splat_opacity";

let runtime: Promise<SpzModule> | null = null;
const spzRuntime = (): Promise<SpzModule> => {
  runtime ??= createSpzModule().catch((error: unknown) => {
    runtime = null;
    throw error;
  });
  return runtime;
};

type SpzGaussianInput = {
  bytes: Uint8Array;
  extension: Record<string, unknown>;
  count: number;
  modelToScene: readonly number[];
  tileUri: string;
  opacity?: {
    extension: Record<string, unknown>;
    readAccessor(
      index: number,
      encoding: "float32" | "float16" | "unorm16",
    ): Float32Array;
  };
};

const finiteArray = (array: Float32Array): boolean =>
  array.every(Number.isFinite);

/** The draft SPZ profile supplies sRGB DC coefficients, logits and log scales. */
export const decodeSpzGaussian = async (
  input: SpzGaussianInput,
): Promise<DecodedGaussianPrimitive> => {
  const profile = gaussianProfileFor(
    { colorSpace: "srgb_rec709_display", ...input.extension },
    input.tileUri,
  );
  if (profile.colorSpace !== "srgb_rec709_display") {
    throw new TileUnsupportedGaussianProfileError(
      input.tileUri,
      "SPZ colorSpace",
      profile.colorSpace,
    );
  }
  const fail = (reason: string): never => {
    throw new TileDecodeError(
      input.tileUri,
      "decode",
      `${SPZ_EXTENSION}: ${reason}`,
    );
  };
  if (!Number.isSafeInteger(input.count) || input.count < 1)
    fail("POSITION count must be a positive integer");
  const spz = await spzRuntime();
  const pointer = spz._malloc(input.bytes.byteLength);
  if (!pointer) fail("could not allocate compressed tile storage");
  let result: DecodedGaussianPrimitive | null = null;
  let completed = false;
  let error: string | null = null;
  const received = new Uint32Array(6);
  try {
    spz.HEAPU8.set(input.bytes, pointer);
    spz.loadSpzStreaming(
      pointer,
      input.bytes.byteLength,
      // Compressed coordinates are authored in the glTF node's local frame.
      // Its matrix and the 3D Tiles up-axis transform supply the placement.
      { to: spz.CoordinateSystem.UNSPECIFIED },
      {
        onHeader(header) {
          if (header.numPoints !== input.count)
            fail("decoded point count does not match POSITION");
          if (
            !Number.isInteger(header.shDegree) ||
            header.shDegree < 0 ||
            header.shDegree > 3
          ) {
            throw new TileUnsupportedGaussianProfileError(
              input.tileUri,
              "SPZ harmonicDegree",
              header.shDegree,
            );
          }
          if (header.antialiased) {
            throw new TileUnsupportedGaussianProfileError(
              input.tileUri,
              "SPZ antialiased",
              true,
            );
          }
          result = {
            kind: "gaussian",
            positions: new Float32Array(input.count * 3),
            rotations: new Float32Array(input.count * 4),
            scales: new Float32Array(input.count * 3),
            opacities: new Float32Array(input.count),
            sphericalHarmonics: new Float32Array(
              input.count * (header.shDegree + 1) ** 2 * 3,
            ),
            harmonicDegree: header.shDegree as 0 | 1 | 2 | 3,
            profile,
            modelToScene: new Float64Array(input.modelToScene),
          };
        },
        onChunk(attribute, offset, values) {
          if (!result) fail("attribute data arrived before its header");
          const output = result!;
          const coefficientCount = (output.harmonicDegree + 1) ** 2;
          const widths = [3, 1, 3, 3, 4, (coefficientCount - 1) * 3];
          const width = widths[attribute] ?? 0;
          if (
            !width ||
            values.length % width !== 0 ||
            offset !== received[attribute]
          )
            fail("invalid attribute chunk sequence");
          const points = values.length / width;
          if (
            offset + points > input.count ||
            (attribute === 1 ? values.some(Number.isNaN) : !finiteArray(values))
          )
            fail("invalid attribute values");
          received[attribute]! += points;
          if (attribute === 0) output.positions.set(values, offset * 3);
          else if (attribute === 1) {
            for (let index = 0; index < values.length; index += 1) {
              output.opacities[offset + index] =
                1 / (1 + Math.exp(-values[index]!));
            }
          } else if (attribute === 3) {
            for (let index = 0; index < values.length; index += 1) {
              const scale = Math.exp(values[index]!);
              if (!Number.isFinite(Math.fround(scale)))
                fail("non-finite decoded scale");
              output.scales[offset * 3 + index] = scale;
            }
          } else if (attribute === 4) output.rotations.set(values, offset * 4);
          else {
            const destinationWidth = coefficientCount * 3;
            const coefficientOffset = attribute === 2 ? 0 : 3;
            for (let point = 0; point < points; point += 1) {
              output.sphericalHarmonics.set(
                values.subarray(point * width, (point + 1) * width),
                (offset + point) * destinationWidth + coefficientOffset,
              );
            }
          }
        },
        onDone() {
          completed = true;
        },
        onError(message) {
          error = message;
        },
      },
    );
  } finally {
    spz._free(pointer);
  }
  if (error || !completed || !result)
    fail(error ?? "SPZ decoder returned incomplete data");
  const output: DecodedGaussianPrimitive = result!;
  for (let attribute = 0; attribute < 6; attribute += 1) {
    if (attribute === 5 && output.harmonicDegree === 0) continue;
    if (received[attribute] !== input.count)
      fail("SPZ decoder returned incomplete attributes");
  }
  applySplatOpacity(output, input.opacity);
  return output;
};

const accessorIndex = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value >= 0;

/** Optional producer opacity is applied only after its complete layout validates. */
const applySplatOpacity = (
  output: DecodedGaussianPrimitive,
  source: SpzGaussianInput["opacity"],
): void => {
  if (!source) return;
  const { extension, readAccessor } = source;
  const count = output.opacities.length;
  const version = extension.version ?? 1;
  try {
    if (version === 1 && accessorIndex(extension.opacityAccessor)) {
      const encoded = readAccessor(extension.opacityAccessor, "float32");
      if (
        encoded.length !== count ||
        encoded.some((value) => !Number.isFinite(value) || value < 0)
      )
        return;
      for (let index = 0; index < count; index += 1) {
        const value = encoded[index]!;
        const shape = 1 + 4 * Math.min(Math.max(value - 1, 0), 1);
        output.opacities[index] =
          value > 1
            ? Math.min(1000, Math.exp((shape * shape - 1) / Math.E))
            : value;
      }
      output.opacityKernel = "power";
    } else if (
      version === 2 &&
      accessorIndex(extension.sourceOpacityAccessor) &&
      accessorIndex(extension.coverageBoostRatioAccessor) &&
      extension.sourceOpacityEncoding === "float16" &&
      extension.coverageBoostMethod === "opacity_anisotropic_v1" &&
      typeof extension.coverageBoostScale === "number" &&
      Number.isFinite(extension.coverageBoostScale) &&
      extension.coverageBoostScale >= 0
    ) {
      const opacity = readAccessor(extension.sourceOpacityAccessor, "float16");
      const ratio = readAccessor(
        extension.coverageBoostRatioAccessor,
        "unorm16",
      );
      if (opacity.length !== count || ratio.length !== count) return;
      for (let index = 0; index < count; index += 1) {
        const alpha = opacity[index]!;
        if (
          !Number.isFinite(alpha) ||
          alpha <= 1 ||
          !Number.isFinite(ratio[index])
        )
          continue;
        const growth =
          1 +
          extension.coverageBoostScale * (Math.sqrt(Math.min(alpha, 1000)) - 1);
        const display = alpha / (growth * growth);
        if (Number.isFinite(display) && display >= 0)
          output.opacities[index] = Math.min(display, 1000);
      }
      output.opacityKernel = "power";
    }
  } catch {
    // The extension defines the complete SPZ representation as its fallback.
  }
};
