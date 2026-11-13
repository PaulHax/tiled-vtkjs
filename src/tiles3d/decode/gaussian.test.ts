import { describe, expect, it } from "vitest";

import {
  gaussianGlb,
  type GaussianFixtureOptions,
} from "../../../test/fixtures/gaussianGlb";
import { IDENTITY } from "../../core/mat4";
import { buildTransferList, type DecodeTileRequest } from ".";
import { decodeTileContent } from "./decoder";

const request = (options: GaussianFixtureOptions = {}): DecodeTileRequest => ({
  content: gaussianGlb(options),
  contentUrl: "https://fixture.invalid/gaussian.glb",
  revision: "synthetic",
  accumulatedTransform: [
    ...IDENTITY,
  ] as DecodeTileRequest["accumulatedTransform"],
  tilesetToScene: [...IDENTITY] as DecodeTileRequest["tilesetToScene"],
  textureCapabilities: { capabilityKey: "rgba", compressedFormats: [] },
});

describe("Gaussian glTF content decoding", () => {
  it("preserves local centers, XYZW, linear scale, unclamped harmonics and default profile", async () => {
    const result = await decodeTileContent(request({ degree: 3 }));
    const primitive = result.primitives[0]!;
    expect(primitive.kind).toBe("gaussian");
    if (primitive.kind !== "gaussian")
      throw new Error("expected Gaussian primitive");
    expect([...primitive.positions]).toEqual([
      0.125, 0.25, 0.5, 1, 2, 3, -1, -2, -3,
    ]);
    expect([...primitive.rotations]).toEqual([
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    ]);
    expect(primitive.scales[0]).toBeCloseTo(0.1);
    expect([...primitive.opacities]).toEqual([0, 0.5, 1]);
    expect(primitive.profile).toEqual({
      kernel: "ellipse",
      colorSpace: "srgb_rec709_display",
      projection: "perspective",
      sortingMethod: "cameraDistance",
    });
    expect(primitive.harmonicDegree).toBe(3);
    expect(primitive.sphericalHarmonics).toHaveLength(3 * 16 * 3);
    expect(Array.from(primitive.sphericalHarmonics.slice(0, 3))).toEqual([
      0, -1, 2,
    ]);
    expect(
      Array.from(primitive.sphericalHarmonics.slice(16 * 3, 16 * 3 + 3)),
    ).toEqual([0, -2, 3]);
    expect(primitive.sphericalHarmonics[15 * 3]).toBeCloseTo(3.6);
    expect([...primitive.modelToScene]).toEqual([
      1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1,
    ]);
  });

  it("keeps node/tile/ECEF placement in Float64 without Earth-scale Float32 centers", async () => {
    const input = request({
      mutate: (json) => {
        json.nodes[0]!.translation = [0.125, 2, 3];
        json.nodes[0]!.rotation = [0, 0, Math.SQRT1_2, Math.SQRT1_2];
        json.nodes[0]!.scale = [2, 3, 4];
      },
    });
    input.accumulatedTransform[12] = 6_378_137.125;
    input.accumulatedTransform[13] = 0.25;
    input.tilesetToScene[12] = -6_378_137;
    const result = await decodeTileContent(input);
    const primitive = result.primitives[0]!;
    if (primitive.kind !== "gaussian")
      throw new Error("expected Gaussian primitive");
    expect(primitive.positions[0]).toBe(0.125);
    expect(primitive.modelToScene).toBeInstanceOf(Float64Array);
    expect(primitive.modelToScene[12]).toBe(0.25);
    expect(primitive.modelToScene[13]).toBe(-2.75);
    expect(primitive.modelToScene[14]).toBe(2);
    expect(primitive.modelToScene[2]).toBeCloseTo(2);
    expect(primitive.modelToScene[4]).toBeCloseTo(-3);
    expect(primitive.modelToScene[9]).toBeCloseTo(-4);
  });

  it("decodes normalized quaternion/opacity accessors and unsigned linear scales", async () => {
    const result = await decodeTileContent(request({ quantized: true }));
    const primitive = result.primitives[0]!;
    if (primitive.kind !== "gaussian")
      throw new Error("expected Gaussian primitive");
    expect([...primitive.rotations]).toEqual([
      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    ]);
    expect([...primitive.scales]).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(primitive.opacities[1]).toBeCloseTo(127 / 255);
  });

  it("keeps mixed meshes and splats distinct and ignores the splat's material", async () => {
    const result = await decodeTileContent(
      request({
        mixed: true,
        mutate: (json) => {
          json.meshes[0]!.primitives[0]!.material = 999;
        },
      }),
    );
    expect(result.primitives.map((primitive) => primitive.kind)).toEqual([
      "gaussian",
      "mesh",
    ]);
    const mesh = result.primitives[1]!;
    if (mesh.kind !== "mesh") throw new Error("expected mesh primitive");
    expect(mesh.material.raw.alphaMode).toBe("OPAQUE");
  });

  it("charges and transfers every distinct attribute and Float64 placement buffer", async () => {
    const result = await decodeTileContent(request({ degree: 3 }));
    const buffers = buildTransferList(result);
    expect(buffers).toHaveLength(6);
    expect(new Set(buffers).size).toBe(buffers.length);
    expect(result.byteEstimate).toEqual({
      geometry: buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0),
      textures: 0,
    });
    const copy = structuredClone(result, { transfer: buffers });
    expect(result.primitives[0]!.positions.byteLength).toBe(0);
    const primitive = copy.primitives[0]!;
    if (primitive.kind !== "gaussian")
      throw new Error("expected Gaussian primitive");
    expect(primitive.modelToScene).toBeInstanceOf(Float64Array);
    expect(primitive.sphericalHarmonics).toHaveLength(144);
  });

  it.each([
    ["kernel", "sphere"],
    ["colorSpace", "unknown"],
    ["projection", "orthographic"],
    ["sortingMethod", "viewDepth"],
  ])("names unsupported %s profiles", async (property, value) => {
    await expect(
      decodeTileContent(request({ extension: { [property]: value } })),
    ).rejects.toMatchObject({
      name: "TileUnsupportedGaussianProfileError",
      tileUri: "https://fixture.invalid/gaussian.glb",
      stage: "profile",
      property,
      value,
    });
  });

  it.each([
    ["OPACITY", 2, /OPACITY must be in/],
    ["SCALE", -1, /SCALE must be non-negative/],
    ["ROTATION", 5, /ROTATION must contain unit XYZW/],
    ["SH_DEGREE_0_COEF_0", NaN, /finite values/],
  ] as const)(
    "rejects invalid %s attributes",
    async (attribute, value, reason) => {
      await expect(
        decodeTileContent(
          request({
            mutate: (_json, arrays) => {
              arrays.get(`KHR_gaussian_splatting:${attribute}`)![0] = value;
            },
          }),
        ),
      ).rejects.toThrow(reason);
    },
  );

  it("rejects partial and skipped harmonic degrees", async () => {
    for (const semantic of [
      "SH_DEGREE_1_COEF_0",
      "SH_DEGREE_1_COEF_1",
      "SH_DEGREE_1_COEF_2",
    ]) {
      await expect(
        decodeTileContent(
          request({
            degree: 2,
            mutate: (json) => {
              delete json.meshes[0]!.primitives[0]!.attributes[
                `KHR_gaussian_splatting:${semantic}`
              ];
            },
          }),
        ),
      ).rejects.toThrow(/all coefficients and all lower degrees/);
    }
    await expect(
      decodeTileContent(
        request({
          degree: 2,
          mutate: (json) => {
            for (let coefficient = 0; coefficient < 3; coefficient += 1)
              delete json.meshes[0]!.primitives[0]!.attributes[
                `KHR_gaussian_splatting:SH_DEGREE_1_COEF_${coefficient}`
              ];
          },
        }),
      ),
    ).rejects.toThrow(/all coefficients and all lower degrees/);
  });

  it("rejects missing required attributes, wrong encodings, wrong mode and compression by name", async () => {
    await expect(
      decodeTileContent(
        request({
          mutate: (json) => {
            delete json.meshes[0]!.primitives[0]!.attributes[
              "KHR_gaussian_splatting:SCALE"
            ];
          },
        }),
      ),
    ).rejects.toThrow(/missing KHR_gaussian_splatting:SCALE accessor/);
    await expect(
      decodeTileContent(
        request({
          quantized: true,
          mutate: (json) => {
            json.accessors[1]!.normalized = false;
          },
        }),
      ),
    ).rejects.toThrow(
      /unsupported KHR_gaussian_splatting:ROTATION accessor encoding/,
    );
    await expect(
      decodeTileContent(
        request({
          mutate: (json) => {
            json.meshes[0]!.primitives[0]!.mode = 4;
          },
        }),
      ),
    ).rejects.toThrow(/mode must be POINTS/);
    await expect(
      decodeTileContent(
        request({
          mutate: (json) => {
            json.extensionsRequired.push(
              "KHR_gaussian_splatting_compression_unknown",
            );
          },
        }),
      ),
    ).rejects.toMatchObject({
      name: "TileUnsupportedExtensionError",
      extension: "KHR_gaussian_splatting_compression_unknown",
    });
  });
});
