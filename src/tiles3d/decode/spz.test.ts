import { describe, expect, it } from "vitest";
import { gzipSync } from "node:zlib";

import { spzGlb } from "../../../test/fixtures/spzGlb";
import { ResourcePolicyError } from "../../core/memoryResources";
import { IDENTITY, transformPoint } from "../../core/mat4";
import { buildTransferList, type DecodeTileRequest } from ".";
import { decodeTileContent } from "./decoder";

const request = (content: ArrayBuffer): DecodeTileRequest => ({
  content,
  contentUrl: "https://fixture.invalid/spz.glb",
  revision: "synthetic",
  accumulatedTransform: [
    ...IDENTITY,
  ] as DecodeTileRequest["accumulatedTransform"],
  tilesetToScene: [...IDENTITY] as DecodeTileRequest["tilesetToScene"],
  textureCapabilities: { capabilityKey: "rgba", compressedFormats: [] },
});

describe("SPZ Gaussian tile decoding", () => {
  it("joins independently centered compressed tiles in their authored glTF frame", async () => {
    const storedSpz = (positions: number[]): Uint8Array => {
      const count = positions.length / 3;
      const raw = Buffer.alloc(16 + count * 20);
      raw.writeUInt32LE(0x5053474e, 0);
      raw.writeUInt32LE(3, 4);
      raw.writeUInt32LE(count, 8);
      raw[13] = 12;
      for (const [index, value] of positions.entries())
        raw.writeIntLE(value * 4096, 16 + index * 3, 3);
      let offset = 16 + count * 9;
      raw.fill(128, offset, offset + count * 4);
      offset += count * 4;
      raw.fill(160, offset, offset + count * 3);
      offset += count * 3;
      for (let point = 0; point < count; point += 1)
        raw.writeUInt32LE(0xc0000000, offset + point * 4);
      return gzipSync(raw);
    };
    const decoded = await Promise.all(
      [
        { center: [8, 17, 24], positions: [2, -3, -6, 3, -4, -8] },
        { center: [12, 24, 35], positions: [-2, 4, 5, -1, 3, 3] },
      ].map(async ({ center, positions }) => {
        const input = request(
          await spzGlb({
            compressedBytes: storedSpz(positions),
            mutate(json) {
              json.nodes[0].matrix = [
                1,
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                -1,
                0,
                ...center,
                1,
              ];
            },
          }),
        );
        input.accumulatedTransform[12] = 6_378_137.125;
        input.tilesetToScene[12] = -6_378_137;
        const content = await decodeTileContent(input);
        const primitive = content.primitives[0]!;
        if (primitive.kind !== "gaussian") throw new Error("expected Gaussian");
        expect([...primitive.positions]).toEqual(positions);
        return [0, 3].map((offset) =>
          transformPoint(primitive.modelToScene, [
            primitive.positions[offset]!,
            primitive.positions[offset + 1]!,
            primitive.positions[offset + 2]!,
          ]),
        );
      }),
    );
    expect(decoded[0]).toEqual([
      [10.125, -30, 20],
      [11.125, -32, 21],
    ]);
    expect(decoded[1]).toEqual(decoded[0]);
  });

  it.each([0, 1, 2, 3] as const)(
    "decodes degree %i through the official codec without losing SH or placement",
    async (degree) => {
      const input = request(await spzGlb({ degree }));
      input.accumulatedTransform[12] = 6_378_137.125;
      input.tilesetToScene[12] = -6_378_137;
      const decoded = await decodeTileContent(input);
      const gaussian = decoded.primitives[0]!;
      if (gaussian.kind !== "gaussian") throw new Error("expected Gaussian");
      expect([...gaussian.positions]).toEqual([1, 2, 3, 4, 5, 6]);
      expect(gaussian.modelToScene[12]).toBe(0.125);
      expect(gaussian.harmonicDegree).toBe(degree);
      expect(gaussian.sphericalHarmonics.length).toBe(
        2 * (degree + 1) ** 2 * 3,
      );
      expect(gaussian.sphericalHarmonics[0]).toBeCloseTo(-1, 1);
      expect(gaussian.sphericalHarmonics[2]).toBeCloseTo(1, 1);
      if (degree > 0)
        expect(gaussian.sphericalHarmonics[3]).toBeCloseTo(-0.25, 2);
      expect(gaussian.opacities[0]).toBeCloseTo(0.5, 2);
      expect(gaussian.opacities[1]).toBe(1);
      expect(gaussian.scales[0]).toBeCloseTo(Math.exp(-1), 5);
      expect(gaussian.rotations[3]).toBe(1);
      expect(gaussian.profile.colorSpace).toBe("srgb_rec709_display");
      const transfers = buildTransferList(decoded);
      expect(decoded.byteEstimate.geometry).toBe(
        transfers.reduce((sum, bytes) => sum + bytes.byteLength, 0),
      );
    },
  );

  it("preserves the producer's nonlinear opacity kernel as an explicit profile", async () => {
    const result = await decodeTileContent(
      request(await spzGlb({ opacity: true })),
    );
    const gaussian = result.primitives[0]!;
    if (gaussian.kind !== "gaussian") throw new Error("expected Gaussian");
    expect(gaussian.opacityKernel).toBe("power");
    expect(gaussian.opacities[0]).toBe(0.5);
    expect(gaussian.opacities[1]).toBeCloseTo(Math.exp(3 / Math.E), 5);
  });

  it("refuses compressed expansion before entering the codec", async () => {
    const content = await spzGlb({
      mutate(json) {
        for (const accessor of json.accessors) accessor.count = 1_000_000;
      },
    });
    expect(content.byteLength).toBeLessThan(10_000);
    await expect(
      decodeTileContent(request(content), {
        reserveMemory: async (bytes) => {
          if (bytes > 1_000_000)
            throw new ResourcePolicyError("work", bytes, 1_000_000);
        },
      }),
    ).rejects.toMatchObject({
      name: "ResourcePolicyError",
      domain: "work",
      limitBytes: 1_000_000,
    });
  });

  it("rejects compressed attribute count mismatches", async () => {
    const content = await spzGlb({
      mutate: (json) => {
        json.accessors[0].count = 3;
      },
    });
    await expect(decodeTileContent(request(content))).rejects.toThrow(
      "point count",
    );
  });

  it("rejects truncated SPZ buffer views", async () => {
    const content = await spzGlb({
      mutate: (json) => {
        json.bufferViews[0].byteLength += 1024;
      },
    });
    await expect(decodeTileContent(request(content))).rejects.toThrow(
      "bufferView",
    );
  });

  it("contains corrupt compressed input and can decode the next tile", async () => {
    const content = await spzGlb();
    const view = new DataView(content);
    const binaryOffset = 28 + view.getUint32(12, true);
    new Uint8Array(content, binaryOffset, 8).fill(0);
    await expect(decodeTileContent(request(content))).rejects.toMatchObject({
      name: "TileDecodeError",
      stage: "decode",
    });
    const next = await decodeTileContent(request(await spzGlb()));
    expect(next.primitives[0]!.positions.length).toBe(6);
  });

  it("rejects unsupported declared color space instead of interpreting SPZ as linear", async () => {
    const content = await spzGlb({
      mutate: (json) => {
        json.meshes[0].primitives[0].extensions.KHR_gaussian_splatting.colorSpace =
          "lin_rec709_display";
      },
    });
    await expect(decodeTileContent(request(content))).rejects.toMatchObject({
      name: "TileUnsupportedGaussianProfileError",
      property: "SPZ colorSpace",
    });
  });
});
