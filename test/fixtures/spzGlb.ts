import createSpzModule from "@adobe/spz";

export const spzGlb = async ({
  degree = 0,
  opacity = false,
  compressedBytes,
  mutate,
}: {
  degree?: 0 | 1 | 2 | 3;
  opacity?: boolean;
  compressedBytes?: Uint8Array;
  mutate?: (json: Record<string, any>) => void;
} = {}): Promise<ArrayBuffer> => {
  const spz = await createSpzModule();
  const bytes =
    compressedBytes ??
    spz.saveSpzToBuffer(
      {
        numPoints: 2,
        shDegree: degree,
        antialiased: false,
        extensions: [],
        positions: Float32Array.of(1, 2, 3, 4, 5, 6),
        scales: Float32Array.of(-1, -2, -3, -1, -2, -3),
        rotations: Float32Array.of(0, 0, 0, 1, 0, 0, 0, 1),
        alphas: Float32Array.of(0, Infinity),
        colors: Float32Array.of(-1, 0, 1, 1, 2, 3),
        sh: Float32Array.from(
          { length: 2 * ((degree + 1) ** 2 - 1) * 3 },
          (_, index) => (index % 17) / 32 - 0.25,
        ),
      },
      {
        version: 3,
        from: spz.CoordinateSystem.UNSPECIFIED,
        sh1Bits: 8,
        shRestBits: 8,
      },
    );
  const aligned = Math.ceil(bytes.length / 4) * 4;
  const binary = new Uint8Array(aligned + (opacity ? 8 : 0));
  binary.set(bytes);
  const extensions: Record<string, unknown> = {
    KHR_gaussian_splatting_compression_spz_2: { bufferView: 0 },
  };
  const json = {
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    extensionsUsed: [
      "KHR_gaussian_splatting",
      "KHR_gaussian_splatting_compression_spz_2",
    ],
    extensionsRequired: [
      "KHR_gaussian_splatting",
      "KHR_gaussian_splatting_compression_spz_2",
    ],
    meshes: [
      {
        primitives: [
          {
            mode: 0,
            attributes: { POSITION: 0 },
            extensions: {
              KHR_gaussian_splatting: { kernel: "ellipse", extensions },
            },
          },
        ],
      },
    ],
    buffers: [{ byteLength: binary.length }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: bytes.length }],
    accessors: [
      { componentType: 5126, count: 2, type: "VEC3" } as Record<
        string,
        unknown
      >,
    ],
  };
  if (opacity) {
    extensions.EXT_splat_opacity = { opacityAccessor: 1 };
    json.extensionsUsed.push("EXT_splat_opacity");
    json.bufferViews.push({ buffer: 0, byteOffset: aligned, byteLength: 8 });
    json.accessors.push({
      componentType: 5126,
      count: 2,
      type: "SCALAR",
      bufferView: 1,
    });
    new DataView(binary.buffer).setFloat32(aligned, 0.5, true);
    new DataView(binary.buffer).setFloat32(aligned + 4, 1.25, true);
  }
  mutate?.(json);
  const encoded = new TextEncoder().encode(JSON.stringify(json));
  const jsonLength = Math.ceil(encoded.length / 4) * 4;
  const output = new ArrayBuffer(28 + jsonLength + binary.length);
  const header = new DataView(output);
  header.setUint32(0, 0x46546c67, true);
  header.setUint32(4, 2, true);
  header.setUint32(8, output.byteLength, true);
  header.setUint32(12, jsonLength, true);
  header.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(output, 20, jsonLength).fill(32);
  new Uint8Array(output, 20, encoded.length).set(encoded);
  header.setUint32(20 + jsonLength, binary.length, true);
  header.setUint32(24 + jsonLength, 0x004e4942, true);
  new Uint8Array(output, 28 + jsonLength).set(binary);
  return output;
};
