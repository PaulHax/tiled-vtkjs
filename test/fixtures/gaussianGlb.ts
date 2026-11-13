type FixtureAccessor = {
  bufferView: number;
  componentType: number;
  count: number;
  type: string;
  normalized?: boolean;
};

type FixturePrimitive = {
  mode: number;
  attributes: Record<string, number>;
  indices?: number;
  material?: number;
  extensions?: Record<string, unknown>;
};

export type GaussianFixtureDocument = {
  asset: { version: string; generator: string };
  extensionsUsed: string[];
  extensionsRequired: string[];
  scene: number;
  scenes: { nodes: number[] }[];
  nodes: {
    mesh: number;
    translation?: number[];
    rotation?: number[];
    scale?: number[];
  }[];
  meshes: { primitives: FixturePrimitive[] }[];
  buffers: { byteLength: number }[];
  bufferViews: { buffer: number; byteOffset: number; byteLength: number }[];
  accessors: FixtureAccessor[];
};

export type GaussianFixtureOptions = {
  degree?: 0 | 1 | 2 | 3;
  quantized?: boolean;
  mixed?: boolean;
  extension?: Record<string, unknown>;
  mutate?: (
    json: GaussianFixtureDocument,
    arrays: Map<string, Float32Array | Int8Array | Uint8Array>,
  ) => void;
};

/** Synthetic unlocated standards fixture, not a geographic reference asset. */
export const gaussianGlb = (
  options: GaussianFixtureOptions = {},
): ArrayBuffer => {
  const extension = "KHR_gaussian_splatting";
  const attributes: Record<string, number> = {};
  const arrays = new Map<string, Float32Array | Int8Array | Uint8Array>();
  const json: GaussianFixtureDocument = {
    asset: { version: "2.0", generator: "Synthetic Gaussian decoder fixture" },
    extensionsUsed: [extension],
    extensionsRequired: [extension],
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [
      {
        primitives: [
          {
            mode: 0,
            attributes,
            extensions: {
              [extension]: {
                kernel: "ellipse",
                colorSpace: "srgb_rec709_display",
                ...options.extension,
              },
            },
          },
        ],
      },
    ],
    buffers: [{ byteLength: 0 }],
    bufferViews: [],
    accessors: [],
  };
  const add = (
    semantic: string,
    array: Float32Array | Int8Array | Uint8Array,
    type: string,
    normalized = false,
  ): void => {
    attributes[semantic] = json.accessors.length;
    json.accessors.push({
      bufferView: json.bufferViews.length,
      componentType:
        array instanceof Float32Array
          ? 5126
          : array instanceof Int8Array
            ? 5120
            : 5121,
      count: 3,
      type,
      ...(normalized ? { normalized: true } : {}),
    });
    const byteOffset = Math.ceil(json.buffers[0]!.byteLength / 4) * 4;
    json.bufferViews.push({
      buffer: 0,
      byteOffset,
      byteLength: array.byteLength,
    });
    json.buffers[0]!.byteLength = byteOffset + array.byteLength;
    arrays.set(semantic, array);
  };
  add(
    "POSITION",
    new Float32Array([0.125, 0.25, 0.5, 1, 2, 3, -1, -2, -3]),
    "VEC3",
  );
  add(
    `${extension}:ROTATION`,
    options.quantized
      ? new Int8Array([0, 0, 0, 127, 0, 0, 0, 127, 0, 0, 0, 127])
      : new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
    "VEC4",
    options.quantized,
  );
  add(
    `${extension}:SCALE`,
    options.quantized
      ? new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9])
      : new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    "VEC3",
  );
  add(
    `${extension}:OPACITY`,
    options.quantized
      ? new Uint8Array([0, 127, 255])
      : new Float32Array([0, 0.5, 1]),
    "SCALAR",
    options.quantized,
  );
  for (let degree = 0; degree <= (options.degree ?? 0); degree += 1) {
    for (let coefficient = 0; coefficient <= 2 * degree; coefficient += 1) {
      add(
        `${extension}:SH_DEGREE_${degree}_COEF_${coefficient}`,
        new Float32Array([
          degree + coefficient / 10,
          -1,
          2,
          degree + coefficient / 10,
          -2,
          3,
          degree + coefficient / 10,
          -3,
          4,
        ]),
        "VEC3",
      );
    }
  }
  if (options.mixed)
    json.meshes[0]!.primitives.push({
      mode: 4,
      attributes: { POSITION: attributes.POSITION! },
    });
  options.mutate?.(json, arrays);
  const binary = new Uint8Array(Math.ceil(json.buffers[0]!.byteLength / 4) * 4);
  let index = 0;
  for (const array of arrays.values()) {
    const view = json.bufferViews[index++]!;
    binary.set(
      new Uint8Array(array.buffer, array.byteOffset, array.byteLength),
      view.byteOffset,
    );
  }
  const encoded = new TextEncoder().encode(JSON.stringify(json));
  const jsonLength = Math.ceil(encoded.byteLength / 4) * 4;
  const content = new ArrayBuffer(12 + 8 + jsonLength + 8 + binary.byteLength);
  const header = new DataView(content);
  header.setUint32(0, 0x46546c67, true);
  header.setUint32(4, 2, true);
  header.setUint32(8, content.byteLength, true);
  header.setUint32(12, jsonLength, true);
  header.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(content, 20, jsonLength).fill(32);
  new Uint8Array(content, 20, encoded.byteLength).set(encoded);
  header.setUint32(20 + jsonLength, binary.byteLength, true);
  header.setUint32(24 + jsonLength, 0x004e4942, true);
  new Uint8Array(content, 28 + jsonLength).set(binary);
  return content;
};
