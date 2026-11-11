import type { Mat4 } from "../../core/mat4";
import type { MemoryWork } from "../../core/memoryResources";

export type CompressedTextureFormat =
  | "astc-4x4"
  | "bc7"
  | "etc2-rgba8"
  | "s3tc-dxt5";

export type DecodeTextureTarget = CompressedTextureFormat | "rgba";

export type TextureCapabilities = {
  /** Stable identity supplied by the live render context. */
  capabilityKey: string;
  /** Canonical formats supported by that context; order is ignored. */
  compressedFormats: readonly CompressedTextureFormat[];
};

export type DecodeWasmUrls = {
  draco?: {
    wrapperUrl: string;
    wasmUrl: string;
  };
  basis?: {
    encoderUrl: string;
    wasmUrl: string;
  };
};

export type SerializableSampler = {
  magFilter: 9728 | 9729;
  minFilter: 9728 | 9729 | 9984 | 9985 | 9986 | 9987;
  wrapS: 33071 | 33648 | 10497;
  wrapT: 33071 | 33648 | 10497;
};

export type SerializableMaterial = {
  version: 1;
  kind: "gltf-material";
  name?: string;
  alphaMode: "OPAQUE" | "MASK" | "BLEND";
  alphaCutoff: number;
  doubleSided: boolean;
  /** KHR_materials_unlit: base color is authored radiance, not a light input. */
  unlit: boolean;
  metallicFactor: number;
  roughnessFactor: number;
  emissiveFactor: [number, number, number];
  baseColorTexture?: {
    texture: number;
    texCoord: number;
    orientation: string;
    sourceColorSpace: "srgb" | "linear";
  };
};

export type TileDecodeStage = "profile" | "decode";

/** Renderer-neutral typed failure carrying the tile and decode stage. */
export class TileDecodeError extends Error {
  readonly tileUri: string;
  readonly stage: TileDecodeStage;
  readonly reason: string;

  constructor(
    tileUri: string,
    stage: TileDecodeStage,
    reason: string,
    options?: ErrorOptions,
  ) {
    super(
      `failed to decode 3D tile ${tileUri} at ${stage}: ${reason}`,
      options,
    );
    this.name = "TileDecodeError";
    this.tileUri = tileUri;
    this.stage = stage;
    this.reason = reason;
  }
}

/** A required glTF extension has no decoder implementation. */
export class TileUnsupportedExtensionError extends TileDecodeError {
  readonly extension: string;

  constructor(tileUri: string, extension: string) {
    super(
      tileUri,
      "profile",
      `unsupported required glTF extension ${extension}`,
    );
    this.name = "TileUnsupportedExtensionError";
    this.extension = extension;
  }
}

/** A Gaussian primitive declares a rendering profile this decoder cannot carry. */
export class TileUnsupportedGaussianProfileError extends TileDecodeError {
  readonly property: string;
  readonly value: unknown;

  constructor(tileUri: string, property: string, value: unknown) {
    super(
      tileUri,
      "profile",
      `unsupported KHR_gaussian_splatting ${property}: ${JSON.stringify(value)}`,
    );
    this.name = "TileUnsupportedGaussianProfileError";
    this.property = property;
    this.value = value;
  }
}

export type CompressedTextureLevel = {
  width: number;
  height: number;
  data: Uint8Array;
};

export type DecodedCompressedTexture = {
  kind: "compressed";
  format: CompressedTextureFormat;
  width: number;
  height: number;
  colorSpace: "srgb";
  levels: CompressedTextureLevel[];
  sampler: SerializableSampler;
  capabilityKey: string;
};

export type DecodedRgbaTexture = {
  kind: "rgba";
  rgba: Uint8Array;
  width: number;
  height: number;
  colorSpace: "srgb";
  sampler: SerializableSampler;
};

export type DecodedTexture = DecodedCompressedTexture | DecodedRgbaTexture;

export type DecodedMaterial = {
  baseColorFactor: [number, number, number, number];
  baseColorTexture?: DecodedTexture;
  raw: SerializableMaterial;
};

export type DecodedMeshPrimitive = {
  kind: "mesh";
  positions: Float32Array;
  normals?: Float32Array;
  colors?: Float32Array;
  uvs?: Float32Array;
  indices?: Uint16Array | Uint32Array;
  material: DecodedMaterial;
};

export type GaussianAppearanceProfile = {
  kernel: "ellipse";
  colorSpace: "srgb_rec709_display" | "lin_rec709_display";
  projection: "perspective";
  sortingMethod: "cameraDistance";
};

/** Gaussian attributes stay in the primitive's model frame. */
export type DecodedGaussianPrimitive = {
  kind: "gaussian";
  positions: Float32Array;
  /** Unit quaternions in glTF XYZW order. */
  rotations: Float32Array;
  /** Linear standard deviations, without logarithmic activation. */
  scales: Float32Array;
  opacities: Float32Array;
  /** EXT_splat_opacity uses a power kernel for semantic opacity above one. */
  opacityKernel?: "power";
  /** Splat-major, then coefficient in degree/order sequence, then RGB. */
  sphericalHarmonics: Float32Array;
  harmonicDegree: 0 | 1 | 2 | 3;
  profile: GaussianAppearanceProfile;
  /** Full column-major placement, independent of the tile's mesh RTC origin. */
  modelToScene: Float64Array;
};

export type DecodedPrimitive = DecodedMeshPrimitive | DecodedGaussianPrimitive;

export type DecodedTileContent = {
  primitives: DecodedPrimitive[];
  /** Float64 scene-local origin retained as JavaScript numbers. */
  origin: [number, number, number];
  byteEstimate: {
    geometry: number;
    textures: number;
  };
  /** Worker-side aggregate timings retained for public operational diagnostics. */
  diagnostics?: {
    totalDecodeMs: number;
    /** One-time worker-local Basis wrapper/WASM initialization time. */
    basisRuntimeInitializationMs: number;
    /** Sum of per-texture KTX2 transcodes, excluding runtime initialization. */
    basisTranscodeMs: number;
    basisTranscodeSamplesMs: number[];
    basisTextures: number;
    basisTarget: DecodeTextureTarget | null;
  };
};

export type DecodedMeshTileContent = Omit<DecodedTileContent, "primitives"> & {
  primitives: DecodedMeshPrimitive[];
};

export type BasisTargetTimingStats = {
  readonly count: number;
  readonly totalMs: number;
  /** Bounded tail of individual texture transcode durations. */
  readonly samplesMs: readonly number[];
};

export type DecodeTileRequest = {
  content: ArrayBuffer;
  contentUrl: string;
  /** Root below which external glTF dependencies may resolve. */
  dependencyRootUrl?: string;
  revision: string;
  accumulatedTransform: Mat4;
  tilesetToScene: Mat4;
  textureCapabilities: TextureCapabilities;
  wasm?: DecodeWasmUrls;
};

export type DecodeJob = {
  promise: Promise<DecodedTileContent>;
  cancel(): void;
};

/** Minimal pool surface supplied to renderer-neutral members. */
export type DecodeWorkerPoolHandle = {
  decode(
    request: DecodeTileRequest,
    options?: { readonly memory?: MemoryWork },
  ): DecodeJob;
  stats?(): {
    readonly size: number;
    readonly queuedJobs: number;
    readonly activeJobs: number;
    readonly completedJobs: number;
    readonly failedJobs: number;
    readonly cancelledJobs: number;
    readonly workerElapsedMs: number;
    readonly basisRuntimeInitializationMs: number;
    readonly basisTranscodeMs: number;
    readonly basisTextures: number;
    readonly basisTargets: Readonly<Record<string, number>>;
    readonly basisTargetTimings: Readonly<
      Record<string, BasisTargetTimingStats>
    >;
  };
  dispose?(): void;
};
