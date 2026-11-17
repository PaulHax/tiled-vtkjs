import { beforeEach, describe, expect, it, vi } from "vitest";

import { IDENTITY } from "../core/mat4";
import { createSubmissionScheduler } from "../core/submissionScheduler";
import type {
  DecodedGaussianPrimitive,
  DecodedMeshTileContent,
  DecodedTileContent,
} from "./decode";
import { TileContentRealizationError } from "./contentAdapter";
import { createMeshContentAdapter } from "./meshContentAdapter";
import { actorInstances, resetStubs } from "../../test/stubs/vtkStub";

const mesh = (): DecodedMeshTileContent => ({
  origin: [0, 0, 0],
  primitives: [
    {
      kind: "mesh",
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      material: {
        baseColorFactor: [1, 1, 1, 1],
        raw: {
          version: 1,
          kind: "gltf-material",
          alphaMode: "OPAQUE",
          alphaCutoff: 0.5,
          doubleSided: false,
          unlit: false,
          metallicFactor: 1,
          roughnessFactor: 1,
          emissiveFactor: [0, 0, 0],
        },
      },
    },
  ],
  byteEstimate: { geometry: 36, textures: 0 },
});

const gaussian = (): DecodedGaussianPrimitive => ({
  kind: "gaussian",
  positions: new Float32Array([0, 0, 0]),
  rotations: new Float32Array([0, 0, 0, 1]),
  scales: new Float32Array([1, 1, 1]),
  opacities: new Float32Array([1]),
  sphericalHarmonics: new Float32Array([0, 0, 0]),
  harmonicDegree: 0,
  modelToScene: new Float64Array(IDENTITY),
  profile: {
    kernel: "ellipse",
    colorSpace: "srgb_rec709_display",
    projection: "perspective",
    sortingMethod: "cameraDistance",
  },
});

const mixed = (): DecodedTileContent => {
  const result: DecodedTileContent = mesh();
  result.primitives.push(gaussian());
  return result;
};

const harness = () => {
  const scheduleRender = vi.fn();
  const submissions = createSubmissionScheduler({
    scheduleRender,
    maxBytesPerFrame: 1024,
    now: () => 0,
  });
  const onError = vi.fn();
  const adapter = createMeshContentAdapter({
    renderer: { addActor: vi.fn(), removeActor: vi.fn() },
    scheduleRender,
    submissions,
    onError,
  });
  return { adapter, submissions, onError };
};

describe("mesh content realization", () => {
  beforeEach(resetStubs);

  it("rejects mixed content before drawing triangles or exposing picks", () => {
    const { adapter, submissions, onError } = harness();
    expect(adapter.submitTile("mixed", mixed())).toBe("failed");
    submissions.prepareFrame();
    adapter.setDrawnTiles(["mixed"]);
    expect(actorInstances).toHaveLength(0);
    expect(adapter.pickableMeshTiles()).toEqual([]);
    expect(adapter.tileState("mixed")).toBe("failed");
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "TileContentRealizationError",
        tileId: "mixed",
        primitiveKind: "gaussian",
      }),
    );
    expect(onError.mock.calls[0]![0]).toBeInstanceOf(
      TileContentRealizationError,
    );
    adapter.submitTile("mixed", mixed());
    expect(onError).toHaveBeenCalledTimes(1);
  });

  it("preserves an admitted parent when a replacement group is unsupported", () => {
    const { adapter, submissions } = harness();
    adapter.submitTile("parent", mesh());
    submissions.prepareFrame();
    adapter.setDrawnTiles(["parent"]);
    const parent = adapter.pickableMeshTiles()[0];
    expect(
      adapter.submitTileGroup(
        [
          { id: "child-mesh", content: mesh() },
          { id: "child-splats", content: mixed() },
        ],
        ["parent"],
      ),
    ).toBe("failed");
    submissions.prepareFrame();
    expect(adapter.submittedTileIds()).toEqual(["parent"]);
    expect(adapter.pickableMeshTiles()).toEqual([parent]);
    expect(adapter.tileState("child-mesh")).toBe("absent");
    expect(actorInstances).toHaveLength(1);
  });

  it("clears rejection state on source replacement and keeps mesh lifecycle", () => {
    const { adapter, submissions } = harness();
    adapter.submitTile("tile", mixed());
    adapter.clearTiles();
    expect(adapter.tileState("tile")).toBe("absent");
    expect(adapter.submitTile("tile", mesh())).toBe("queued");
    submissions.prepareFrame();
    adapter.setDrawnTiles(["tile"]);
    expect(adapter.pickableMeshTiles()).toHaveLength(1);
    adapter.setVisible(false);
    expect(actorInstances[0]!.visibility).toBe(false);
    adapter.dispose();
    expect(adapter.pickableMeshTiles()).toEqual([]);
    expect(adapter.submitTile("later", mesh())).toBe("failed");
  });
});
