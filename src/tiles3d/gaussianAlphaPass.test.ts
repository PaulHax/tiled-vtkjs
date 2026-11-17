import { describe, expect, it, vi } from "vitest";
import {
  createGaussianAlphaPass,
  createGaussianContentAdapterFactory,
} from "./gaussianAdapter";
import { createSubmissionScheduler } from "../core/submissionScheduler";

vi.mock("@kitware/vtk.js/Rendering/Core/Actor", async () => {
  const { makeActor } = await import("../../test/stubs/vtkStub");
  return {
    default: { newInstance: () => ({ ...makeActor(), setPickable() {} }) },
  };
});
vi.mock("@kitware/vtk.js/Common/DataModel/PolyData", async () => {
  const { makePolyData } = await import("../../test/stubs/vtkStub");
  return {
    default: {
      newInstance: () => {
        const poly = makePolyData();
        return {
          ...poly,
          getPointData: () => ({ ...poly.getPointData(), addArray() {} }),
        };
      },
    },
  };
});

vi.mock("@kitware/vtk.js/Common/Core/DataArray", async () => {
  const { default: dataArray } = await vi.importActual<
    typeof import("../../test/stubs/vtkDataArray")
  >("../../test/stubs/vtkDataArray");
  return {
    default: {
      newInstance: (...args: Parameters<typeof dataArray.newInstance>) => ({
        ...dataArray.newInstance(...args),
        delete() {},
      }),
    },
  };
});

const ownedGaussian = () => {
  const addActor = vi.fn();
  const adapter = createGaussianContentAdapterFactory({
    createSortBackend: () => ({
      sort: async () => new Uint32Array(),
      dispose() {},
    }),
  })({
    renderer: { addActor, removeActor() {} },
    scheduleRender() {},
    submissions: createSubmissionScheduler({ scheduleRender() {} }),
  });
  return {
    actor: addActor.mock.calls[0]![0] as object,
    dispose: () => adapter.dispose(),
  };
};

// Model the framebuffer blend equation at the actor boundary. Mesh shaders
// return straight RGB; the Gaussian shader returns RGB multiplied by alpha.
const context = () => {
  const gl = {
    BLEND: 1,
    DEPTH_TEST: 2,
    DEPTH_WRITEMASK: 3,
    BLEND_SRC_RGB: 4,
    BLEND_DST_RGB: 5,
    BLEND_SRC_ALPHA: 6,
    BLEND_DST_ALPHA: 7,
    BLEND_EQUATION_RGB: 8,
    BLEND_EQUATION_ALPHA: 9,
    ONE: 10,
    SRC_ALPHA: 11,
    ONE_MINUS_SRC_ALPHA: 12,
    FUNC_ADD: 13,
    enabled: new Set<number>(),
    parameters: new Map<number, number | boolean>(),
    isEnabled(name: number) {
      return this.enabled.has(name);
    },
    enable(name: number) {
      this.enabled.add(name);
    },
    disable(name: number) {
      this.enabled.delete(name);
    },
    getParameter(name: number) {
      return this.parameters.get(name);
    },
    depthMask(value: boolean) {
      this.parameters.set(this.DEPTH_WRITEMASK, value);
    },
    blendFuncSeparate(a: number, b: number, c: number, d: number) {
      [a, b, c, d].forEach((value, index) =>
        this.parameters.set(4 + index, value),
      );
    },
    blendEquationSeparate(rgb: number, alpha: number) {
      this.parameters.set(this.BLEND_EQUATION_RGB, rgb);
      this.parameters.set(this.BLEND_EQUATION_ALPHA, alpha);
    },
  };
  gl.depthMask(true);
  gl.blendFuncSeparate(20, 21, 22, 23);
  gl.blendEquationSeparate(24, 25);
  return gl;
};
type Context = ReturnType<typeof context>;
type Pixel = [number, number, number, number];
const snapshot = (gl: Context) => ({
  enabled: [...gl.enabled],
  parameters: [...gl.parameters],
});
const blendPixel = (gl: Context, output: Pixel, destination: Pixel) => {
  expect(gl.isEnabled(gl.BLEND)).toBe(true);
  expect(gl.isEnabled(gl.DEPTH_TEST)).toBe(true);
  expect(gl.getParameter(gl.DEPTH_WRITEMASK)).toBe(false);
  const sourceFactor =
    gl.getParameter(gl.BLEND_SRC_RGB) === gl.ONE ? 1 : output[3];
  for (let component = 0; component < 3; component++)
    destination[component] =
      output[component]! * sourceFactor +
      destination[component]! * (1 - output[3]);
  destination[3] = output[3] + destination[3] * (1 - output[3]);
};
const actor = (
  gl: Context,
  pixel: Pixel,
  gaussian: boolean,
  output: Pixel,
  owned?: object,
) => ({
  getRenderable: () =>
    owned ?? {
      getMapper: () => ({
        isA: (name: string) => gaussian && name === "vtkPointGaussianMapper",
      }),
    },
  traverse(pass: { getOperation(): string }) {
    expect(pass.getOperation()).toBe("translucentPass");
    blendPixel(gl, output, pixel);
  },
});
const renderer = (children: ReturnType<typeof actor>[]) => ({
  apply: vi.fn(),
  getChildren: () => children,
  traverse(pass: Parameters<ReturnType<typeof actor>["traverse"]>[0]) {
    this.apply(pass, true);
    children.forEach((child) => child.traverse(pass));
    this.apply(pass, false);
  },
});
const passFor = (gl: Context) =>
  createGaussianAlphaPass(() => gl as unknown as WebGL2RenderingContext);

describe("Gaussian alpha pass", () => {
  it("keeps ordinary translucent actors at their opacity alongside Gaussian actors", () => {
    const gl = context();
    const before = snapshot(gl);
    const owned = ownedGaussian();
    const mesh: Pixel = [0, 0, 0, 1];
    const gaussian: Pixel = [0, 0, 0, 1];
    const afterGaussian: Pixel = [0, 0, 0, 1];
    const scene = renderer([
      actor(gl, mesh, false, [1, 0, 0, 0.5]),
      actor(gl, gaussian, true, [0, 0.5, 0, 0.5], owned.actor),
      actor(gl, afterGaussian, false, [0, 0, 1, 0.25]),
    ]);
    passFor(gl).traverse(null, scene);
    expect(mesh).toEqual([0.5, 0, 0, 1]);
    expect(gaussian).toEqual([0, 0.5, 0, 1]);
    expect(afterGaussian).toEqual([0, 0, 0.25, 1]);
    expect(scene.apply.mock.calls.map((call) => call[1])).toEqual([
      true,
      false,
    ]);
    expect(snapshot(gl)).toEqual(before);
    owned.dispose();
  });

  it("keeps actor order and independent renderer viewports", () => {
    const gl = context();
    gl.enable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(false);
    const before = snapshot(gl);
    const owned = ownedGaussian();
    const left: Pixel = [0, 0, 0, 1];
    const right: Pixel = [0, 0, 0, 1];
    const pass = passFor(gl);
    pass.traverse(
      null,
      renderer([
        actor(gl, left, false, [1, 0, 0, 0.5]),
        actor(gl, left, true, [0, 0.5, 0, 0.5], owned.actor),
      ]),
    );
    pass.traverse(null, renderer([actor(gl, right, false, [0, 0, 1, 0.5])]));
    expect(left).toEqual([0.25, 0.5, 0, 1]);
    expect(right).toEqual([0, 0, 0.5, 1]);
    expect(snapshot(gl)).toEqual(before);
    owned.dispose();
  });

  it("preserves unrelated point Gaussian mapper actors with straight-alpha shaders", () => {
    const gl = context();
    const pixel: Pixel = [0, 0, 0, 1];
    passFor(gl).traverse(
      null,
      renderer([actor(gl, pixel, true, [1, 0, 0, 0.5])]),
    );
    expect(pixel).toEqual([0.5, 0, 0, 1]);
  });

  it("restores renderer and GL state when an actor fails", () => {
    const gl = context();
    const before = snapshot(gl);
    const failing = actor(gl, [0, 0, 0, 1], false, [1, 0, 0, 0.5]);
    failing.traverse = () => {
      throw new Error("draw failed");
    };
    const scene = renderer([failing]);
    expect(() => passFor(gl).traverse(null, scene)).toThrow("draw failed");
    expect(scene.apply.mock.calls.map((call) => call[1])).toEqual([
      true,
      false,
    ]);
    expect(snapshot(gl)).toEqual(before);
  });
});
