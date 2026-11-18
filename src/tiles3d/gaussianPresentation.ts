import type { PreparedPointTextures } from "@kitware/vtk.js/Rendering/Core/PointGaussianMapper";
import {
  createPointTextureBank,
  type PointTextureBank,
} from "@kitware/vtk.js/Rendering/OpenGL/PointGaussianMapper/dataTextures";
import type { CameraView } from "../core/camera";
import { gaussianModelCamera, writeGaussianColor } from "./gaussianGeometry";
import type {
  GaussianAttributes,
  GaussianPackedRange,
  GaussianTupleRange,
} from "./gaussianPacking";

export type GaussianGpuContext = Pick<
  PreparedPointTextures,
  "window" | "context" | "generation"
>;

export type GaussianGpuAtlas = ReturnType<typeof createGaussianGpuAtlas>;

/** Geometry occupies stable ranges. Two small sets of presentation textures
 * keep writes for the next view away from the current draw's colors and order.
 */
export const createGaussianGpuAtlas = (
  context: GaussianGpuContext,
  positions: Float32Array,
) => {
  const geometry = createPointTextureBank(context.window);
  const makePresentation = () => ({
    textures: createPointTextureBank(context.window),
    staticColors: new Set<number>(),
  });
  const presentations = [makePresentation(), makePresentation()] as const;
  let active = 0;
  const geometryRanges = new Set<number>();
  return {
    positions,
    geometry,
    geometryRanges,
    next: () => presentations[1 - active]!,
    activate() {
      active = 1 - active;
    },
    retireExcept(live: ReadonlySet<number>) {
      for (const ranges of [
        geometryRanges,
        ...presentations.map((draw) => draw.staticColors),
      ])
        for (const revision of ranges)
          if (!live.has(revision)) ranges.delete(revision);
    },
    prepared(maximumId: number): PreparedPointTextures {
      const draw = presentations[1 - active]!.textures;
      return {
        ...context,
        maximumId,
        bind(program) {
          geometry.bind(program);
          draw.bind(program);
        },
        unbind() {
          geometry.unbind();
          draw.unbind();
        },
      };
    },
    release() {
      geometry.release();
      for (const { textures } of presentations) textures.release();
    },
  };
};

const contiguousTupleRanges = (
  primitives: readonly GaussianTupleRange[],
): [number, number][] => {
  const ranges: [number, number][] = [];
  for (const { offset, count } of [...primitives].sort(
    (a, b) => a.offset - b.offset,
  )) {
    const previous = ranges.at(-1);
    if (previous && previous[0] + previous[1] === offset) previous[1] += count;
    else ranges.push([offset, count]);
  }
  return ranges;
};

/** Translate the sort worker's dense permutation to stable atlas slots. */
export function* stageGaussianOrder(
  ranges: readonly GaussianTupleRange[],
  sorted: Uint32Array,
  order: Uint32Array,
  drawToStorage: Uint32Array,
  mappingStep: number,
  orderStep: number,
) {
  let drawnOffset = 0;
  for (const { offset, count } of ranges) {
    const base = drawnOffset;
    drawnOffset += count;
    for (let first = 0; first < count; first += mappingStep) {
      const end = Math.min(count, first + mappingStep);
      yield {
        bytes: (end - first) * 4,
        run() {
          for (let index = first; index < end; index++)
            drawToStorage[base + index] = offset + index;
        },
      };
    }
  }
  let maximumId = -1;
  for (let first = 0; first < order.length; first += orderStep) {
    const end = Math.min(order.length, first + orderStep);
    yield {
      bytes: (end - first) * 8,
      run() {
        for (let index = first; index < end; index++) {
          const id = sorted[index]!;
          const offset = drawToStorage[id];
          if (offset === undefined || offset === 0xffffffff)
            throw new Error("Gaussian sort returned an invalid draw index");
          drawToStorage[id] = 0xffffffff;
          order[index] = offset;
          maximumId = Math.max(maximumId, offset);
        }
      },
    };
  }
  return maximumId;
}

function* uploadRanges(
  bank: PointTextureBank,
  name: string,
  values: Float32Array | Uint32Array,
  components: number,
  capacity: number,
  ranges: readonly [number, number][],
  tuplesPerStep: (bytes: number) => number,
  onUpload: (bytes: number) => void,
) {
  const bytesPerTuple = components * values.BYTES_PER_ELEMENT;
  const step = tuplesPerStep(bytesPerTuple);
  for (const [offset, count] of ranges)
    for (let first = 0; first < count; first += step) {
      const size = Math.min(step, count - first);
      const bytes = size * bytesPerTuple;
      yield {
        bytes,
        run() {
          bank.write(name, values, components, capacity, offset + first, size);
          onUpload(bytes);
        },
      };
    }
}

type GaussianAttributePlan<Range extends GaussianPackedRange> = {
  attributes: GaussianAttributes;
  ranges: readonly Range[];
  gpu: GaussianGpuAtlas | null;
  camera: CameraView["position"];
  recolor: (range: Range) => boolean;
  order: Uint32Array;
  colorTuplesPerStep: number;
  uploadTuplesPerStep: (bytes: number) => number;
  onAttributeUpload: (bytes: number) => void;
  onOrderUpload: (bytes: number) => void;
};

/** Stage geometry into unaccepted ranges and colors and order into the next draw bank. */
export function* stageGaussianAttributes<Range extends GaussianPackedRange>({
  attributes,
  ranges,
  gpu,
  camera,
  recolor,
  order,
  colorTuplesPerStep,
  uploadTuplesPerStep,
  onAttributeUpload,
  onOrderUpload,
}: GaussianAttributePlan<Range>) {
  const capacity = attributes.positions.length / 3;
  const draw = gpu?.next();
  if (gpu) {
    const dirtyGeometry = ranges.filter(
      ({ allocationRevision }) => !gpu.geometryRanges.has(allocationRevision),
    );
    const updates = contiguousTupleRanges(dirtyGeometry);
    for (const [name, values, components] of [
      ["pointPositions", attributes.positions, 3],
      ["pointRadius", attributes.scales, 3],
      ["pointRotation", attributes.rotations, 4],
    ] as const)
      yield* uploadRanges(
        gpu.geometry,
        name,
        values,
        components,
        capacity,
        updates,
        uploadTuplesPerStep,
        onAttributeUpload,
      );
    for (const { allocationRevision } of dirtyGeometry)
      gpu.geometryRanges.add(allocationRevision);
  }

  const colorUpdates: [number, number][] = [];
  const dirtyColors: GaussianPackedRange[] = [];
  for (const range of ranges) {
    const { allocationRevision, offset, count, primitive } = range;
    const harmonic = primitive.harmonicDegree > 0;
    const colorChanged = harmonic && (!!draw || recolor(range));
    if (!colorChanged && (!draw || draw.staticColors.has(allocationRevision)))
      continue;
    const modelCamera = harmonic
      ? gaussianModelCamera(range.matrix, camera)
      : null;
    if (modelCamera) {
      for (let first = 0; first < count; first += colorTuplesPerStep) {
        const size = Math.min(colorTuplesPerStep, count - first);
        yield {
          bytes: size * 16,
          run() {
            for (let index = first; index < first + size; index++)
              writeGaussianColor(
                primitive,
                index,
                modelCamera,
                attributes.colors,
                (offset + index) * 4,
              );
          },
        };
      }
    }
    if (draw) dirtyColors.push(range);
    if (colorChanged) colorUpdates.push([offset, count]);
  }
  if (draw) {
    yield* uploadRanges(
      draw.textures,
      "pointColors",
      attributes.colors,
      4,
      capacity,
      contiguousTupleRanges(dirtyColors),
      uploadTuplesPerStep,
      onAttributeUpload,
    );
    for (const { primitive, allocationRevision } of dirtyColors)
      if (primitive.harmonicDegree === 0)
        draw.staticColors.add(allocationRevision);
    yield* uploadRanges(
      draw.textures,
      "pointOrder",
      order,
      1,
      capacity,
      [[0, order.length]],
      uploadTuplesPerStep,
      onOrderUpload,
    );
  }
  return colorUpdates;
}
