import { gaussianGlb } from "./gaussianGlb";

const tileGlb = (rgb: readonly number[], shift: number): Uint8Array =>
  new Uint8Array(
    gaussianGlb({
      mutate(_json, arrays) {
        const extension = "KHR_gaussian_splatting:";
        arrays
          .get("POSITION")!
          .set([-0.8 + shift, 0, 0, shift, 0, 0.1, 0.8 + shift, 0, 0]);
        arrays.get(`${extension}SCALE`)!.fill(0.32);
        arrays.get(`${extension}OPACITY`)!.fill(0.7);
        const harmonics = arrays.get(`${extension}SH_DEGREE_0_COEF_0`)!;
        for (let point = 0; point < 3; point++)
          for (let channel = 0; channel < 3; channel++) {
            harmonics[point * 3 + channel] =
              (rgb[channel]! - 0.5) / 0.2820947917738781;
          }
      },
    }),
  );

/** Tiny synthetic replacement hierarchy for deterministic browser correctness. */
export const gaussianTilesFixture = () => {
  const box = [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2];
  const document = {
    asset: {
      version: "1.1",
      generator: "Synthetic Gaussian replacement fixture",
    },
    geometricError: 0.3,
    root: {
      boundingVolume: { box },
      geometricError: 0.3,
      refine: "REPLACE",
      content: { uri: "root.glb" },
      children: ["left", "right"].map((id) => ({
        boundingVolume: { box },
        geometricError: 0,
        content: { uri: `${id}.glb` },
      })),
    },
  };
  return {
    document,
    files: {
      "root.glb": tileGlb([1, 0.05, 0.03], 0),
      "left.glb": tileGlb([0.03, 0.15, 1], -0.5),
      "right.glb": tileGlb([0.03, 1, 0.12], 0.5),
    },
  };
};
