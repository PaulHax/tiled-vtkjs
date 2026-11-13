import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const stub = (name: string): string =>
  fileURLToPath(new URL(`./test/stubs/${name}.ts`, import.meta.url));

export default defineConfig({
  build: {
    lib: {
      entry: {
        index: "src/index.ts",
        core: "src/core/index.ts",
        points: "src/points/index.ts",
        "points-vtk": "src/points/vtk.ts",
        copcWorker: "src/points/copcWorker.ts",
        tiles3d: "src/tiles3d/index.ts",
        "tiles3d-vtk": "src/tiles3d/vtk.ts",
        gaussianSortWorker: "src/tiles3d/gaussianSort.worker.ts",
      },
      formats: ["es"],
    },
    rollupOptions: {
      // Never bundle the vtk.js peer or the copc reader; consumers resolve them.
      external: [/^@kitware\/vtk\.js/, "copc"],
    },
    sourcemap: true,
  },
  test: {
    // The browser checks need a built example, a served fixture, and Chromium.
    // They have their own config and command so this suite stays a fast,
    // dependency-free unit run.
    exclude: ["**/node_modules/**", "**/dist/**", "test/browser/**"],
    // The adapter's vtk.js deep imports resolve to recording stubs so the
    // suite records adapter calls; the packed consumer browser gate uses VTK.
    alias: {
      "@kitware/vtk.js/Rendering/Core/Actor": stub("vtkActor"),
      "@kitware/vtk.js/Rendering/Core/PointSpriteMapper": stub(
        "vtkPointSpriteMapper",
      ),
      "@kitware/vtk.js/Common/DataModel/PolyData": stub("vtkPolyData"),
      "@kitware/vtk.js/Common/Core/DataArray": stub("vtkDataArray"),
      "@kitware/vtk.js/Rendering/Core/Mapper": stub("vtkMapper"),
      "@kitware/vtk.js/Rendering/Core/Texture": stub("vtkTexture"),
    },
  },
});
