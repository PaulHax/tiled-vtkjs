import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const modules = ["core", "points", "tiles3d"];
const sourceFiles = async (directory) =>
  (
    await Promise.all(
      (
        await readdir(directory, { withFileTypes: true })
      ).map((entry) => {
        const path = resolve(directory, entry.name);
        return entry.isDirectory()
          ? sourceFiles(path)
          : path.endsWith(".ts") && !path.endsWith(".test.ts")
            ? [path]
            : [];
      }),
    )
  ).flat();

for (const module of modules) {
  const paths = await sourceFiles(resolve(root, "src", module));
  if (module === "core") paths.push(resolve(root, "src/index.ts"));
  for (const path of paths) {
    const source = (await readFile(path, "utf8")).replace(
      /\/\*[\s\S]*?\*\/|\/\/[^\n]*/g,
      "",
    );
    const check = (specifier) => {
      const location = relative(root, path);
      if (specifier.startsWith(".")) {
        const target = relative(
          resolve(root, "src"),
          resolve(dirname(path), specifier),
        );
        const owner = target.split("/")[0];
        assert(
          owner === module || (module !== "core" && owner === "core"),
          `${location} crosses its module boundary: ${specifier}`,
        );
      } else {
        assert(module !== "core", `${location} imports ${specifier}`);
        if (specifier.startsWith("@kitware/vtk.js")) {
          assert(
            /\/(rendererAdapter|meshAdapter)\.ts$/.test(path),
            `${location} imports VTK outside its adapter`,
          );
        }
      }
    };
    // Match static imports/re-exports and literal dynamic/type imports,
    // including type-only references that disappear from the built graph.
    const imports = source.matchAll(
      /\b(?:import|export)\s+(?:[^;"']*?\s+from\s+)?["']([^"']+)["']|\bimport\s*\(\s*["']([^"']+)["']/g,
    );
    for (const match of imports) check(match[1] ?? match[2]);
  }
}
console.log("core, points and tiles3d source boundaries verified");
