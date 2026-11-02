import { expect, it } from "vitest";
import { verifyPackedPackage } from "../../scripts/verifyPackage.mjs";

it("consumes the packed public API with isolated bundles and real mixed-format rendering", async () => {
  const result = await verifyPackedPackage();
  expect(result.exports.length).toBeGreaterThanOrEqual(8);
  expect(result.browser?.refined.mesh.renderer.drawnTileIds).toEqual([
    "root/0",
    "root/1",
  ]);
}, 120_000);
