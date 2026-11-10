import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { chromium, type Browser } from "playwright";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { gaussianGlb } from "../fixtures/gaussianGlb";
import { startStaticServer, type StaticServer } from "./server";

let browser: Browser;
let server: StaticServer;
let directory = "";

beforeAll(async () => {
  directory = await mkdtemp(resolve(tmpdir(), "gaussian-workers-"));
  await writeFile(
    resolve(directory, "index.html"),
    '<!doctype html><title>Gaussian workers</title><script type="module">import * as api from "/dist/tiles3d.js"; globalThis.gaussianApi = api;</script>',
  );
  await writeFile(
    resolve(directory, "standard.glb"),
    new Uint8Array(gaussianGlb({ degree: 3, mixed: true })),
  );
  await writeFile(
    resolve(directory, "unsupported.glb"),
    new Uint8Array(
      gaussianGlb({ extension: { projection: "unsupported-projection" } }),
    ),
  );
  server = await startStaticServer({
    "/dist": resolve("dist"),
    "/fixture": directory,
  });
  browser = await chromium.launch({ headless: true });
});

afterAll(async () => {
  await browser?.close();
  await server?.close();
  if (directory) await rm(directory, { recursive: true, force: true });
});

describe("packaged Gaussian workers", () => {
  it("decodes mixed standard GLB through the classic pool, preserving harmonics and precision", async () => {
    const page = await browser.newPage();
    try {
      await page.goto(`${server.origin}/fixture/index.html`);
      await page.waitForFunction(() => "gaussianApi" in globalThis);
      const result = await page.evaluate(async (origin) => {
        const api = (
          globalThis as unknown as {
            gaussianApi: typeof import("../../src/tiles3d/index");
          }
        ).gaussianApi;
        const pool = new api.DecodeWorkerPool({
          size: 1,
          workerUrl: `${origin}/dist/tiles3dDecodeWorker.classic.js`,
        });
        try {
          const response = await fetch(`${origin}/fixture/standard.glb`);
          const content = await response.arrayBuffer();
          const decoded = await pool.decode({
            content,
            contentUrl: `${origin}/fixture/standard.glb`,
            revision: "synthetic-unlocated",
            accumulatedTransform: [
              1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 6000000.125, 20, 30, 1,
            ],
            tilesetToScene: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            textureCapabilities: {
              capabilityKey: "rgba",
              compressedFormats: [],
            },
          }).promise;
          const gaussian = decoded.primitives.find(
            (primitive) => primitive.kind === "gaussian",
          );
          if (!gaussian || gaussian.kind !== "gaussian")
            throw new Error("Gaussian primitive missing");
          return {
            kinds: decoded.primitives.map((primitive) => primitive.kind),
            positions: Array.from(gaussian.positions),
            rotations: Array.from(gaussian.rotations.slice(0, 4)),
            degree: gaussian.harmonicDegree,
            coefficientValues: gaussian.sphericalHarmonics.length,
            unclampedColorCoefficient: gaussian.sphericalHarmonics[2],
            transformIsFloat64: gaussian.modelToScene instanceof Float64Array,
            translation: gaussian.modelToScene[12],
            profile: gaussian.profile,
            bytes: decoded.byteEstimate.geometry,
          };
        } finally {
          pool.dispose();
        }
      }, server.origin);
      expect(result).toMatchObject({
        kinds: ["gaussian", "mesh"],
        positions: [0.125, 0.25, 0.5, 1, 2, 3, -1, -2, -3],
        rotations: [0, 0, 0, 1],
        degree: 3,
        coefficientValues: 3 * 16 * 3,
        unclampedColorCoefficient: 2,
        transformIsFloat64: true,
        translation: 6000000.125,
        profile: {
          sortingMethod: "cameraDistance",
          colorSpace: "srgb_rec709_display",
        },
      });
      expect(result.bytes).toBeGreaterThan(3 * 16 * 3 * 4);
    } finally {
      await page.close();
    }
  });

  it("retains named unsupported-profile errors across the worker boundary", async () => {
    const page = await browser.newPage();
    try {
      await page.goto(`${server.origin}/fixture/index.html`);
      await page.waitForFunction(() => "gaussianApi" in globalThis);
      const result = await page.evaluate(async (origin) => {
        const api = (
          globalThis as unknown as {
            gaussianApi: typeof import("../../src/tiles3d/index");
          }
        ).gaussianApi;
        const pool = new api.DecodeWorkerPool({
          size: 1,
          workerUrl: `${origin}/dist/tiles3dDecodeWorker.classic.js`,
        });
        try {
          const content = await (
            await fetch(`${origin}/fixture/unsupported.glb`)
          ).arrayBuffer();
          try {
            await pool.decode({
              content,
              contentUrl: `${origin}/fixture/unsupported.glb`,
              revision: "synthetic-unlocated",
              accumulatedTransform: [
                1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
              ],
              tilesetToScene: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
              textureCapabilities: {
                capabilityKey: "rgba",
                compressedFormats: [],
              },
            }).promise;
            throw new Error("unsupported profile was accepted");
          } catch (error) {
            if (!(error instanceof Error)) throw error;
            return { name: error.name, message: error.message };
          }
        } finally {
          pool.dispose();
        }
      }, server.origin);
      expect(result.name).toBe("TileUnsupportedGaussianProfileError");
      expect(result.message).toContain("projection");
      expect(result.message).toContain("unsupported-projection");
    } finally {
      await page.close();
    }
  });

  it("globally sorts overlapping ranges in the module worker and shuts it down", async () => {
    const page = await browser.newPage();
    try {
      await page.goto(`${server.origin}/fixture/index.html`);
      await page.waitForFunction(() => "gaussianApi" in globalThis);
      const result = await page.evaluate(async (origin) => {
        const api = (
          globalThis as unknown as {
            gaussianApi: typeof import("../../src/tiles3d/index");
          }
        ).gaussianApi;
        const worker = new Worker(`${origin}/dist/gaussianSortWorker.js`, {
          type: "module",
        });
        const orders: number[][] = [];
        const errors: string[] = [];
        const controller = api.createGaussianSortController({
          backend: api.createGaussianSortWorkerBackend(worker),
          onOrder: (presentation) =>
            orders.push(Array.from(presentation.order)),
          onError: (error) => errors.push(String(error)),
        });
        try {
          const modelToScene = new Float64Array([
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 6000000, 0, 0, 1,
          ]);
          controller.setScene([
            {
              id: "tileset-a/root",
              generation: 1,
              positions: new Float32Array([100, 0, -1, 0, 0, -10]),
              modelToScene,
            },
            {
              id: "tileset-b/root",
              generation: 2,
              positions: new Float32Array([0, 0, -20]),
              modelToScene,
            },
          ]);
          controller.request([6000000, 0, 0]);
          await controller.whenIdle();
          controller.request([6000100, 0, 0]);
          await controller.whenIdle();
          controller.request([6000100, 0, 0]);
          await controller.whenIdle();
          return { orders, errors };
        } finally {
          controller.dispose();
        }
      }, server.origin);
      expect(result).toEqual({
        orders: [
          [0, 2, 1],
          [2, 1, 0],
        ],
        errors: [],
      });
    } finally {
      await page.close();
    }
  });
});
