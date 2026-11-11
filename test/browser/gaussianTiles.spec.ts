import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { chromium, type Browser, type Page } from "playwright";
import sharp from "sharp";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { gaussianTilesFixture } from "../fixtures/gaussianTiles";
import { DATASETS } from "../../examples/vtk/scene/gaussianDatasets";
import { startStaticServer, type StaticServer } from "./server";

let browser: Browser;
let server: StaticServer;
let directory = "";

beforeAll(async () => {
  directory = await mkdtemp(resolve(tmpdir(), "gaussian-tiles-browser-"));
  const fixture = gaussianTilesFixture();
  await writeFile(
    resolve(directory, "tileset.json"),
    JSON.stringify(fixture.document),
  );
  for (const [name, bytes] of Object.entries(fixture.files))
    await writeFile(resolve(directory, name), bytes);
  server = await startStaticServer({
    "/fixture": directory,
    "/": resolve("examples/vtk/dist"),
  });
  browser = await chromium.launch({ headless: true });
});

afterAll(async () => {
  await browser?.close();
  await server?.close();
  if (directory) await rm(directory, { recursive: true, force: true });
});

const coloredPixels = async (page: Page) => {
  const image = await sharp(await page.locator("canvas").screenshot())
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const counts = { red: 0, green: 0, blue: 0 };
  for (let y = 0; y < image.info.height; y++)
    for (let x = 0; x < image.info.width; x++) {
      const offset = (y * image.info.width + x) * 4;
      const r = image.data[offset]!,
        g = image.data[offset + 1]!,
        b = image.data[offset + 2]!;
      if (r > 60 && r > g * 1.5 && r > b * 1.5) counts.red++;
      if (g > 60 && g > r * 1.5 && g > b * 1.5) counts.green++;
      if (b > 60 && b > r * 1.5 && b > g * 1.5) counts.blue++;
    }
  return counts;
};

describe("Gaussian tiles in the main dataset explorer", () => {
  it("uses the standard dataset controls and keeps complete replacement coverage", async () => {
    const page = await browser.newPage({
      viewport: { width: 1000, height: 700 },
    });
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    type SceneWindow = {
      pointCloudScene: {
        camera: {
          place(value: { position: number[]; focalPoint: number[] }): void;
        };
      };
    };
    const gate = () => {
      let release!: () => void;
      let started!: () => void;
      return {
        ready: new Promise<void>((resolve) => {
          started = resolve;
        }),
        waiting: new Promise<void>((resolve) => {
          release = resolve;
        }),
        release: () => release(),
        started: () => started(),
      };
    };
    const root = gate(),
      left = gate(),
      right = gate();
    for (const [name, held] of [
      ["root", root],
      ["left", left],
      ["right", right],
    ] as const) {
      await page.route(`**/fixture/${name}.glb`, async (route) => {
        held.started();
        await held.waiting;
        await route.continue();
      });
    }
    const hasColor = async (color: "red" | "green" | "blue") =>
      expect
        .poll(async () => (await coloredPixels(page))[color])
        .toBeGreaterThan(40);
    const noColor = async (color: "red" | "green" | "blue") =>
      expect.poll(async () => (await coloredPixels(page))[color]).toBe(0);
    try {
      const fixtureUrl = `${server.origin}/fixture/tileset.json`;
      await page.goto(
        `${server.origin}/?harness=1&tiles=${encodeURIComponent(fixtureUrl)}`,
      );
      await root.ready;
      const card = page.locator("#dataset-list > .dataset-card");
      await card.waitFor();
      const sse = card.locator('[data-role="sse"]');
      await sse.focus();
      await sse.press("End");
      root.release();
      await hasColor("red");
      expect(await card.count()).toBe(1);
      expect(await page.locator("#governor-activity").count()).toBe(1);
      expect(await page.locator('a[href="/gaussian-tiles/"]').count()).toBe(0);
      expect(new URL(page.url()).searchParams.get("tiles")).toBe(fixtureUrl);

      await sse.focus();
      await sse.press("Home");
      await Promise.all([left.ready, right.ready]);
      await hasColor("red");
      left.release();
      await hasColor("red");
      await noColor("blue");
      right.release();
      await hasColor("blue");
      await hasColor("green");
      await noColor("red");

      await page.waitForFunction(() =>
        Boolean((globalThis as unknown as SceneWindow).pointCloudScene),
      );
      await page.evaluate(() =>
        (globalThis as unknown as SceneWindow).pointCloudScene.camera.place({
          position: [60, 70, 45],
          focalPoint: [0, 0, 0],
        }),
      );
      // At this distance the coarser representation is selected; reset uses
      // the explorer's dataset framing and restores the cached detail.
      await expect
        .poll(async () => (await coloredPixels(page)).red)
        .toBeGreaterThan(0);
      await page.locator("#reset-view").click();
      await hasColor("blue");
      await hasColor("green");

      const visible = card.locator('[data-role="visible"]');
      await visible.uncheck();
      await noColor("red");
      await noColor("green");
      await noColor("blue");
      await visible.check();
      await hasColor("blue");
      await hasColor("green");

      // The same URL restores a Gaussian dataset through the main page.
      await page.reload();
      await expect
        .poll(async () => {
          const colors = await coloredPixels(page);
          return colors.red + colors.blue + colors.green;
        })
        .toBeGreaterThan(40);
      expect(await page.locator("#dataset-list > .dataset-card").count()).toBe(
        1,
      );
      await page.locator('#dataset-list [data-role="remove"]').click();
      await expect
        .poll(async () => page.locator("#dataset-list > .dataset-card").count())
        .toBe(0);
      await noColor("red");
      // The hosted Gaussian preset uses the same Add dataset dialog. Its
      // placement is supplied as ECEF; these synthetic payloads stay local.
      const fixture = gaussianTilesFixture();
      const located = {
        ...fixture.document,
        root: {
          ...fixture.document.root,
          transform: [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 6378137, 0, 0, 1],
        },
      };
      await page.route("**/gaussian-city/**", async (route) => {
        const name = new URL(route.request().url()).pathname.split("/").at(-1)!;
        const bytes = fixture.files[name as keyof typeof fixture.files];
        await route.fulfill(
          name === "tileset.json"
            ? { contentType: "application/json", body: JSON.stringify(located) }
            : { contentType: "model/gltf-binary", body: Buffer.from(bytes) },
        );
      });
      await page.locator("#open-add-dataset").click();
      await page.locator("#dataset-type").selectOption("tiles");
      await page
        .locator("#dataset-preset")
        .selectOption({ label: DATASETS[0]!.label });
      await page.locator("#submit-dataset").click();
      await expect
        .poll(async () => {
          const colors = await coloredPixels(page);
          return colors.red + colors.blue + colors.green;
        })
        .toBeGreaterThan(40);
      expect(new URL(page.url()).searchParams.get("tiles")).toBe(
        DATASETS[0]!.url,
      );
      expect(await page.locator("#dataset-list").textContent()).toContain(
        "Andrii Shramko",
      );
      expect(
        await page
          .locator(
            '#dataset-list a[href="https://creativecommons.org/licenses/by/4.0/"]',
          )
          .count(),
      ).toBe(1);
      expect(errors).toEqual([]);
    } finally {
      root.release();
      left.release();
      right.release();
      await page.close();
    }
  });
});
