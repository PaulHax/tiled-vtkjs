import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { chromium, type Browser, type Page } from "playwright";
import sharp from "sharp";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { gaussianTilesFixture } from "../fixtures/gaussianTiles";
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

type DemoStats = {
  fetchedTiles: number;
  totalTiles: number;
  renderer: {
    displayedSplats: number;
    pendingSort: boolean;
    sortCount: number;
    attributeUploadBytes: number;
    residentBytes: number;
  } | null;
  member: {
    renderer: { drawnTiles: number; drawnTileIds: string[] };
    queue: { workPending: boolean } | null;
  } | null;
};

const stats = (page: Page): Promise<DemoStats> =>
  page.evaluate(() =>
    (
      globalThis as unknown as { gaussianTiles: { stats(): DemoStats } }
    ).gaussianTiles.stats(),
  );

const coloredPixels = async (page: Page) => {
  const image = await sharp(await page.locator("canvas").screenshot())
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const counts = { red: 0, green: 0, blue: 0 };
  for (let y = 0; y < image.info.height; y++)
    for (let x = 380; x < image.info.width; x++) {
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

describe("Gaussian 3D Tiles example", () => {
  it("renders a coarse view, retains its parent while children load, and swaps the complete replacement", async () => {
    const page = await browser.newPage({
      viewport: { width: 1000, height: 700 },
    });
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    let leftRelease!: () => void;
    let rightRelease!: () => void;
    let leftRequested!: () => void;
    let rightRequested!: () => void;
    const leftGate = new Promise<void>((resolve) => {
      leftRelease = resolve;
    });
    const rightGate = new Promise<void>((resolve) => {
      rightRelease = resolve;
    });
    const leftStarted = new Promise<void>((resolve) => {
      leftRequested = resolve;
    });
    const rightStarted = new Promise<void>((resolve) => {
      rightRequested = resolve;
    });
    await page.route("**/fixture/left.glb", async (route) => {
      leftRequested();
      await leftGate;
      await route.continue();
    });
    await page.route("**/fixture/right.glb", async (route) => {
      rightRequested();
      await rightGate;
      await route.continue();
    });
    try {
      const url = `${server.origin}/gaussian-tiles/index.html?test=1&sse=96&tileset=${encodeURIComponent(`${server.origin}/fixture/tileset.json`)}`;
      await page.goto(url);
      await page.locator('#status[data-state="ready"]').waitFor();
      expect(await stats(page)).toMatchObject({
        totalTiles: 3,
        fetchedTiles: 1,
        renderer: { displayedSplats: 3 },
        member: { renderer: { drawnTileIds: ["root"] } },
      });
      expect((await coloredPixels(page)).red).toBeGreaterThan(40);

      await page.locator("#quality").focus();
      await page.locator("#quality").press("Home");
      await Promise.all([leftStarted, rightStarted]);
      expect((await stats(page)).member?.renderer.drawnTileIds).toEqual([
        "root",
      ]);
      leftRelease();
      await page.waitForFunction(
        () =>
          (
            globalThis as unknown as { gaussianTiles: { stats(): DemoStats } }
          ).gaussianTiles.stats().fetchedTiles === 2,
      );
      expect((await stats(page)).renderer?.displayedSplats).toBe(3);
      rightRelease();
      await page.waitForFunction(
        () =>
          (
            globalThis as unknown as { gaussianTiles: { stats(): DemoStats } }
          ).gaussianTiles.stats().renderer?.displayedSplats === 6,
      );
      expect((await stats(page)).member?.renderer.drawnTileIds).toEqual([
        "root/0",
        "root/1",
      ]);
      const refined = await coloredPixels(page);
      expect(refined.red).toBe(0);
      expect(refined.blue).toBeGreaterThan(40);
      expect(refined.green).toBeGreaterThan(40);
      expect(errors).toEqual([]);

      await page.evaluate(() =>
        (
          globalThis as unknown as {
            gaussianTiles: {
              moveCamera(position: number[], target: number[]): void;
            };
          }
        ).gaussianTiles.moveCamera([6000, 7000, 4500], [0, 0, 0]),
      );
      await page.waitForFunction(
        () =>
          (
            globalThis as unknown as { gaussianTiles: { stats(): DemoStats } }
          ).gaussianTiles.stats().renderer?.displayedSplats === 3,
      );
      expect((await stats(page)).member?.renderer.drawnTileIds).toEqual([
        "root",
      ]);
      await page.locator("#reset-view").click();
      await page.waitForFunction(
        () =>
          (
            globalThis as unknown as { gaussianTiles: { stats(): DemoStats } }
          ).gaussianTiles.stats().renderer?.displayedSplats === 6,
      );

      await page.locator("#visible").uncheck();
      await page.waitForFunction(
        () =>
          (
            globalThis as unknown as { gaussianTiles: { stats(): DemoStats } }
          ).gaussianTiles.stats().renderer?.displayedSplats === 0,
      );
      expect(await coloredPixels(page)).toEqual({ red: 0, green: 0, blue: 0 });
      await page.locator("#visible").check();
      await page.waitForFunction(
        () =>
          (
            globalThis as unknown as { gaussianTiles: { stats(): DemoStats } }
          ).gaussianTiles.stats().renderer?.displayedSplats === 6,
      );
      await page.evaluate(() =>
        (
          globalThis as unknown as { gaussianTiles: { dispose(): void } }
        ).gaussianTiles.dispose(),
      );
      expect(await page.locator('#status[data-state="disposed"]').count()).toBe(
        1,
      );
    } finally {
      leftRelease();
      rightRelease();
      await page.close();
    }
  });
});
