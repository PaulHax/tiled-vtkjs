import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { chromium, type Browser, type Page } from "playwright";
import sharp from "sharp";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { gaussianTilesFixture } from "../fixtures/gaussianTiles";
import { DATASETS } from "../../examples/vtk/scene/gaussianDatasets";
import type { Tiles3dMemberStats } from "../../src/tiles3d/memberTypes";
import type { SceneWindow } from "./sceneHarness";
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
      const first = await Promise.race([
        left.ready.then(() => left),
        right.ready.then(() => right),
      ]);
      const second = first === left ? right : left;
      await hasColor("red");
      first.release();
      await second.ready;
      await hasColor("red");
      await noColor("blue");
      await noColor("green");
      second.release();
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

const openMotionFixture = async (page: Page) => {
  const fixtureUrl = server.origin + "/fixture/tileset.json";
  await page.goto(
    server.origin + "/?harness=1&tiles=" + encodeURIComponent(fixtureUrl),
  );
  await page.waitForFunction(
    () =>
      (window as unknown as SceneWindow).pointCloudScene?.stats().members
        .length,
  );
};

const readMotionView = (page: Page) =>
  page.evaluate(() => {
    const scene = (window as unknown as SceneWindow).pointCloudScene;
    const { coordinator, members, paints } = scene.stats();
    const tiles = members[0]?.stats as Tiles3dMemberStats | undefined;
    if (tiles?.kind !== "tiles3d")
      throw new Error("Expected a Gaussian tileset");
    return {
      drawn: [...tiles.renderer.drawnTileIds].sort(),
      fetching: tiles.queue?.active ?? 0,
      governor: coordinator.governor,
      position: scene.camera.read().position,
      paints,
    };
  });

const viewerCenter = async (page: Page) => {
  const box = await page.locator("#viewer").boundingBox();
  if (!box) throw new Error("Expected a visible viewer");
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
};

const highDetail = ["root/0", "root/1"];
const expectHighDetailPixels = async (page: Page) => {
  const pixels = await coloredPixels(page);
  expect(pixels.red).toBe(0);
  expect(pixels.green).toBeGreaterThan(40);
  expect(pixels.blue).toBeGreaterThan(40);
};

it("keeps Gaussian coverage through a second gesture while finer tiles are in flight", async () => {
  const page = await browser.newPage({
    viewport: { width: 1000, height: 700 },
  });
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route(/\/fixture\/(left|right)\.glb$/, async (route) => {
    await held;
    await route.continue();
  });
  const frame = () =>
    page.evaluate(
      () =>
        new Promise<void>((resolve) => requestAnimationFrame(() => resolve())),
    );
  try {
    await openMotionFixture(page);
    const sse = page.locator('[data-role="sse"]');
    await sse.focus();
    await sse.press("End");
    await expect
      .poll(async () => (await coloredPixels(page)).red)
      .toBeGreaterThan(40);
    await page.evaluate(() => {
      const scene = (window as unknown as SceneWindow).pointCloudScene;
      scene.camera.place({ position: [6, 7, 4.5], focalPoint: [0, 0, 0] });
      scene.settings.setScreenSpaceErrorPx(scene.datasets()[0]!.id, 40);
    });
    const { x, y } = await viewerCenter(page);
    await page.mouse.move(x, y);
    for (let i = 0; i < 6; i++) {
      await page.mouse.wheel(0, -100);
      await frame();
    }
    await expect
      .poll(async () => (await readMotionView(page)).fetching)
      .toBeGreaterThan(0);
    await expect
      .poll(async () => (await readMotionView(page)).governor.regime)
      .toBe("stationary");
    const firstStop = await readMotionView(page);
    await page.mouse.down();
    for (let i = 1; i <= 18; i++) {
      await page.mouse.move(x + 2 * i, y + i / 2);
      await frame();
      expect((await readMotionView(page)).drawn).toEqual(["root"]);
    }
    const moved = await readMotionView(page);
    expect(moved.governor.regime).toBe("interaction");
    expect(moved.fetching).toBeGreaterThan(0);
    expect(moved.paints).toBeGreaterThan(firstStop.paints);
    expect(moved.position).not.toEqual(firstStop.position);
    expect((await coloredPixels(page)).red).toBeGreaterThan(40);
    await page.mouse.up();
    await expect
      .poll(async () => (await readMotionView(page)).governor.regime)
      .toBe("stationary");
    const stopped = await readMotionView(page);
    release();
    await expect
      .poll(async () => (await readMotionView(page)).drawn)
      .toEqual(highDetail);
    expect((await readMotionView(page)).position).toEqual(stopped.position);
    await expectHighDetailPixels(page);
  } finally {
    release();
    await page.close();
  }
});

it("restores visible Gaussian detail after a costly drag and one delayed idle paint", async () => {
  const page = await browser.newPage({
    viewport: { width: 1000, height: 700 },
  });
  try {
    await page.clock.install();
    await openMotionFixture(page);
    await expect
      .poll(async () => (await readMotionView(page)).drawn)
      .toEqual(highDetail);
    // In this view the root no longer meets the requested error at full
    // quality, but does meet it after the frame budget has cut quality.
    await page.evaluate(() =>
      (window as unknown as SceneWindow).pointCloudScene.camera.place({
        position: [0, -21, 5],
        focalPoint: [0, 0, 0],
      }),
    );
    await expect
      .poll(async () => (await readMotionView(page)).governor.regime)
      .toBe("stationary");
    await expectHighDetailPixels(page);
    const now = await page.evaluate(() => Date.now());
    await page.clock.pauseAt(new Date(now + 1000));
    // Seed the previous stationary view with expensive presentations. A host
    // such as a map can repaint independently of the tiles' own demand.
    for (let frame = 0; frame < 120; frame++) {
      await page.evaluate(() =>
        (window as unknown as SceneWindow).pointCloudScene.render(),
      );
      await page.clock.fastForward(64);
      if ((await readMotionView(page)).drawn.length === 1) break;
    }
    expect((await readMotionView(page)).drawn).toEqual(["root"]);
    const { x, y } = await viewerCenter(page);
    await page.mouse.move(x, y);
    await page.mouse.down();
    const start = await readMotionView(page);
    // Model a cheaper draw once the parent replaces its children. These
    // controlled presentation intervals prescribe load for correctness.
    for (let frame = 0; frame < 40; frame++) {
      await page.mouse.move(x + 15 * Math.sin(frame / 2), y);
      const previous = await readMotionView(page);
      await page.clock.fastForward(previous.drawn.length === 2 ? 75 : 32);
      const { drawn } = await readMotionView(page);
      expect([["root"], highDetail]).toContainEqual(drawn);
    }
    await page.mouse.up();
    const stopped = await readMotionView(page);
    expect(stopped.paints).toBeGreaterThan(start.paints);
    expect(stopped.position).not.toEqual(start.position);
    expect(stopped.drawn, JSON.stringify(stopped.governor)).toEqual(["root"]);

    // Let only work requested by the viewer paint. The outlier comes after
    // streaming and the camera have settled so it enters the idle sample.
    let outlier = false;
    let view = stopped;
    for (let frame = 0; frame < 300; frame++) {
      const g = view.governor;
      const inject: boolean =
        !outlier &&
        g.regime === "stationary" &&
        g.samples > 0 &&
        g.capacitySamples.lastEligible === true;
      await page.clock.fastForward(inject ? 50 : 16);
      outlier ||= inject;
      view = await readMotionView(page);
      if (outlier && view.drawn.length === 2) break;
    }
    expect(outlier).toBe(true);
    expect(view.drawn, JSON.stringify(view.governor.lastAdjustment)).toEqual(
      highDetail,
    );
    expect(view.position).toEqual(stopped.position);
    await expectHighDetailPixels(page);
  } finally {
    await page.close();
  }
});
