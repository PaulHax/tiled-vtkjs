import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import type { Page, Request } from "playwright";
import sharp from "sharp";
import { afterAll, expect, it } from "vitest";

import type { PointCloudMemberStats } from "../../src/points/pointCloudMember";
import type { Tiles3dMemberStats } from "../../src/tiles3d/memberTypes";
import { writeMixedMotionFixture } from "../fixtures/mixedMotion";
import {
  browserFor,
  closeBrowsers,
  collectPageFailures,
  EXAMPLE_DIST,
  FIXTURES,
} from "./browserSession";
import type { SceneWindow } from "./sceneHarness";
import { startStaticServer, type StaticServer } from "./server";

type View = {
  paint: number;
  camera: number[];
  splats: string[];
  triangles: number;
  points: number;
  activeRequests: number;
  regime: "stationary" | "interaction";
  pending: boolean;
  needsFrame: boolean;
};

type MotionWindow = SceneWindow & {
  readMotionView(): View;
  motionPaints: View[];
};

const readView = (page: Page) =>
  page.evaluate(() => (window as unknown as MotionWindow).readMotionView());

// Advance browser time without requesting renders or making simulated progress
// depend on the test runner's wall-clock polling cadence.
const advanceUntil = async (
  page: Page,
  frameLimit: number,
  predicate: (view: View) => boolean,
): Promise<View> => {
  let view = await readView(page);
  for (let frame = 0; frame < frameLimit && !predicate(view); frame++) {
    await page.clock.runFor(16);
    view = await readView(page);
  }
  return view;
};

const childPath = /^\/fixture\/splats\/(left|right)\.glb$/;

const installViewReader = (page: Page) =>
  page.evaluate(() => {
    const w = window as unknown as MotionWindow;
    const scene = w.pointCloudScene;
    w.readMotionView = () => {
      const { coordinator, members, paints } = scene.stats();
      const [points, mesh, splats] = members;
      const pointStats = points!.stats as unknown as PointCloudMemberStats;
      const splatStats = splats?.stats as unknown as
        | Tiles3dMemberStats
        | undefined;
      const meshStats = mesh!.stats as unknown as Tiles3dMemberStats;
      return {
        paint: paints,
        camera: [...scene.camera.read().position],
        splats: [...(splatStats?.renderer.drawnTileIds ?? [])].sort(),
        triangles: meshStats.renderer.drawnTriangles,
        points: pointStats.renderer.drawnPoints,
        activeRequests: splatStats?.queue?.active ?? 0,
        regime: coordinator.governor.regime,
        pending: coordinator.governor.activity.workPending,
        needsFrame: coordinator.governor.needsFrame,
      };
    };
  });

const pixels = async (page: Page) => {
  const { data, info } = await sharp(await page.locator("canvas").screenshot())
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const counts = { red: 0, blue: 0, green: 0, cyan: 0 };
  for (let i = 0; i < info.width * info.height; i++) {
    const r = data[i * 4]!,
      g = data[i * 4 + 1]!,
      b = data[i * 4 + 2]!;
    if (g > 100 && b > 100 && Math.min(g, b) > r * 1.6) counts.cyan++;
    // Ignore most COPC colors when counting the centered Gaussian fixture.
    const x = (i % info.width) / info.width;
    if (x < 0.25 || x > 0.75) continue;
    if (r > 100 && r > Math.max(g, b) * 1.6) counts.red++;
    if (g > 100 && g > Math.max(r, b) * 1.6) counts.green++;
    if (b > 100 && b > Math.max(r, g) * 1.6) counts.blue++;
  }
  return counts;
};

afterAll(closeBrowsers);

it("keeps all three members drawn while moving and wakes on held Gaussian refinement", async () => {
  const directory = await mkdtemp(resolve(tmpdir(), "mixed-motion-"));
  let openTileset!: () => void;
  const metadataHeld = new Promise<void>((done) => {
    openTileset = done;
  });
  let release!: () => void;
  const held = new Promise<void>((done) => {
    release = done;
  });
  let server: StaticServer | undefined;
  let page: Page | undefined;
  try {
    await writeMixedMotionFixture(
      directory,
      resolve(FIXTURES, "fixture.copc.laz"),
    );
    const localServer = await startStaticServer({
      "/fixture": directory,
      "/": EXAMPLE_DIST,
    });
    server = localServer;
    const browser = await browserFor("software");
    const tab = await browser.newPage({
      viewport: { width: 1000, height: 700 },
    });
    page = tab;
    const errors = collectPageFailures(tab);
    const external: string[] = [];
    const heldChildren = new Set<Request>();
    tab.on("requestfailed", (request) => heldChildren.delete(request));
    await tab.route("**/*", async (route) => {
      const url = new URL(route.request().url());
      if (url.origin !== localServer.origin) {
        external.push(url.href);
        return route.abort();
      }
      if (url.pathname === "/fixture/splats/tileset.json") await metadataHeld;
      if (!childPath.test(url.pathname)) return route.continue();
      const request = route.request();
      heldChildren.add(request);
      try {
        await held;
        if (heldChildren.has(request)) await route.continue();
      } finally {
        heldChildren.delete(request);
      }
    });

    const url = new URL(localServer.origin);
    url.searchParams.set("harness", "1");
    // Let the first dataset frame itself synchronously when it attaches.
    url.searchParams.append(
      "data",
      `points-url:${localServer.origin}/fixture/points.copc.laz`,
    );
    url.searchParams.append(
      "data",
      `tiles-url:${localServer.origin}/fixture/mesh/tileset.json`,
    );
    await tab.clock.install();
    await tab.goto(url.href);
    await tab.waitForFunction(
      () =>
        (window as unknown as SceneWindow).pointCloudScene?.datasets()
          .length === 2,
    );
    await installViewReader(tab);
    await tab.evaluate(() => {
      const scene = (window as unknown as SceneWindow).pointCloudScene;
      scene.camera.place({
        position: [0, -21, 10],
        focalPoint: [0, 0, 0],
        viewUp: [0, 0, 1],
      });
    });
    // Establish the mixed-scene baseline before introducing Gaussian work.
    // Bootstrap admission must be finished before exercising the child gate.
    await expect
      .poll(
        async () => {
          const { triangles, points } = await readView(tab);
          return [triangles, points > 0];
        },
        { timeout: 10_000 },
      )
      .toEqual([1, true]);
    await tab.locator("#open-add-dataset").click();
    await tab.locator("#dataset-type").selectOption("tiles");
    await tab.locator("#source-type").selectOption("url");
    await tab
      .locator("#dataset-url")
      .fill(`${localServer.origin}/fixture/splats/tileset.json`);
    await tab.locator("#submit-dataset").click();
    await tab.waitForFunction(
      () =>
        (window as unknown as SceneWindow).pointCloudScene.datasets().length ===
        3,
    );
    await tab.evaluate(() => {
      const scene = (window as unknown as SceneWindow).pointCloudScene;
      scene.settings.setScreenSpaceErrorPx(scene.datasets()[2]!.id, 64);
    });
    openTileset();

    const coarse = ["root"];
    const fine = ["root/0", "root/1"];
    await expect
      .poll(
        async () => {
          const view = await readView(tab);
          return [view.splats, view.triangles, view.points > 0];
        },
        { timeout: 10_000 },
      )
      .toEqual([coarse, 1, true]);
    const initial = await pixels(tab);
    expect(initial.red).toBeGreaterThan(40);
    expect(initial.cyan).toBeGreaterThan(40);

    expect(heldChildren.size).toBe(0);
    const now = await tab.evaluate(() => Date.now());
    await tab.clock.pauseAt(new Date(now + 200));
    const firstPaint = await tab.evaluate(() => {
      const w = window as unknown as MotionWindow;
      const scene = w.pointCloudScene;
      w.motionPaints = [];
      scene.onPaint(() => w.motionPaints.push(w.readMotionView()));
      scene.settings.setScreenSpaceErrorPx(scene.datasets()[2]!.id, 4);
      return scene.stats().paints;
    });
    const started = await advanceUntil(
      tab,
      64,
      (view) => view.activeRequests > 0,
    );
    expect(started.activeRequests).toBeGreaterThan(0);
    await expect
      .poll(() => heldChildren.size, { timeout: 10_000 })
      .toBeGreaterThan(0);

    const viewer = (await tab.locator("#viewer").boundingBox())!;
    const x = viewer.x + viewer.width / 2;
    const y = viewer.y + viewer.height / 2;
    await tab.mouse.move(x, y);
    await tab.mouse.down();
    for (let step = 0; step < 40; step++) {
      await tab.mouse.move(x + 8 * Math.sin(step / 3), y);
      await tab.clock.fastForward(44);
    }
    const moving = await readView(tab);
    expect(moving).toMatchObject({ regime: "interaction", splats: coarse });
    expect(moving.activeRequests).toBeGreaterThan(0);
    expect(heldChildren.size).toBeGreaterThan(0);
    expect(moving.camera).not.toEqual(started.camera);
    expect(moving.paint).toBeGreaterThan(started.paint);

    await tab.mouse.up();
    const stopped = await advanceUntil(
      tab,
      64,
      (view) => view.regime === "stationary" && !view.needsFrame,
    );
    expect(stopped).toMatchObject({
      regime: "stationary",
      needsFrame: false,
      pending: true,
    });
    expect(stopped.splats).toEqual(coarse);
    await tab.clock.runFor(160);
    expect((await readView(tab)).paint).toBe(stopped.paint);
    expect(heldChildren.size).toBeGreaterThan(0);

    const childFinished = tab.waitForEvent("requestfinished", {
      predicate: (request) => childPath.test(new URL(request.url()).pathname),
      timeout: 10_000,
    });
    release();
    await childFinished;
    const refined = await advanceUntil(
      tab,
      240,
      (view) => view.splats.length === fine.length,
    );
    expect(refined.splats).toEqual(fine);
    expect(refined.camera).toEqual(stopped.camera);
    expect(refined.paint).toBeGreaterThan(stopped.paint);

    const paints = await tab.evaluate(
      () => (window as unknown as MotionWindow).motionPaints,
    );
    expect(paints.map(({ paint }) => paint)).toEqual(
      Array.from(
        { length: refined.paint - firstPaint },
        (_, i) => firstPaint + i + 1,
      ),
    );
    let displayingFine = false;
    for (const view of paints) {
      if (view.paint > stopped.paint)
        expect(view.camera).toEqual(stopped.camera);
      if (view.splats.length === fine.length) {
        expect(view.paint).toBeGreaterThan(stopped.paint);
        displayingFine = true;
      }
      expect(view.splats).toEqual(displayingFine ? fine : coarse);
      expect(view.triangles).toBe(1);
      expect(view.points).toBeGreaterThan(0);
    }
    const final = await pixels(tab);
    expect(final.red).toBeLessThan(initial.red / 8);
    expect(final.blue).toBeGreaterThan(initial.blue + 40);
    expect(final.green).toBeGreaterThan(initial.green + 40);
    expect(final.cyan).toBeGreaterThan(40);
    expect(errors).toEqual([]);
    expect(external).toEqual([]);
  } finally {
    openTileset();
    release();
    await page?.close();
    await server?.close();
    await rm(directory, { recursive: true, force: true });
  }
}, 45_000);
