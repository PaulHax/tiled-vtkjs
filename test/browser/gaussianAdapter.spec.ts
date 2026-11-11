import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { chromium, type Browser, type Page } from "playwright";
import sharp from "sharp";
import { build } from "vite";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { startStaticServer, type StaticServer } from "./server";

let browser: Browser;
let server: StaticServer;
let directory = "";

beforeAll(async () => {
  directory = await mkdtemp(resolve(tmpdir(), "gaussian-shared-browser-"));
  await build({
    configFile: false,
    logLevel: "silent",
    build: {
      emptyOutDir: true,
      outDir: directory,
      lib: {
        entry: resolve("test/browser/gaussianAdapterPage.mjs"),
        fileName: "gaussianAdapterPage",
        formats: ["es"],
      },
    },
  });
  await writeFile(
    resolve(directory, "index.html"),
    '<!doctype html><style>html,body,#view{margin:0;width:256px;height:256px;overflow:hidden}</style><div id="view"></div><script type="module" src="gaussianAdapterPage.js"></script>',
  );
  server = await startStaticServer({ "/fixture": directory });
  browser = await chromium.launch({ headless: true });
});

afterAll(async () => {
  await browser?.close();
  await server?.close();
  if (directory) await rm(directory, { recursive: true, force: true });
});

type Stats = {
  displayedSplats: number;
  pendingSort: boolean;
  attributeUploadBytes: number;
  sortCount: number;
  residentBytes: number;
};
type Fixture = {
  stats(): Stats;
  errors(): string[];
  pending(): number;
  hold(): void;
  release(): void;
  setCamera(position: number[]): void;
  settle(): Promise<Stats>;
  moveRepeatedly(count: number): Promise<Stats>;
};

const open = async (mode: string): Promise<Page> => {
  const page = await browser.newPage({ viewport: { width: 256, height: 256 } });
  await page.goto(`${server.origin}/fixture/index.html?mode=${mode}`);
  await page.locator('html[data-ready="true"]').waitFor();
  return page;
};

const pixels = async (page: Page) =>
  sharp(await page.locator("canvas").screenshot())
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
const pixel = (
  image: Awaited<ReturnType<typeof pixels>>,
  x: number,
  y: number,
): number[] => [
  ...image.data.subarray(
    (y * image.info.width + x) * 4,
    (y * image.info.width + x) * 4 + 3,
  ),
];

describe("shared Gaussian realization", () => {
  it("rejects mixed translucent triangles before any Gaussian presentation", async () => {
    const page = await open("blend");
    try {
      const result = await page.evaluate(() => {
        const api = (globalThis as unknown as { gaussianFixture: Fixture })
          .gaussianFixture;
        return { stats: api.stats(), errors: api.errors() };
      });
      expect(result.stats.displayedSplats).toBe(0);
      expect(result.stats.pendingSort).toBe(false);
      expect(result.errors).toEqual([
        expect.stringContaining(
          "GaussianRenderingProfileError: translucent triangle",
        ),
      ]);
      const image = await pixels(page);
      expect(
        image.data
          .filter((_value, index) => index % 4 !== 3)
          .every((value) => value === 0),
      ).toBe(true);
    } finally {
      await page.close();
    }
  });

  it("keeps the last presentation when placement overlap exceeds its budget", async () => {
    const page = await open("budget");
    try {
      const before = await pixels(page);
      const final = await page.evaluate(async () => {
        const api = (globalThis as unknown as { gaussianFixture: Fixture })
          .gaussianFixture;
        await api.moveRepeatedly(1);
        return api.settle();
      });
      expect(final.pendingSort).toBe(false);
      expect(final.residentBytes).toBeLessThanOrEqual(1600);
      expect((await pixels(page)).data).toEqual(before.data);
      expect(
        await page.evaluate(() =>
          (
            globalThis as unknown as { gaussianFixture: Fixture }
          ).gaussianFixture.errors(),
        ),
      ).toEqual([
        expect.stringContaining("GaussianResourceLimitError"),
        expect.stringContaining("GaussianResourceLimitError"),
      ]);
    } finally {
      await page.close();
    }
  });

  it("interleaves two members exactly like one field and tests opaque depth", async () => {
    const combined = await open("single"),
      shared = await open("shared"),
      occluded = await open("occlusion");
    try {
      const reference = await pixels(combined),
        actual = await pixels(shared),
        depth = await pixels(occluded);
      expect(actual.data).toEqual(reference.data);
      const center = pixel(actual, 128, 128);
      expect(center[2]).toBeGreaterThan(center[1]! * 1.8);
      expect(center[1]).toBeGreaterThan(center[0]! * 1.8);
      const blocked = pixel(depth, 100, 128),
        clear = pixel(depth, 156, 128);
      expect(Math.max(...blocked) - Math.min(...blocked)).toBeLessThanOrEqual(
        1,
      );
      expect(blocked[0]).toBeGreaterThan(50);
      expect(clear[2]).toBeGreaterThan(clear[0]! * 2);
      expect(
        await shared.evaluate(() =>
          (
            globalThis as unknown as { gaussianFixture: Fixture }
          ).gaussianFixture.errors(),
        ),
      ).toEqual([]);
    } finally {
      await Promise.all([combined.close(), shared.close(), occluded.close()]);
    }
  });

  it("updates directional harmonics while leaving unchanged geometry resident", async () => {
    const page = await open("sh");
    try {
      const before = await pixels(page);
      const initial = await page.evaluate(() =>
        (
          globalThis as unknown as { gaussianFixture: Fixture }
        ).gaussianFixture.stats(),
      );
      const final = await page.evaluate(async () => {
        const api = (globalThis as unknown as { gaussianFixture: Fixture })
          .gaussianFixture;
        api.setCamera([0, 0, -10]);
        return api.settle();
      });
      const after = await pixels(page);
      expect(
        pixel(after, 128, 128)[0]! - pixel(before, 128, 128)[0]!,
      ).toBeGreaterThan(100);
      expect(final.attributeUploadBytes - initial.attributeUploadBytes).toBe(
        16,
      );
      expect(final.displayedSplats).toBe(1);
    } finally {
      await page.close();
    }
  });

  it("settles A to B to A camera changes and bounds obsolete placement ranges", async () => {
    const page = await open("shared");
    try {
      const initial = await page.evaluate(() =>
        (
          globalThis as unknown as { gaussianFixture: Fixture }
        ).gaussianFixture.stats(),
      );
      await page.evaluate(() => {
        const api = (globalThis as unknown as { gaussianFixture: Fixture })
          .gaussianFixture;
        api.hold();
        api.setCamera([1, 0, 10]);
      });
      await page.waitForFunction(
        () =>
          (
            globalThis as unknown as { gaussianFixture: Fixture }
          ).gaussianFixture.pending() === 1,
      );
      const returned = await page.evaluate(async () => {
        const api = (globalThis as unknown as { gaussianFixture: Fixture })
          .gaussianFixture;
        api.setCamera([0, 0, 10]);
        api.release();
        return api.settle();
      });
      expect(returned.pendingSort).toBe(false);
      expect(returned.sortCount).toBe(initial.sortCount);
      expect(returned.attributeUploadBytes).toBe(initial.attributeUploadBytes);
      await page.evaluate(() =>
        (
          globalThis as unknown as { gaussianFixture: Fixture }
        ).gaussianFixture.hold(),
      );
      const first = await page.evaluate(() =>
        (
          globalThis as unknown as { gaussianFixture: Fixture }
        ).gaussianFixture.moveRepeatedly(10),
      );
      const repeated = await page.evaluate(() =>
        (
          globalThis as unknown as { gaussianFixture: Fixture }
        ).gaussianFixture.moveRepeatedly(200),
      );
      expect(repeated.residentBytes).toBe(first.residentBytes);
      expect(repeated.residentBytes).toBeLessThanOrEqual(131072);
      const settled = await page.evaluate(async () => {
        const api = (globalThis as unknown as { gaussianFixture: Fixture })
          .gaussianFixture;
        api.release();
        return api.settle();
      });
      expect(settled.pendingSort).toBe(false);
      expect(settled.displayedSplats).toBe(3);
      expect(
        await page.evaluate(() =>
          (
            globalThis as unknown as { gaussianFixture: Fixture }
          ).gaussianFixture.errors(),
        ),
      ).toEqual([]);
    } finally {
      await page.close();
    }
  });
});
