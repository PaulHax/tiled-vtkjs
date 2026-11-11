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
  ledger(): {
    rendererBytes: number;
    cpuRetainedBytes: number;
    cpuReservedBytes: number;
    workReservedBytes: number;
    activeOperations: number;
    owners: number;
  };
  refusals(): string[];
  setWorkLimit(bytes: number): void;
  setupMixed(): Promise<void>;
  setupGroup(): Promise<void>;
  admitGroup(): Promise<string>;
  selectMixed(): void;
  moveMixed(): void;
  hideMixed(): void;
  showMixed(): void;
  clearMixed(): void;
  growMixed(): void;
  removeMixed(): void;
  presentation(): {
    stats: { drawnTileIds: string[] };
    picks: string[];
    placement: number[];
    hit: { status: string } | null;
  };
  dispose(): void;
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
  it.each(["blend", "group-blend-gaussian"])(
    "rejects mixed translucent triangles in %s before any presentation",
    async (mode) => {
      const page = await open(mode);
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
    },
  );

  it("preserves translucent mesh rendering with the ordinary host pass", async () => {
    const page = await open("mesh-blend");
    try {
      const center = pixel(await pixels(page), 128, 128);
      expect(center[0]).toBeGreaterThan(110);
      expect(center[0]).toBeLessThan(145);
      expect(center[1]).toBe(0);
      expect(center[2]).toBeGreaterThan(110);
      expect(center[2]).toBeLessThan(145);
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

  it.each([
    ["blend-before-gaussian", "mesh-blend"],
    ["gaussian-before-blend", "gaussian-only"],
  ])(
    "rejects cross-member %s while preserving the admitted scene",
    async (mode, referenceMode) => {
      const reference = await open(referenceMode);
      const page = await open(mode);
      try {
        expect((await pixels(page)).data).toEqual(
          (await pixels(reference)).data,
        );
        expect(
          await page.evaluate(() =>
            (
              globalThis as unknown as { gaussianFixture: Fixture }
            ).gaussianFixture.errors(),
          ),
        ).toEqual([
          expect.stringContaining(
            "GaussianRenderingProfileError: translucent triangle",
          ),
        ]);
      } finally {
        await Promise.all([page.close(), reference.close()]);
      }
    },
  );

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
      const refusals = await page.evaluate(() =>
        (
          globalThis as unknown as { gaussianFixture: Fixture }
        ).gaussianFixture.refusals(),
      );
      expect(refusals.length).toBeGreaterThan(0);
      expect(
        refusals.every((value) => value.includes("ResourcePolicyError")),
      ).toBe(true);
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

  it("globally orders members with distinct mesh appearance wrappers", async () => {
    const reference = await open("single"),
      wrapped = await open("wrapped");
    try {
      expect((await pixels(wrapped)).data).toEqual(
        (await pixels(reference)).data,
      );
      expect(
        await wrapped.evaluate(() =>
          (
            globalThis as unknown as { gaussianFixture: Fixture }
          ).gaussianFixture.stats(),
        ),
      ).toMatchObject({ displayedSplats: 3, pendingSort: false });
    } finally {
      await reference.close();
      await wrapped.close();
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

describe("coherent mixed presentations and ownership", () => {
  const api = <K extends keyof Fixture>(
    page: Page,
    method: K,
    ...args: Parameters<Fixture[K]>
  ): Promise<Awaited<ReturnType<Fixture[K]>>> =>
    page.evaluate(
      ({ method, args }) => {
        const fixture = (globalThis as unknown as { gaussianFixture: Fixture })
          .gaussianFixture;
        return (fixture[method] as (...args: unknown[]) => unknown)(...args);
      },
      { method, args },
    ) as Promise<Awaited<ReturnType<Fixture[K]>>>;

  it("accepts mesh coverage, Gaussian coverage and picking together", async () => {
    const page = await open("gaussian-only");
    try {
      await api(page, "setupMixed");
      const before = await pixels(page);
      await api(page, "selectMixed");
      await expect.poll(() => api(page, "pending")).toBe(1);
      expect((await pixels(page)).data).toEqual(before.data);
      const pending = await api(page, "presentation");
      expect(pending.stats.drawnTileIds).toEqual(["parent"]);
      expect(pending.picks).toEqual(["parent"]);
      expect(pending.hit?.status).toBe("hit");
      await api(page, "release");
      await api(page, "settle");
      const accepted = await api(page, "presentation");
      expect(accepted.stats.drawnTileIds).toEqual(["child"]);
      expect(accepted.picks).toEqual(["child"]);
      const center = pixel(await pixels(page), 128, 128);
      expect(center[1]).toBeGreaterThan(center[0]! * 1.8);
      expect(center[1]).toBeGreaterThan(center[2]! * 1.8);
    } finally {
      await page.close();
    }
  });

  it("accepts placement and ray picking with the same pixels", async () => {
    const page = await open("gaussian-only");
    try {
      await api(page, "setupMixed");
      const before = await pixels(page);
      await api(page, "moveMixed");
      await expect.poll(() => api(page, "pending")).toBe(1);
      expect((await pixels(page)).data).toEqual(before.data);
      const pending = await api(page, "presentation");
      expect(pending.placement[12]).toBe(0);
      expect(pending.hit?.status).toBe("hit");
      await api(page, "release");
      await api(page, "settle");
      const accepted = await api(page, "presentation");
      expect(accepted.placement[12]).toBe(0.6);
      expect(accepted.hit?.status).toBe("miss");
      expect((await pixels(page)).data).not.toEqual(before.data);
    } finally {
      await page.close();
    }
  });

  it.each(["hideMixed", "removeMixed"] as const)(
    "%s suppresses pixels and picks while sort work is held",
    async (action) => {
      const page = await open("gaussian-only");
      try {
        await api(page, "setupMixed");
        await api(page, "selectMixed");
        await expect.poll(() => api(page, "pending")).toBe(1);
        await page.evaluate(
          (action) =>
            (
              globalThis as unknown as { gaussianFixture: Fixture }
            ).gaussianFixture[action](),
          action,
        );
        expect(pixel(await pixels(page), 128, 128)).toEqual([0, 0, 0]);
        const presentation = await api(page, "presentation");
        expect(presentation.stats.drawnTileIds).toEqual([]);
        expect(presentation.picks).toEqual([]);
        const held = await api(page, "ledger");
        expect(held.activeOperations).toBe(1);
        expect(held.workReservedBytes).toBeGreaterThan(0);
        await api(page, "release");
        await api(page, "settle");
        expect(pixel(await pixels(page), 128, 128)).toEqual([0, 0, 0]);
        await api(page, "dispose");
        await expect
          .poll(() => api(page, "ledger"))
          .toMatchObject({
            rendererBytes: 0,
            cpuRetainedBytes: 0,
            cpuReservedBytes: 0,
            workReservedBytes: 0,
            activeOperations: 0,
            owners: 0,
          });
      } finally {
        await page.close();
      }
    },
  );
  it("reveals mixed content only after the shown presentation is ready", async () => {
    const page = await open("gaussian-only");
    try {
      await api(page, "setupMixed");
      await api(page, "selectMixed");
      await expect.poll(() => api(page, "pending")).toBe(1);
      await api(page, "hideMixed");
      await api(page, "showMixed");
      expect(pixel(await pixels(page), 128, 128)).toEqual([0, 0, 0]);
      expect((await api(page, "presentation")).picks).toEqual([]);
      await api(page, "release");
      await api(page, "settle");
      expect((await api(page, "presentation")).picks).toEqual(["child"]);
      expect(pixel(await pixels(page), 128, 128)[1]).toBeGreaterThan(200);
    } finally {
      await page.close();
    }
  });

  it("clears unaccepted capacity growth while another empty owner remains", async () => {
    const page = await open("gaussian-only");
    try {
      await api(page, "setupMixed");
      const before = await api(page, "ledger");
      await api(page, "growMixed");
      await expect.poll(() => api(page, "pending")).toBe(1);
      expect((await api(page, "ledger")).rendererBytes).toBeGreaterThan(
        before.rendererBytes,
      );
      await api(page, "clearMixed");
      await expect
        .poll(async () => (await api(page, "ledger")).rendererBytes)
        .toBe(0);
      expect(pixel(await pixels(page), 128, 128)).toEqual([0, 0, 0]);
      expect((await api(page, "ledger")).activeOperations).toBe(1);
      await api(page, "release");
      await api(page, "settle");
      expect(await api(page, "ledger")).toMatchObject({
        rendererBytes: 0,
        cpuRetainedBytes: 0,
        workReservedBytes: 0,
        activeOperations: 0,
      });
      expect(await api(page, "refusals")).toEqual([]);
      await api(page, "setupMixed");
      expect(pixel(await pixels(page), 128, 128)[0]).toBeGreaterThan(100);
      await api(page, "dispose");
      await expect
        .poll(() => api(page, "ledger"))
        .toMatchObject({
          rendererBytes: 0,
          cpuRetainedBytes: 0,
          cpuReservedBytes: 0,
          workReservedBytes: 0,
          activeOperations: 0,
          owners: 0,
        });
    } finally {
      await page.close();
    }
  });

  it("retains accepted coverage on sort refusal and retries after policy changes", async () => {
    const page = await open("gaussian-only");
    try {
      await api(page, "setupMixed");
      const before = await pixels(page);
      await api(page, "setWorkLimit", 0);
      await api(page, "selectMixed");
      await api(page, "settle");
      expect((await pixels(page)).data).toEqual(before.data);
      expect((await api(page, "presentation")).picks).toEqual(["parent"]);
      const refusals = await api(page, "refusals");
      expect(refusals.some((value) => value.includes("work memory"))).toBe(
        true,
      );
      for (let frame = 0; frame < 3; frame++) await api(page, "settle");
      expect(await api(page, "refusals")).toEqual(refusals);
      await api(page, "setWorkLimit", 1048576);
      await api(page, "release");
      await api(page, "settle");
      expect((await api(page, "presentation")).picks).toEqual(["child"]);
      expect(pixel(await pixels(page), 128, 128)[1]).toBeGreaterThan(200);
      expect(await api(page, "errors")).toEqual([]);
    } finally {
      await page.close();
    }
  });
  it("admits a mixed replacement group within final and overlap limits", async () => {
    const page = await open("group-budget");
    try {
      await api(page, "setupGroup");
      const outcome = await api(page, "admitGroup");
      expect(outcome, JSON.stringify(await api(page, "refusals"))).toBe(
        "queued",
      );
      expect(await api(page, "refusals")).toEqual([]);
      expect((await api(page, "presentation")).picks).toEqual([
        "left",
        "right",
      ]);
      expect((await api(page, "ledger")).rendererBytes).toBeLessThanOrEqual(
        360,
      );
      const center = pixel(await pixels(page), 128, 128);
      expect(center[2]).toBeGreaterThan(center[0]! * 2);
    } finally {
      await page.close();
    }
  });
});
