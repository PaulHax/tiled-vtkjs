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
  pendingPreparation: boolean;
  attributeUploadBytes: number;
  sortCount: number;
  residentBytes: number;
};
type MeshCacheState = {
  drawn: string[];
  picks: string[];
  pendingJobs: number;
  pendingTiles: number;
  queuedJobs: number;
  residentBytes: number;
  residentTextureBytes: number;
  textureRepresentation: string;
  hits: { status: string; rayDepth?: number; scenePoint?: number[] }[];
  gap: { status: string };
  camera: {
    position: number[];
    focalPoint: number[];
    parallelScale: number;
    viewport: number[];
  };
  cached: string[];
  cacheReads: string[];
  errors: string[];
  refusals: string[];
};
type Fixture = {
  stageMixedProbe(
    id: string,
    count: number | number[],
    triangles: number,
    replacement?: string[],
  ): string;
  stepMixedProbe(): ProbeFrame;
  addInvisibleProbe(id: string): string;
  cancelMixedProbe(id: string): void;
  restoreMixedProbeContext(): Promise<{ before: number; after: number }>;
  showCachedMeshParent(): Promise<string>;
  refineCachedMeshesCold(): Promise<string>;
  hideCachedMeshes(): Promise<void>;
  refineCachedMeshesWarm(): Promise<{
    first: string;
    second: string;
    retainedAfterFirst: string[];
  }>;
  cachedMeshState(): MeshCacheState;
  setupRecovery(): void;
  refineRecovery(): void;
  panRecovery(evict: boolean): void;
  flushRecovery(): void;
  recoveryStats(): {
    stats: {
      renderer: { pendingTiles: number; drawnTileIds: string[] };
      queue: { cached: number; workPending: boolean };
    };
    fetches: string[];
    work: number;
    pick: { status: string } | null;
  };
  stats(): Stats;
  submitWithoutCamera(prepared: boolean): string;
  cancelWithoutCamera(clear: boolean): void;
  drawUnplaced(): Promise<void>;
  ledger(): {
    rendererBytes: number;
    cpuRetainedBytes: number;
    cpuReservedBytes: number;
    workReservedBytes: number;
    activeOperations: number;
    queuedOperations: number;
    owners: number;
  };
  refusals(): string[];
  setWorkLimit(bytes: number): void;
  setupMixed(childCount?: number): Promise<void>;
  setupGroup(): Promise<void>;
  setupDemand(gaussian: boolean): Promise<void>;
  queueDemand(gaussian: boolean): string;
  additionalGaussian(): string;
  replacementGaussian(): string;
  flushUploads(): void;
  admissionRefusals(): { ids: string[]; message: string }[];
  demandState(): {
    required: number;
    renderer: {
      residentBytes: number;
      pendingTiles: number;
      pendingJobs: number;
    };
    queue: { queuedJobs: number };
  };
  reduceDemand(): string[];
  raiseDemand(bytes: number): void;
  admitGroup(): Promise<string>;
  selectMixed(): void;
  moveMixed(x?: number): void;
  hideMixed(): void;
  showMixed(): void;
  clearMixed(): void;
  growMixed(): void;
  stageGrowth(id: string, count: number, restoreWork?: boolean): string;
  cancelGrowth(id: string): boolean;
  selectGrowth(id: string): void;
  prepareInvisible(ids: string[]): string;
  removeMixed(): void;
  presentation(): {
    stats: { drawnTileIds: string[]; submittedTileIds: string[] };
    picks: string[];
    placement: number[];
    hit: { status: string } | null;
  };
  dispose(): void;
  errors(): string[];
  pending(): number;
  blockWork(kind?: "slots" | "bytes" | "growth"): void;
  releaseWork(): void;
  releaseWorkSlot(): void;
  releaseNext(): void;
  releaseWhilePreparing(ids: string[]): Promise<string>;
  hold(): void;
  release(): void;
  setCamera(position: number[]): void;
  settle(): Promise<Stats>;
  moveRepeatedly(count: number): Promise<Stats>;
};

type ProbeFrame = {
  pixel: number[];
  uploadStateReads: number;
  stateRestored: boolean;
  uploads: Array<{
    phase: "prepare" | "draw";
    kind: "mesh" | "order" | "attributes";
    bytes: number;
    count?: number;
    x?: number;
  }>;
  stats: Stats & { orderUploadBytes: number; presentationRevision: number };
  picks: string[];
  queued: boolean;
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
      const before = await pixels(page);
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
      expect((await pixels(page)).data).toEqual(before.data);
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
      const placementOverlap = initial.residentBytes * 3;
      expect(first.residentBytes).toBeLessThanOrEqual(placementOverlap);
      expect(repeated.residentBytes).toBeLessThanOrEqual(placementOverlap);
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

  const stepUntil = async (
    page: Page,
    condition: (frame: ProbeFrame) => boolean,
    frames: ProbeFrame[] = [],
  ) => {
    for (let i = 0; i < 200; i++) {
      const frame = await api(page, "stepMixedProbe");
      frames.push(frame);
      if (condition(frame)) return frame;
    }
    throw new Error(
      "The streamed presentation did not reach the requested state",
    );
  };

  it("budgets mixed upload ranges before paint and preserves the accepted draw on cancellation", async () => {
    const page = await open("budgeted-uploads");
    try {
      const frames: ProbeFrame[] = [];
      expect(await api(page, "stageMixedProbe", "parent", 1, 1)).toBe("queued");
      const parent = await stepUntil(
        page,
        (frame) => frame.stats.displayedSplats === 1,
        frames,
      );
      expect(parent.picks).toEqual(["parent"]);
      expect(parent.pixel[0]).toBeGreaterThan(30);

      expect(await api(page, "stageMixedProbe", "cancelled", 80, 30)).toBe(
        "queued",
      );
      const partial = await stepUntil(
        page,
        (frame) =>
          frame.uploads.some(
            (upload) => upload.kind === "attributes" && upload.count !== 1,
          ),
        frames,
      );
      expect(partial.picks).toEqual(["parent"]);
      expect(partial.pixel).toEqual(parent.pixel);
      expect(partial.stats.displayedSplats).toBe(1);
      await api(page, "cancelMixedProbe", "cancelled");
      const settled = await stepUntil(
        page,
        (frame) =>
          !frame.stats.pendingSort &&
          !frame.stats.pendingPreparation &&
          !frame.queued,
        frames,
      );
      expect(settled.picks).toEqual(["parent"]);
      expect(settled.pixel).toEqual(parent.pixel);

      expect(
        await api(page, "stageMixedProbe", "child", 80, 30, ["parent"]),
      ).toBe("queued");
      const start = frames.length;
      const child = await stepUntil(
        page,
        (frame) => frame.stats.displayedSplats === 80,
        frames,
      );
      const transition = frames.slice(start, -1);
      expect(
        transition.filter((frame) =>
          frame.uploads.some((upload) => upload.kind === "mesh"),
        ).length,
      ).toBeGreaterThan(1);
      expect(
        transition.filter((frame) =>
          frame.uploads.some((upload) => upload.kind === "attributes"),
        ).length,
      ).toBeGreaterThan(1);
      for (const frame of transition) {
        expect(frame.picks).toEqual(["parent"]);
        expect(frame.pixel).toEqual(parent.pixel);
      }
      expect(child.picks).toEqual(["child"]);
      expect(child.pixel[2]).toBeGreaterThan(parent.pixel[2]!);
      await stepUntil(
        page,
        (frame) =>
          !frame.stats.pendingSort &&
          !frame.stats.pendingPreparation &&
          !frame.queued,
        frames,
      );

      let attributes = 0;
      let order = 0;
      let lastRevision = 0;
      for (const frame of frames) {
        expect(
          frame.uploads.every((upload) => upload.phase === "prepare"),
        ).toBe(true);
        expect(
          frame.uploads.reduce((sum, upload) => sum + upload.bytes, 0),
        ).toBeLessThanOrEqual(1024);
        for (const upload of frame.uploads) {
          if (upload.kind === "attributes") attributes += upload.bytes;
          if (upload.kind === "order") order += upload.bytes;
        }
        expect(frame.stats.attributeUploadBytes).toBe(attributes);
        expect(frame.stats.orderUploadBytes).toBe(order);
        expect(frame.stats.presentationRevision).toBeGreaterThanOrEqual(
          lastRevision,
        );
        lastRevision = frame.stats.presentationRevision;
      }
      expect(
        frames.some((frame) =>
          frame.uploads.some((upload) => (upload.x ?? 0) > 0),
        ),
      ).toBe(true);
      expect(await api(page, "errors")).toEqual([]);
    } finally {
      await page.close();
    }
  });

  it("uploads adjacent dirty primitives together within the frame cap", async () => {
    const page = await open("budgeted-uploads");
    try {
      expect(await api(page, "stageMixedProbe", "parent", [30, 30], 1)).toBe(
        "queued",
      );
      const frames: ProbeFrame[] = [];
      const accepted = await stepUntil(
        page,
        (frame) => frame.stats.displayedSplats === 60,
        frames,
      );
      const attributes = frames.flatMap(({ uploads }) =>
        uploads.filter(({ kind }) => kind === "attributes"),
      );
      expect(attributes.map(({ count }) => count)).toEqual([60, 60, 60, 60]);
      expect(attributes.reduce((sum, { bytes }) => sum + bytes, 0)).toBe(
        60 * 56,
      );
      expect(accepted.picks).toEqual(["parent"]);
      expect(accepted.pixel[0]).toBeGreaterThan(150);
      expect(await api(page, "errors")).toEqual([]);
    } finally {
      await page.close();
    }
  });

  it("finishes a compatible upload through new content arrivals", async () => {
    const page = await open("budgeted-uploads");
    try {
      expect(await api(page, "stageMixedProbe", "parent", 1, 1)).toBe("queued");
      const parent = await stepUntil(
        page,
        (frame) => frame.stats.displayedSplats === 1,
      );
      expect(
        await api(page, "stageMixedProbe", "child", 80, 30, ["parent"]),
      ).toBe("queued");
      const frames: ProbeFrame[] = [];
      await stepUntil(
        page,
        (frame) => frame.uploads.some(({ kind }) => kind === "attributes"),
        frames,
      );
      for (let index = 0; index < 8; index++) {
        expect(await api(page, "addInvisibleProbe", `arrival-${index}`)).toBe(
          "queued",
        );
        frames.push(await api(page, "stepMixedProbe"));
      }
      const child = await stepUntil(
        page,
        (frame) => frame.stats.displayedSplats === 80,
        frames,
      );
      const attrs = frames.flatMap((frame) =>
        frame.uploads.filter(({ kind }) => kind === "attributes"),
      );
      expect(attrs.reduce((sum, { bytes }) => sum + bytes, 0)).toBe(80 * 56);
      expect(child.picks).toEqual(["child"]);
      expect(child.pixel[2]).toBeGreaterThan(150);
      expect(child.stats.orderUploadBytes - parent.stats.orderUploadBytes).toBe(
        80 * 4,
      );
      expect(frames.every((frame) => frame.stateRestored)).toBe(true);
      expect(
        frames.some(
          (frame) =>
            frame.uploads.filter(({ kind }) => kind !== "mesh").length > 1 &&
            frame.uploadStateReads === 1,
        ),
      ).toBe(true);
      expect(await api(page, "errors")).toEqual([]);
    } finally {
      await page.close();
    }
  });

  it("reuploads a partially staged snapshot on the restored WebGL context", async () => {
    const page = await open("budgeted-uploads");
    try {
      expect(await api(page, "stageMixedProbe", "parent", 1, 1)).toBe("queued");
      await stepUntil(page, (frame) => frame.stats.displayedSplats === 1);
      expect(
        await api(page, "stageMixedProbe", "child", 80, 30, ["parent"]),
      ).toBe("queued");
      const partial = await stepUntil(page, (frame) =>
        frame.uploads.some(
          (upload) => upload.kind === "attributes" && upload.count !== 1,
        ),
      );
      expect(partial.picks).toEqual(["parent"]);
      const generation = await api(page, "restoreMixedProbeContext");
      expect(generation.after).toBeGreaterThan(generation.before);
      const frames: ProbeFrame[] = [];
      const child = await stepUntil(
        page,
        (frame) => frame.stats.displayedSplats === 80,
        frames,
      );
      const textures = frames.flatMap((frame) =>
        frame.uploads.filter(({ kind }) => kind !== "mesh"),
      );
      expect(textures.every((upload) => upload.phase === "prepare")).toBe(true);
      expect(
        textures
          .filter(({ kind }) => kind === "attributes")
          .reduce((sum, upload) => sum + upload.bytes, 0),
      ).toBe(80 * 56);
      expect(
        textures
          .filter(({ kind }) => kind === "order")
          .reduce((sum, upload) => sum + upload.bytes, 0),
      ).toBe(80 * 4);
      expect(child.picks).toEqual(["child"]);
      expect(child.pixel[2]).toBeGreaterThan(150);
      expect(await api(page, "errors")).toEqual([]);
    } finally {
      await page.close();
    }
  });

  it("restores the textured fine frontier after warm-cache hide and show", async () => {
    const page = await open("mesh-cache-restoration");
    const cyanPixels = (image: Awaited<ReturnType<typeof pixels>>): number => {
      let count = 0;
      for (let index = 0; index < image.data.length; index += 4) {
        const r = image.data[index]!,
          g = image.data[index + 1]!,
          b = image.data[index + 2]!;
        if (r < 80 && g > 105 && b > 105 && Math.abs(g - b) < 80) count++;
      }
      return count;
    };
    try {
      expect(await api(page, "showCachedMeshParent")).toBe("queued");
      const coarsePixels = cyanPixels(await pixels(page));
      expect(await api(page, "refineCachedMeshesCold")).toBe("queued");
      const before = await api(page, "cachedMeshState");
      const finePixels = cyanPixels(await pixels(page));
      expect(finePixels).toBeGreaterThan(200);
      expect(coarsePixels).toBeGreaterThan(finePixels * 1.5);
      expect(before.drawn).toEqual(["cache-a", "cache-b"]);
      expect(before.picks).toEqual(before.drawn);
      expect(
        before.hits.every((hit) => hit.status === "hit" && hit.rayDepth! > 0),
      ).toBe(true);
      for (const hit of before.hits)
        expect(hit.scenePoint?.[2]).toBeCloseTo(0, 6);
      expect(before.gap.status).toBe("miss");
      expect(before.textureRepresentation).toBe("rgba");
      expect(before.residentTextureBytes).toBeGreaterThan(0);
      expect(before.errors).toEqual([]);
      expect(before.refusals).toEqual([]);

      await api(page, "hideCachedMeshes");
      expect(cyanPixels(await pixels(page))).toBe(0);
      const hidden = await api(page, "cachedMeshState");
      expect(hidden.drawn).toEqual([]);
      expect(hidden.picks).toEqual([]);
      expect(hidden.hits.every((hit) => hit.status === "miss")).toBe(true);
      expect(hidden.residentBytes).toBe(0);
      expect((await api(page, "ledger")).rendererBytes).toBe(0);
      expect((await api(page, "ledger")).cpuRetainedBytes).toBeGreaterThan(0);
      expect(hidden.cached).toEqual(before.cached);

      expect(await api(page, "showCachedMeshParent")).toBe("queued");
      const replay = await api(page, "refineCachedMeshesWarm");
      const after = await api(page, "cachedMeshState");
      const restoredPixels = cyanPixels(await pixels(page));
      expect(
        Math.abs(restoredPixels - finePixels),
        JSON.stringify({ finePixels, restoredPixels, replay, after }),
      ).toBeLessThanOrEqual(Math.max(50, Math.floor(finePixels / 10)));
      expect(replay).toEqual({
        first: "queued",
        second: "queued",
        retainedAfterFirst: ["cache-parent"],
      });
      expect(after.drawn).toEqual(before.drawn);
      expect(after.picks).toEqual(before.picks);
      expect(
        after.hits.every((hit) => hit.status === "hit" && hit.rayDepth! > 0),
      ).toBe(true);
      for (const hit of after.hits)
        expect(hit.scenePoint?.[2]).toBeCloseTo(0, 6);
      expect(after.gap.status).toBe("miss");
      expect(after.camera).toEqual(before.camera);
      expect(after).toMatchObject({
        pendingJobs: 0,
        pendingTiles: 0,
        queuedJobs: 0,
      });
      expect(after.cacheReads).toEqual([
        "cache-parent",
        "cache-a",
        "cache-b",
        "cache-parent",
        "cache-a",
        "cache-b",
      ]);
      expect(after.errors).toEqual([]);
      expect(after.refusals).toEqual([]);
      await api(page, "dispose");
      expect(await api(page, "ledger")).toMatchObject({
        rendererBytes: 0,
        cpuRetainedBytes: 0,
        cpuReservedBytes: 0,
        workReservedBytes: 0,
        activeOperations: 0,
        queuedOperations: 0,
        owners: 0,
      });
    } finally {
      await page.close();
    }
  });

  it.each([false, true])(
    "renders a wanted sibling after group cancellation (evict=%s)",
    async (evict) => {
      const page = await open("member-recovery");
      try {
        await api(page, "setupRecovery");
        await expect
          .poll(
            async () =>
              (await api(page, "recoveryStats")).stats.renderer.pendingTiles,
          )
          .toBe(1);
        await api(page, "flushRecovery");
        expect(pixel(await pixels(page), 128, 128)[0]).toBeGreaterThan(200);
        await api(page, "refineRecovery");
        await expect
          .poll(
            async () =>
              (await api(page, "recoveryStats")).stats.renderer.pendingTiles,
          )
          .toBe(2);
        await api(page, "panRecovery", evict);
        await expect
          .poll(
            async () =>
              (await api(page, "recoveryStats")).stats.renderer.pendingTiles,
          )
          .toBe(1);
        expect(pixel(await pixels(page), 128, 128)[0]).toBeGreaterThan(200);
        await api(page, "flushRecovery");
        const restored = await api(page, "recoveryStats");
        expect(restored.stats.renderer.drawnTileIds).toEqual(["root/0"]);
        expect(restored.pick?.status).toBe("hit");
        expect(restored.work).toBe(0);
        expect(restored.stats.queue.workPending).toBe(false);
        expect(
          restored.fetches.filter((url) => url.endsWith("/child-0.glb")),
        ).toHaveLength(1);
        expect(
          restored.fetches.filter((url) => url.endsWith("/child-1.glb")),
        ).toHaveLength(1);
        const green = pixel(await pixels(page), 128, 128);
        expect(green[1]).toBeGreaterThan(200);
        expect(green[0]).toBe(0);
        await api(page, "dispose");
        await expect
          .poll(() => api(page, "ledger"))
          .toMatchObject({
            rendererBytes: 0,
            cpuRetainedBytes: 0,
            cpuReservedBytes: 0,
            activeOperations: 0,
            queuedOperations: 0,
            workReservedBytes: 0,
            owners: 0,
          });
      } finally {
        await page.close();
      }
    },
  );

  it.each([false, true])(
    "presents a camera sort during off-frontier preparation (prepare at completion=%s)",
    async (atCompletion) => {
      const page = await open("sh");
      const waitForOrder = () =>
        page.waitForFunction(
          () =>
            !(
              globalThis as unknown as { gaussianFixture: Fixture }
            ).gaussianFixture.stats().pendingSort,
          null,
          { timeout: 10000 },
        );
      try {
        const before = await pixels(page);
        await api(page, "hold");
        await api(page, "setCamera", [0, 0, -10]);
        await expect.poll(() => api(page, "pending")).toBe(1);
        const ids = ["off-frontier-a", "off-frontier-b"];
        if (atCompletion) {
          expect(await api(page, "releaseWhilePreparing", ids)).toBe("queued");
        } else {
          expect(await api(page, "prepareInvisible", ids)).toBe("queued");
          await api(page, "releaseNext");
        }
        await waitForOrder();
        await api(page, "settle");
        const after = await pixels(page);
        expect(
          pixel(after, 128, 128)[0]! - pixel(before, 128, 128)[0]!,
        ).toBeGreaterThan(100);
        expect(
          (await api(page, "presentation")).stats.submittedTileIds,
        ).toEqual(expect.arrayContaining(["off-frontier-a", "off-frontier-b"]));
        const sorted = await api(page, "stats");
        expect(await api(page, "prepareInvisible", ["off-frontier-c"])).toBe(
          "queued",
        );
        await expect
          .poll(
            async () =>
              (await api(page, "presentation")).stats.submittedTileIds,
          )
          .toContain("off-frontier-c");
        await api(page, "settle");
        expect((await pixels(page)).data).toEqual(after.data);
        expect((await api(page, "stats")).sortCount).toBe(sorted.sortCount);
        await api(page, "setCamera", [0, 0, 10]);
        await expect.poll(() => api(page, "pending")).toBe(1);
        expect(await api(page, "prepareInvisible", ["off-frontier-d"])).toBe(
          "queued",
        );
        await api(page, "releaseNext");
        await waitForOrder();
        await api(page, "settle");
        expect((await pixels(page)).data).toEqual(before.data);
        expect((await api(page, "stats")).displayedSplats).toBe(1);
        expect(await api(page, "errors")).toEqual([]);
        expect(await api(page, "refusals")).toEqual([]);
        await api(page, "dispose");
        await expect
          .poll(async () => (await api(page, "ledger")).activeOperations)
          .toBe(0);
        expect(await api(page, "ledger")).toMatchObject({
          queuedOperations: 0,
          workReservedBytes: 0,
          rendererBytes: 0,
          cpuRetainedBytes: 0,
          cpuReservedBytes: 0,
          owners: 0,
        });
      } finally {
        await page.close();
      }
    },
  );

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

  it.each(["slots", "bytes", "growth"] as const)(
    "presents the latest mixed scene after held work (%s)",
    async (kind) => {
      const page = await open("gaussian-only");
      try {
        await api(page, "setupMixed", kind === "growth" ? 3 : 1);
        const before = await pixels(page);
        await api(page, "blockWork", kind);
        await api(page, "moveMixed", 0.6);
        await expect
          .poll(async () => (await api(page, "ledger")).queuedOperations)
          .toBe(1);
        const waiting = await api(page, "ledger");
        expect(waiting.activeOperations).toBe(kind === "bytes" ? 0 : 1);
        expect(waiting.workReservedBytes).toBe(
          kind === "bytes" ? 1584 : kind === "growth" ? 1056 : 0,
        );
        expect(await api(page, "errors")).toEqual([]);
        expect(await api(page, "refusals")).toEqual([]);
        await api(page, "moveMixed", -0.6);
        await api(page, "selectMixed");
        expect((await pixels(page)).data).toEqual(before.data);
        expect((await api(page, "presentation")).picks).toEqual(["parent"]);
        if (kind === "growth") {
          await api(page, "releaseWorkSlot");
          await expect
            .poll(async () => (await api(page, "ledger")).activeOperations)
            .toBe(0);
          expect((await api(page, "ledger")).queuedOperations).toBe(1);
          expect((await api(page, "ledger")).workReservedBytes).toBe(1056);
          expect((await pixels(page)).data).toEqual(before.data);
          expect(await api(page, "refusals")).toEqual([]);
        }
        await page.waitForFunction(
          () =>
            !(
              globalThis as unknown as { gaussianFixture: Fixture }
            ).gaussianFixture.stats().pendingPreparation,
        );
        await api(page, "releaseWork");
        await expect.poll(() => api(page, "pending")).toBe(1);
        await api(page, "releaseNext");
        await page.waitForFunction(
          () => {
            const fixture = (
              globalThis as unknown as { gaussianFixture: Fixture }
            ).gaussianFixture;
            return fixture.presentation().placement[12] === -0.6;
          },
          null,
          { timeout: 10000 },
        );
        await api(page, "settle");
        const accepted = await api(page, "presentation");
        expect(accepted.picks).toEqual(["child"]);
        expect(accepted.stats.drawnTileIds).toEqual(["child"]);
        const green = pixel(await pixels(page), 64, 128);
        expect(green[1]).toBeGreaterThan(green[0]! * 1.8);
        expect(green[1]).toBeGreaterThan(green[2]! * 1.8);
        const ledger = await api(page, "ledger");
        expect(ledger.activeOperations).toBe(0);
        expect(ledger.queuedOperations).toBe(0);
        expect(ledger.workReservedBytes).toBe(0);
        expect(await api(page, "errors")).toEqual([]);
        expect(await api(page, "refusals")).toEqual([]);
      } finally {
        await page.close();
      }
    },
  );

  it("cancels a sort waiting for byte admission on disposal", async () => {
    const page = await open("gaussian-only");
    try {
      await api(page, "setupMixed");
      await api(page, "blockWork", "bytes");
      await api(page, "moveMixed");
      await expect
        .poll(async () => (await api(page, "ledger")).queuedOperations)
        .toBe(1);
      await api(page, "dispose");
      await expect
        .poll(async () => (await api(page, "ledger")).activeOperations)
        .toBe(0);
      const ledger = await api(page, "ledger");
      expect(ledger.queuedOperations).toBe(0);
      expect(ledger.workReservedBytes).toBe(0);
      expect(ledger.cpuRetainedBytes).toBe(0);
      expect(ledger.cpuReservedBytes).toBe(0);
      expect(ledger.rendererBytes).toBe(0);
      expect(ledger.owners).toBe(0);
      expect(await api(page, "errors")).toEqual([]);
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

  it.each([false, true])(
    "retries cancelled growth without reusing a claimed lease (sort refusal=%s)",
    async (refusal) => {
      const page = await open("gaussian-only");
      try {
        if (refusal) {
          await api(page, "setWorkLimit", 0);
          await api(page, "growMixed");
          await api(page, "settle");
          expect((await api(page, "stats")).pendingSort).toBe(false);
        } else {
          await api(page, "hold");
          await api(page, "setCamera", [0.1, 0, 10]);
          await expect.poll(() => api(page, "pending")).toBe(1);
        }
        expect(
          await api(page, "stageGrowth", "cancelled", refusal ? 32 : 8),
        ).toBe("queued");
        const reserved = await api(page, "ledger");
        if (!refusal) {
          await api(page, "releaseNext");
          await page.waitForFunction(
            () =>
              !(
                globalThis as unknown as { gaussianFixture: Fixture }
              ).gaussianFixture.stats().pendingSort,
          );
          expect((await api(page, "ledger")).rendererBytes).toBe(
            reserved.rendererBytes,
          );
          expect((await api(page, "ledger")).cpuReservedBytes).toBe(
            reserved.cpuReservedBytes,
          );
        }
        expect(await api(page, "cancelGrowth", "cancelled")).toBe(true);
        await api(page, "settle");
        expect(await api(page, "stageGrowth", "retry", 32, refusal)).toBe(
          "queued",
        );
        await api(page, "release");
        await api(page, "settle");
        expect(pixel(await pixels(page), 128, 128)[0]).toBeGreaterThan(100);
        const accepted = await api(page, "stats");
        expect((await api(page, "ledger")).rendererBytes).toBe(
          accepted.residentBytes,
        );
        await api(page, "selectGrowth", "retry");
        await expect.poll(() => api(page, "pending")).toBe(1);
        expect(pixel(await pixels(page), 128, 128)[0]).toBeGreaterThan(100);
        await api(page, "release");
        await api(page, "settle");
        const blue = pixel(await pixels(page), 128, 128);
        expect(blue[2]).toBeGreaterThan(200);
        expect(blue[0]).toBe(0);
        expect(await api(page, "errors")).toEqual([]);
        await api(page, "dispose");
        await expect
          .poll(() => api(page, "ledger"))
          .toMatchObject({
            rendererBytes: 0,
            cpuRetainedBytes: 0,
            cpuReservedBytes: 0,
            activeOperations: 0,
            queuedOperations: 0,
            workReservedBytes: 0,
            owners: 0,
          });
      } finally {
        await page.close();
      }
    },
  );

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
  it.each([
    { clear: false, prepared: false },
    { clear: true, prepared: false },
    { clear: false, prepared: true },
    { clear: true, prepared: true },
  ])(
    "releases canceled reservations before receiving a camera: %o",
    async ({ clear, prepared }) => {
      const page = await open("no-camera");
      try {
        expect(await api(page, "submitWithoutCamera", prepared)).toBe("queued");
        await api(page, "settle");
        const reserved = await api(page, "ledger");
        expect(reserved.rendererBytes).toBeGreaterThan(0);
        expect(reserved.cpuReservedBytes).toBeGreaterThan(0);
        await api(page, "cancelWithoutCamera", clear);
        await api(page, "settle");
        const released = await api(page, "ledger");
        expect(released).toMatchObject({
          rendererBytes: 0,
          cpuRetainedBytes: 0,
          cpuReservedBytes: 0,
          workReservedBytes: 0,
          activeOperations: 0,
        });
        expect(released.owners).toBeGreaterThan(0);
        expect(pixel(await pixels(page), 128, 128)).toEqual([0, 0, 0]);
        expect(await api(page, "submitWithoutCamera", false)).toBe("queued");
        await api(page, "setCamera", [0, 0, 10]);
        await api(page, "settle");
        await api(page, "drawUnplaced");
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
    },
  );
  it.each([false, true])(
    "propagates grant cancellation and retries pending content (Gaussian=%s)",
    async (gaussian) => {
      const page = await open("gaussian-only");
      try {
        await api(page, "setupDemand", gaussian);
        const before = await pixels(page);
        expect(await api(page, "queueDemand", gaussian)).toBe("queued");
        const pending = await api(page, "demandState");
        const cancelled = await api(page, "reduceDemand");
        await api(page, "settle");
        expect([...cancelled].sort()).toEqual(["left", "right"]);
        expect(pending.required).toBeGreaterThan(
          pending.renderer.residentBytes,
        );
        expect(await api(page, "demandState")).toMatchObject({
          renderer: { pendingTiles: 0, pendingJobs: 0 },
          queue: { queuedJobs: 0 },
        });
        expect((await pixels(page)).data).toEqual(before.data);
        expect(await api(page, "queueDemand", gaussian)).toMatchObject({
          status: "blocked",
        });
        const refused = await api(page, "demandState");
        expect(refused.required).toBeGreaterThan(
          refused.renderer.residentBytes,
        );
        for (let frame = 0; frame < 3; frame++) await api(page, "settle");
        expect((await pixels(page)).data).toEqual(before.data);
        await api(page, "raiseDemand", refused.required);
        expect(await api(page, "queueDemand", gaussian)).toBe("queued");
        await api(page, "settle");
        expect(await api(page, "demandState")).toMatchObject({
          renderer: { pendingTiles: 0, pendingJobs: 0 },
          queue: { queuedJobs: 0 },
        });
        expect((await api(page, "presentation")).picks).toEqual([
          "left",
          "right",
        ]);
        expect(pixel(await pixels(page), 128, 128)[1]).toBeGreaterThan(200);
        expect(await api(page, "errors")).toEqual([]);
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
  it("preserves queued meshes when additional Gaussian capacity cannot fit", async () => {
    const page = await open("gaussian-only");
    try {
      await api(page, "setupDemand", false);
      await api(page, "raiseDemand", 200);
      const before = await pixels(page);
      expect(await api(page, "queueDemand", false)).toBe("queued");
      expect(await api(page, "additionalGaussian")).toMatchObject({
        status: "blocked",
      });
      const pending = await api(page, "demandState");
      expect(pending.renderer.pendingTiles).toBe(2);
      expect(pending.queue.queuedJobs).toBeGreaterThan(0);
      expect((await pixels(page)).data).toEqual(before.data);
      await api(page, "settle");
      expect(await api(page, "demandState")).toMatchObject({
        renderer: { pendingTiles: 0, pendingJobs: 0 },
        queue: { queuedJobs: 0 },
      });
      expect((await api(page, "presentation")).picks).toEqual([
        "left",
        "right",
      ]);
      expect(pixel(await pixels(page), 128, 128)).toEqual([0, 255, 0]);
      const refused = await api(page, "demandState");
      expect(refused.required).toBeGreaterThan(200);
      await api(page, "raiseDemand", refused.required);
      expect(await api(page, "additionalGaussian")).toBe("queued");
      await api(page, "settle");
      expect(await api(page, "demandState")).toMatchObject({
        renderer: { pendingTiles: 0, pendingJobs: 0 },
        queue: { queuedJobs: 0 },
      });
      expect(pixel(await pixels(page), 128, 128)[2]).toBeGreaterThan(100);
      expect(await api(page, "errors")).toEqual([]);
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
  it("notifies canceled outer groups when submission changes the mesh allowance", async () => {
    const page = await open("gaussian-only");
    try {
      await api(page, "setupDemand", false);
      await api(page, "raiseDemand", 200);
      const before = await pixels(page);
      expect(await api(page, "queueDemand", false)).toBe("queued");
      expect(await api(page, "replacementGaussian")).toMatchObject({
        status: "blocked",
      });
      await api(page, "settle");
      const refusals = await api(page, "admissionRefusals");
      expect(refusals).toHaveLength(1);
      expect([...refusals[0]!.ids].sort()).toEqual(["left", "right"]);
      expect(refusals[0]!.message).toContain("ResourcePolicyError");
      expect((await api(page, "demandState")).required).toBeGreaterThan(200);
      expect(await api(page, "demandState")).toMatchObject({
        renderer: { pendingTiles: 0, pendingJobs: 0 },
        queue: { queuedJobs: 0 },
      });
      expect((await pixels(page)).data).toEqual(before.data);
      expect(await api(page, "replacementGaussian")).toBe("queued");
      await api(page, "flushUploads");
      await expect.poll(() => api(page, "pending")).toBe(1);
      expect((await pixels(page)).data).toEqual(before.data);
      expect((await api(page, "presentation")).picks).toEqual(["parent"]);
      await api(page, "release");
      await api(page, "settle");
      expect(await api(page, "demandState")).toMatchObject({
        renderer: { pendingTiles: 0, pendingJobs: 0 },
        queue: { queuedJobs: 0 },
      });
      expect((await api(page, "presentation")).picks).toEqual([]);
      expect(pixel(await pixels(page), 128, 128)[1]).toBeGreaterThan(100);
      expect(await api(page, "errors")).toEqual([]);
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
});
