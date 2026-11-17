import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { chromium, type Browser } from "playwright";
import { build } from "vite";
import { afterAll, beforeAll, expect, it } from "vitest";
import { startStaticServer, type StaticServer } from "./server";

let browser: Browser,
  server: StaticServer,
  directory = "";
beforeAll(async () => {
  directory = await mkdtemp(resolve(tmpdir(), "gaussian-partial-replacement-"));
  await build({
    configFile: false,
    logLevel: "silent",
    build: {
      outDir: directory,
      lib: {
        entry: resolve("test/browser/gaussianPartialReplacementPage.mjs"),
        fileName: "fixture",
        formats: ["es"],
      },
    },
  });
  await writeFile(
    resolve(directory, "index.html"),
    '<!doctype html><style>html,body,#view{margin:0;width:256px;height:256px;overflow:hidden}</style><div id="view"></div><script type="module" src="fixture.js"></script>',
  );
  server = await startStaticServer({ "/fixture": directory });
  browser = await chromium.launch({ headless: true, channel: "chromium" });
});
afterAll(async () => {
  await browser?.close();
  await server?.close();
  if (directory) await rm(directory, { recursive: true, force: true });
});
type Snapshot = {
  drawn: string[];
  submitted: string[];
  splats: number;
  grant: number;
  gpuBytes: number;
  picks: string[];
  hits: (string | null)[];
  translation: number;
  a: number[];
  b: number[];
  meshA: number[];
  meshB: number[];
};
type Ledger = {
  rendererBytes: number;
  totalBytes: number;
  transitionBytes: number;
  cpuRetainedBytes: number;
  cpuReservedBytes: number;
  cpuBytes: number;
  workReservedBytes: number;
  workBytes: number;
  activeOperations: number;
  queuedOperations: number;
  owners: number;
};
type Fixture = {
  setup(mixed: boolean): Promise<{
    old: Snapshot;
    first: Snapshot & { state: string };
    second: { status: string; reason?: string };
    waiting: Snapshot;
  }>;
  selectPartial(): Promise<Snapshot>;
  release(): Promise<Snapshot & { errors: string[]; ledgers: Ledger[] }>;
  dispose(): Promise<Ledger & { actors: number }>;
};
const sorted = (ids: string[]) => [...ids].sort();
it.each([false, true])(
  "publishes a compact partial replacement without a coverage hole (mixed=%s)",
  async (mixed) => {
    const page = await browser.newPage({
      viewport: { width: 256, height: 256 },
    });
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    let cleanup: Awaited<ReturnType<Fixture["dispose"]>> | undefined;
    try {
      await page.goto(server.origin + "/fixture/index.html");
      await page.locator('html[data-ready="true"]').waitFor();
      const initial = await page.evaluate(
        (mixed) =>
          (window as unknown as { fixture: Fixture }).fixture.setup(mixed),
        mixed,
      );
      for (const stage of [initial.old, initial.first, initial.waiting]) {
        expect(sorted(stage.drawn)).toEqual(["oldA", "oldB"]);
        expect(stage.splats).toBe(5);
        expect(stage.grant).toBe(mixed ? 636 : 480);
        expect(stage.gpuBytes).toBeLessThanOrEqual(480);
        expect(stage.a[0]).toBeGreaterThan(150);
        expect(stage.b[1]).toBeGreaterThan(220);
      }
      expect(initial.first.state).toBe("submitted");
      expect(initial.second).toMatchObject({
        status: "blocked",
        reason: "transition-unavailable",
      });
      const held = await page.evaluate(() =>
        (window as unknown as { fixture: Fixture }).fixture.selectPartial(),
      );
      expect(held.drawn).toContain("oldB");
      expect(held.drawn.some((id) => id === "oldA" || id === "newA")).toBe(
        true,
      );
      expect(held.a[0]! + held.a[2]!).toBeGreaterThan(150);
      expect(held.b[1]).toBeGreaterThan(220);
      expect(held.translation).toBe(0);
      if (mixed) {
        expect(sorted(held.picks)).toEqual(sorted(held.drawn));
        expect(held.hits).toEqual(["hit", "hit"]);
        expect(held.meshA[0]! + held.meshA[2]!).toBe(255);
        expect(held.meshB).toEqual([0, 255, 0, 255]);
      }
      const final = await page.evaluate(() =>
        (window as unknown as { fixture: Fixture }).fixture.release(),
      );
      expect(
        final.a[2],
        JSON.stringify({
          drawn: final.drawn,
          splats: final.splats,
          translation: final.translation,
          a: final.a,
          b: final.b,
          picks: final.picks,
          hits: final.hits,
        }),
      ).toBeGreaterThan(220);
      expect(sorted(final.drawn)).toEqual(["newA", "oldB"]);
      expect(final.splats).toBe(8);
      expect(final.gpuBytes).toBeLessThanOrEqual(480);
      expect(final.translation).toBe(0.3);
      expect(final.a[0]).toBe(0);
      expect(final.b[1]).toBeGreaterThan(220);
      expect(final.b[0]).toBe(0);
      expect(sorted(final.picks)).toEqual(mixed ? ["newA", "oldB"] : []);
      if (mixed) {
        expect(final.hits).toEqual(["hit", "hit"]);
        expect(final.meshA).toEqual([0, 0, 255, 255]);
        expect(final.meshB).toEqual([0, 255, 0, 255]);
      }
      expect(final.errors).toEqual([]);
      for (const ledger of final.ledgers) {
        expect(ledger.rendererBytes).toBeLessThanOrEqual(
          ledger.totalBytes + ledger.transitionBytes,
        );
        expect(
          ledger.cpuRetainedBytes + ledger.cpuReservedBytes,
        ).toBeLessThanOrEqual(ledger.cpuBytes);
        expect(ledger.workReservedBytes).toBeLessThanOrEqual(ledger.workBytes);
      }
    } finally {
      cleanup = await page.evaluate(() =>
        (window as unknown as { fixture: Fixture }).fixture.dispose(),
      );
      await page.close();
    }
    expect(errors).toEqual([]);
    expect(cleanup).toMatchObject({
      rendererBytes: 0,
      cpuRetainedBytes: 0,
      cpuReservedBytes: 0,
      workReservedBytes: 0,
      activeOperations: 0,
      queuedOperations: 0,
      owners: 0,
      actors: 0,
    });
  },
);
