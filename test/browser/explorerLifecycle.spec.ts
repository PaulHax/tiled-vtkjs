import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { chromium, type Browser, type Page, type Worker } from "playwright";
import { afterAll, beforeAll, expect, it } from "vitest";
import { startStaticServer, type StaticServer } from "./server";

let browser: Browser;
let server: StaticServer;
let directory: string;
const manifest = (x = 0) => ({
  asset: { version: "1.1" },
  geometricError: 0,
  root: {
    refine: "REPLACE",
    geometricError: 0,
    boundingVolume: { box: [x, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1] },
  },
});
beforeAll(async () => {
  directory = await mkdtemp(resolve(tmpdir(), "explorer-lifecycle-"));
  for (const name of ["tileset", "replacement"])
    await writeFile(
      resolve(directory, `${name}.json`),
      JSON.stringify(manifest()),
    );
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
const open = async () => {
  const page = await browser.newPage();
  const workers = new Set<Worker>();
  const failures: string[] = [];
  page.on("worker", (worker) => {
    workers.add(worker);
    worker.on("close", () => workers.delete(worker));
  });
  page.on("pageerror", (error) => failures.push(error.message));
  await page.goto(
    `${server.origin}/?harness=1&tiles=${encodeURIComponent(`${server.origin}/fixture/tileset.json`)}`,
  );
  await page.locator("#dataset-list .dataset-heading select").waitFor();
  await page.waitForFunction(() => "pointCloudScene" in globalThis);
  return { page, workers, failures };
};
it("preserves persisted pages and tears down on the subsequent real pagehide", async () => {
  const { page, workers, failures } = await open();
  try {
    expect(await page.locator("#viewer canvas").count()).toBe(1);
    await page.evaluate(() => {
      window.dispatchEvent(
        new PageTransitionEvent("pagehide", { persisted: true }),
      );
      window.dispatchEvent(
        new PageTransitionEvent("pageshow", { persisted: true }),
      );
    });
    expect(await page.locator("#viewer canvas").count()).toBe(1);
    await page.locator("#open-add-dataset").click();
    expect(
      await page
        .locator("#add-dataset-dialog")
        .evaluate((element) => (element as HTMLDialogElement).open),
    ).toBe(true);
    await page.evaluate(() =>
      window.dispatchEvent(
        new PageTransitionEvent("pagehide", { persisted: false }),
      ),
    );
    await expect.poll(() => page.locator("#viewer canvas").count()).toBe(0);
    await expect.poll(() => workers.size).toBe(0);
    await page.evaluate(() =>
      window.dispatchEvent(
        new PageTransitionEvent("pagehide", { persisted: false }),
      ),
    );
    expect(failures).toEqual([]);
  } finally {
    await page.close();
  }
});

const gate = () => {
  let release!: () => void, started!: () => void;
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
const custom = async (
  page: Page,
  kind: "tiles" | "points",
  url: string,
  name = "",
) => {
  await page
    .locator("#dataset-list .dataset-heading select")
    .selectOption("custom");
  await page.locator("#dataset-type").selectOption(kind);
  await page.locator("#source-type").selectOption("url");
  await page.locator("#dataset-url").fill(url);
  if (name) await page.locator("#dataset-name").fill(name);
  await page.locator("#submit-dataset").click();
};

for (const kind of ["tiles", "points"] as const) {
  for (const action of ["remove", "replace"] as const)
    it(`${action} invalidates pending ${kind} source preparation`, async () => {
      const { page, workers, failures } = await open();
      const held = gate();
      const bytes =
        kind === "points"
          ? await readFile(resolve("test/fixtures/fixture.copc.laz"))
          : Buffer.from(JSON.stringify(manifest(6378137)));
      await page.route(
        kind === "tiles" ? "**/gaussian-city/tileset.json" : "**/held.copc.laz",
        async (route) => {
          held.started();
          await held.waiting;
          const range = route
            .request()
            .headers()
            .range?.match(/bytes=(\d+)-(\d+)/);
          const start = range ? Number(range[1]) : 0,
            end = range
              ? Math.min(Number(range[2]), bytes.length - 1)
              : bytes.length - 1;
          await route
            .fulfill({
              status: range ? 206 : 200,
              headers: {
                "content-type":
                  kind === "tiles"
                    ? "application/json"
                    : "application/octet-stream",
                ...(range
                  ? { "content-range": `bytes ${start}-${end}/${bytes.length}` }
                  : {}),
              },
              body: bytes.subarray(start, end + 1),
            })
            .catch(() => {});
        },
      );
      try {
        if (kind === "tiles")
          await page
            .locator("#dataset-list .dataset-heading select")
            .selectOption({ label: "Jastrzębia Góra, Poland · 103.5M splats" });
        else await custom(page, "points", `${server.origin}/held.copc.laz`);
        await held.ready;
        if (kind === "points") await page.locator("#close-add-dataset").click();
        if (action === "remove")
          await page.locator('#dataset-list [data-role="remove"]').click();
        else
          await custom(
            page,
            "tiles",
            `${server.origin}/fixture/replacement.json`,
            "Latest dataset",
          );
        held.release();
        await expect
          .poll(() =>
            page.evaluate(
              () =>
                (
                  globalThis as unknown as {
                    pointCloudScene: { stats(): { loading: number } };
                  }
                ).pointCloudScene.stats().loading,
            ),
          )
          .toBe(0);
        const cards = page.locator("#dataset-list > .dataset-card");
        await expect
          .poll(() => cards.count())
          .toBe(action === "remove" ? 0 : 1);
        await expect
          .poll(
            () =>
              [...workers].filter((worker) =>
                worker.url().includes("copc.worker"),
              ).length,
          )
          .toBe(0);
        await expect
          .poll(() => page.locator("#message").textContent())
          .not.toContain("insertBefore");
        if (action === "replace")
          expect(
            await cards
              .locator(".dataset-heading select")
              .getAttribute("aria-label"),
          ).toBe("Change Latest dataset");
        expect(failures).toEqual([]);
      } finally {
        held.release();
        await page.close();
      }
    });
}
