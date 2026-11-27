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
    "/point-fixture": resolve("test/fixtures"),
    "/": resolve("examples/vtk/dist"),
  });
  browser = await chromium.launch({ headless: true });
});
afterAll(async () => {
  await browser?.close();
  await server?.close();
  if (directory) await rm(directory, { recursive: true, force: true });
});
const open = async (source: "tiles" | "points" = "tiles") => {
  const page = await browser.newPage();
  const workers = new Set<Worker>();
  const failures: string[] = [];
  page.on("worker", (worker) => {
    workers.add(worker);
    worker.on("close", () => workers.delete(worker));
  });
  page.on("pageerror", (error) => failures.push(error.message));
  const url = new URL(server.origin);
  url.searchParams.set("harness", "1");
  if (source === "points")
    url.searchParams.set(
      "url",
      `${server.origin}/point-fixture/fixture.copc.laz`,
    );
  else url.searchParams.set("tiles", `${server.origin}/fixture/tileset.json`);
  await page.goto(url.href);
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
it("shows a failed Gaussian choice and lets the user retry from the point cloud", async () => {
  const { page, failures } = await open("points");
  const held = gate();
  const dropdown = page.locator(".dataset-source-select");
  const status = page.locator(".dataset-switch-status");
  const label = "Jastrzębia Góra, Poland · 103.5M splats";
  try {
    const original = await dropdown.inputValue();
    expect(original).toContain("fixture.copc.laz");
    await page.route(
      "**/gaussian-city/tileset.json",
      async (route) => {
        held.started();
        await held.waiting;
        await route.fulfill({ status: 404 });
      },
      { times: 1 },
    );
    await dropdown.selectOption({ label });
    await held.ready;
    expect(await dropdown.locator("option:checked").textContent()).toBe(label);
    expect(await status.textContent()).toBe("Loading selected dataset…");
    held.release();
    await expect.poll(() => status.textContent()).toContain("HTTP 404");
    expect(await dropdown.inputValue()).toBe(original);
    await page.route("**/gaussian-city/tileset.json", (route) =>
      route.fulfill({ json: manifest(6378137) }),
    );
    await dropdown.selectOption({ label });
    await page.getByRole("combobox", { name: `Change ${label}` }).waitFor();
    expect(await page.locator("#dataset-list > .dataset-card").count()).toBe(1);
    expect(new URL(page.url()).searchParams.get("tiles")).toContain(
      "JastrebiaGora",
    );
    expect(failures).toEqual([]);
  } finally {
    held.release();
    await page.close();
  }
});

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
