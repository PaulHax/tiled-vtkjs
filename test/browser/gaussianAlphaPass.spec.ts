import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { chromium } from "playwright";
import { build } from "vite";
import { expect, it } from "vitest";
import { startStaticServer } from "./server";

it("preserves Gaussian, ordinary mesh and ordinary point opacity across viewports", async () => {
  const directory = await mkdtemp(resolve(tmpdir(), "gaussian-alpha-pass-"));
  const browser = await chromium.launch({
    headless: true,
    channel: "chromium",
  });
  let server: Awaited<ReturnType<typeof startStaticServer>> | undefined;
  try {
    await build({
      configFile: false,
      logLevel: "silent",
      build: {
        outDir: directory,
        lib: {
          entry: resolve("test/browser/gaussianAlphaPassPage.mjs"),
          fileName: "fixture",
          formats: ["es"],
        },
      },
    });
    await writeFile(
      resolve(directory, "index.html"),
      '<!doctype html><style>html,body,#view{margin:0;width:160px;height:80px;overflow:hidden}</style><div id="view"></div><script type="module" src="fixture.js"></script>',
    );
    server = await startStaticServer({ "/fixture": directory });
    const page = await browser.newPage({
      viewport: { width: 160, height: 80 },
    });
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(server.origin + "/fixture/index.html");
    await page.locator('html[data-ready="true"]').waitFor();
    const pixels = await page.evaluate(() => {
      const fixture = (
        window as unknown as {
          fixture: {
            capture(): Record<string, number[]>;
            dispose(): void;
          };
        }
      ).fixture;
      const result = fixture.capture();
      fixture.dispose();
      return result;
    });
    expect(errors).toEqual([]);
    expect(pixels.gaussian![0]).toBeGreaterThan(120);
    expect(pixels.gaussian![0]).toBeLessThan(136);
    expect(pixels.mesh).toEqual([0, 0, 128, 255]);
    expect(pixels.point).toEqual([0, 128, 0, 255]);
  } finally {
    await browser.close();
    await server?.close();
    await rm(directory, { recursive: true, force: true });
  }
}, 60_000);
