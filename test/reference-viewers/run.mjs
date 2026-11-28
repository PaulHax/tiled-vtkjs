import { mkdir, readFile, readdir, rename, writeFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { chromium } from "playwright";
import { createServer } from "vite";
import sharp from "sharp";
import { startStaticServer } from "../browser/server.ts";
import { isSoftwareRenderer } from "../../examples/vtk/softwareRenderer.ts";
import { hashAssetCorpus } from "./asset-corpus.mjs";
import { initialBenchmarkPose } from "./driver.mjs";
import { createFixtures } from "./fixtures.mjs";
import { foregroundFraction } from "./visual.mjs";
import { launchHeadlessGpu } from "./headless-gpu.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../..");
const { values } = parseArgs({
  options: {
    manifest: { type: "string" },
    case: { type: "string", multiple: true },
    out: { type: "string" },
    "vtk-dist": { type: "string" },
    gpu: { type: "boolean", default: false },
    coverage: { type: "boolean", default: false },
    checkpoints: { type: "boolean", default: false },
    trace: { type: "boolean", default: false },
    "settled-reference": { type: "boolean", default: false },
    "prepare-assets": { type: "boolean", default: false },
    repeats: { type: "string", default: "1" },
    latency: { type: "string", default: "0" },
    mbps: { type: "string" },
  },
});
const settledReference = values["settled-reference"];
const preparingAssets = values["prepare-assets"];
const pass = preparingAssets
  ? "asset-discovery"
  : settledReference
    ? "settled-reference"
    : values.checkpoints
      ? "checkpoints"
      : values.coverage
        ? "coverage"
        : values.gpu
          ? "hardware"
          : "smoke";
const output = resolve(
  repo,
  values.out ?? "artifacts/reference-viewers/" + pass,
);
if (settledReference && !values.checkpoints)
  throw new Error("Settled references require --coverage --checkpoints");
if (values.checkpoints && !values.coverage)
  throw new Error(
    "Checkpoint capture requires --coverage and is not a timing run",
  );
if (preparingAssets && values.checkpoints)
  throw new Error("Quality checkpoints require a frozen local asset corpus");
const repeats = Number(values.repeats);
if (!Number.isSafeInteger(repeats) || repeats < 1)
  throw new Error("--repeats must be a positive integer");
await mkdir(output, { recursive: true });
const manifestPath = values.manifest && resolve(repo, values.manifest);
const manifest = manifestPath
  ? JSON.parse(await readFile(manifestPath, "utf8"))
  : await createFixtures(resolve(output, "fixtures"), here);
if (!manifestPath)
  await writeFile(
    resolve(output, "manifest.json"),
    JSON.stringify(manifest, null, 2) + "\n",
  );
const vtkDist = resolve(
  repo,
  values["vtk-dist"] ?? resolve(repo, "examples/vtk/dist"),
);
const assetRoot = resolve(
  manifestPath ? dirname(manifestPath) : output,
  manifest.assetRoot,
);

const source = manifest.assetSource;
if (preparingAssets && !source)
  throw new Error("Asset preparation needs a manifest source");
const downloads = new Map();
const populate = preparingAssets
  ? async (path, file) => {
      const prefix = "/data/" + source.directory + "/";
      if (!path.startsWith(prefix)) return;
      const relative = path.slice(prefix.length);
      if (!/^tiles\/(?:\d+\/)+\d+\.glb$/.test(relative)) return;
      let pending = downloads.get(file);
      if (!pending) {
        pending = (async () => {
          const url = new URL(relative, source.url);
          let bytes;
          for (let attempt = 0; attempt < 3; attempt++) {
            try {
              const response = await fetch(url, {
                signal: AbortSignal.timeout(30000),
              });
              if (!response.ok)
                throw new Error("Asset HTTP " + response.status);
              bytes = Buffer.from(await response.arrayBuffer());
              break;
            } catch (error) {
              if (attempt === 2)
                throw new Error("Asset download failed " + relative, {
                  cause: error,
                });
            }
          }
          if (
            bytes.length < 12 ||
            bytes.toString("ascii", 0, 4) !== "glTF" ||
            bytes.readUInt32LE(8) !== bytes.length
          )
            throw new Error("Invalid GLB " + relative);
          await mkdir(dirname(file), { recursive: true });
          await writeFile(file + ".part", bytes);
          await rename(file + ".part", file);
        })();
        downloads.set(file, pending);
      }
      try {
        await pending;
      } catch (error) {
        console.error(error.message);
        throw error;
      } finally {
        downloads.delete(file);
      }
    }
  : undefined;

const builtInTraces = {
  approach: [
    [0, 0, 0],
    [550, 1, 0],
    [1500, 1, 0],
  ],
  reverse: [
    [0, 1, 0],
    [350, 0, 0],
    [950, 0, 0],
    [1300, 1, 0],
    [2100, 1, 0],
  ],
  boundary: [
    [0, 1, 0],
    [250, 0.5, 0],
    [490, 0.53, 0],
    [730, 0.47, 0],
    [970, 0.53, 0],
    [1210, 0.47, 0],
    [1450, 0.53, 0],
    [1700, 0.5, 0],
  ],
  orbit: [
    [0, 0.5, 0],
    [400, 0.5, 0.45],
    [800, 0.5, -0.45],
    [1250, 0.5, 0],
    [1700, 0.5, 0],
  ],
};
const tests = manifest.cases.filter(
  ({ id }) => !values.case || values.case.includes(id),
);
if (!tests.length) throw new Error("No matching benchmark cases");
const slugs = new Set();
for (const test of tests) {
  if (!/^[a-z0-9-]+$/.test(test.id) || slugs.has(test.id))
    throw new Error("Invalid or duplicate case ID: " + test.id);
  slugs.add(test.id);
  if (!test.asset || /^(?:[a-z]+:|\/|.*\.\.)/i.test(test.asset))
    throw new Error("Use a relative local asset path in " + test.id);
}
for (const id of values.case ?? []) {
  if (!slugs.has(id)) throw new Error("Unknown benchmark case: " + id);
}
if (
  !Number.isFinite(Number(values.latency)) ||
  Number(values.latency) < 0 ||
  !Number.isFinite(Number(values.mbps ?? 80)) ||
  Number(values.mbps ?? 80) <= 0
)
  throw new Error("--latency must be nonnegative and --mbps must be positive");
const mbps = Number(values.mbps ?? 80);
const profiles =
  Number(values.latency) > 0 || values.mbps !== undefined
    ? {
        paths: (
          await readdir(assetRoot, { recursive: true, withFileTypes: true })
        )
          .filter((entry) => entry.isFile())
          .map(
            (entry) =>
              "/data/" +
              resolve(entry.parentPath, entry.name).slice(assetRoot.length + 1),
          ),
        latencyMs: Number(values.latency),
        bytesPerSecond: mbps * 125_000,
      }
    : undefined;

const assetCorpusSha256 = values.checkpoints
  ? await hashAssetCorpus(assetRoot)
  : undefined;

let fileServer;
let server;
let browser;
let gpuSession;
const results = [];
const percentile = (values, p) =>
  values.toSorted((a, b) => a - b)[
    Math.max(0, Math.ceil(values.length * p) - 1)
  ] ?? null;
const turnover = (samples, key) => {
  let replacements = 0;
  let tileAdds = 0;
  let tileDrops = 0;
  let tileReturns = 0;
  let previous;
  const dropped = new Set();
  for (const sample of samples) {
    if (!sample[key]) continue;
    const selected = new Set(sample[key]);
    if (previous) {
      const added = [...selected].filter((id) => !previous.has(id));
      const removed = [...previous].filter((id) => !selected.has(id));
      if (added.length || removed.length) replacements++;
      tileAdds += added.length;
      tileDrops += removed.length;
      for (const id of added) if (dropped.delete(id)) tileReturns++;
      for (const id of removed) dropped.add(id);
    }
    previous = selected;
  }
  return previous
    ? {
        changes: replacements,
        adds: tileAdds,
        drops: tileDrops,
        returns: tileReturns,
      }
    : null;
};
const summary = (samples) => {
  const frame = samples.map(({ intervalMs }) => intervalMs);
  const visible = turnover(samples, "selected");
  const ready = samples.find(({ ready }) => ready);
  const coverage = samples.flatMap(({ foregroundFraction }) =>
    foregroundFraction === undefined ? [] : [foregroundFraction],
  );
  const peak = (key) => {
    const values = samples.map((sample) => sample[key]).filter(Number.isFinite);
    return values.length ? Math.max(...values) : null;
  };
  return {
    frameCount: frame.length,
    rafP95Ms: percentile(frame, 0.95),
    rafP99Ms: percentile(frame, 0.99),
    excessFrameTimeMs: Object.fromEntries(
      [16.7, 33.3].map((budget) => [
        budget,
        frame.reduce(
          (total, interval) => total + Math.max(0, interval - budget),
          0,
        ),
      ]),
    ),
    framesOver20Ms: frame.filter((ms) => ms > 20).length,
    hostWorkP95Ms: percentile(
      samples.map(({ mainThreadMs }) => mainThreadMs),
      0.95,
    ),
    viewerReadyAtEnd: samples.at(-1)?.ready ?? null,
    viewerReportedUnreadyFrames: samples.filter(({ ready }) => !ready).length,
    selectionChanges: visible?.changes ?? null,
    tileAdds: visible?.adds ?? null,
    tileDrops: visible?.drops ?? null,
    tileReturns: visible?.returns ?? null,
    desiredTurnover: turnover(samples, "desired"),
    requestedTurnover: turnover(samples, "requested"),
    peakResidentBytes: peak("residentBytes"),
    residentLimitBytes: peak("residentLimitBytes"),
    peakCpuRetainedBytes: peak("cpuRetainedBytes"),
    peakWorkReservedBytes: peak("workReservedBytes"),
    peakBlockedGroups: peak("blockedGroups"),
    peakEffectiveSse: peak("effectiveSse"),
    peakCacheEvictions: peak("cacheEvictions"),
    peakUploadBytes: peak("cumulativeUploadBytes"),
    sampledPressureMs: samples.some(
      (sample) => typeof sample.pressure === "boolean",
    )
      ? samples.reduce(
          (total, sample) => total + (sample.pressure ? sample.intervalMs : 0),
          0,
        )
      : null,
    pressureFrames: samples.some(
      (sample) => typeof sample.pressure === "boolean",
    )
      ? samples.filter((sample) => sample.pressure).length
      : null,
    endTileCount: samples.at(-1)?.selected?.length ?? null,
    firstReadyMs: ready?.elapsed ?? null,
    minimumForegroundFraction: coverage.length ? Math.min(...coverage) : null,
    blankCanvasSamples: coverage.length
      ? coverage.filter((fraction) => fraction < 0.001).length
      : null,
  };
};

try {
  fileServer = await startStaticServer(
    {
      "/data": assetRoot,
      "/cesium": resolve(here, "node_modules/cesium/Build/Cesium"),
      "/three-basis": resolve(
        here,
        "node_modules/three/examples/jsm/libs/basis",
      ),
      "/maplibre": resolve(here, "node_modules/maplibre-gl/dist"),
      "/": vtkDist,
    },
    {},
    profiles,
    populate,
  );
  const proxy = Object.fromEntries(
    [
      "/data",
      "/cesium",
      "/maplibre",
      "/three-basis",
      "/combined",
      "/mesh",
      "/assets",
      "/tiles3d-codecs",
      "/tiles3d-decode-worker.js",
      "/laz-perf.wasm",
    ].map((prefix) => [prefix, fileServer.origin]),
  );
  server = await createServer({
    root: here,
    configFile: false,
    cacheDir: resolve(here, "node_modules/.vite"),
    optimizeDeps: {
      noDiscovery: true,
      include: [
        "ol/Map.js",
        "ol/View.js",
        "ol/source/GeoTIFF.js",
        "ol/layer/WebGLTile.js",
        "ol/proj.js",
        "three/addons/loaders/GLTFLoader.js",
        "three/addons/loaders/DRACOLoader.js",
        "three/addons/loaders/KTX2Loader.js",
        "3d-tiles-renderer/plugins",
        ...Object.keys(
          JSON.parse(await readFile(resolve(here, "package.json"), "utf8"))
            .dependencies,
        ).filter((name) => name !== "cesium" && name !== "ol"),
      ],
    },
    server: { host: "127.0.0.1", port: 0, hmr: false, watch: null, proxy },
  });
  await server.listen();
  const origin = server.resolvedUrls.local[0];
  if (values.gpu) {
    gpuSession = await launchHeadlessGpu(chromium);
    browser = gpuSession.browser;
  } else {
    browser = await chromium.launch({
      headless: true,
      args: [
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--disable-dev-shm-usage",
      ],
    });
  }
  for (let run = 0; run < repeats; run++) {
    for (const test of tests) {
      const context = await browser.newContext({
        viewport: { width: 960, height: 720 },
        deviceScaleFactor: 1,
      });
      const page = await context.newPage();
      const errors = [];
      const requests = [];
      let phase = "initial";
      page.on("pageerror", (error) => errors.push(error.message));
      const consoleErrors = [];
      const graphicsWarnings = new Set();
      page.on("console", (message) => {
        if (/^WebGL:/i.test(message.text()))
          graphicsWarnings.add(message.text());
        if (message.type() !== "error") return;
        void Promise.all(
          message
            .args()
            .map((arg) =>
              arg.evaluate((value) =>
                value && typeof value === "object" && "message" in value
                  ? value.message
                  : null,
              ),
            ),
        ).then(
          (details) =>
            consoleErrors.push(
              [message.text(), ...details.filter(Boolean)].join(" "),
            ),
          () => consoleErrors.push(message.text()),
        );
      });
      const requestPhases = new WeakMap();
      const wanted = (request) =>
        new URL(request.url()).pathname.startsWith("/data/");
      page.on("request", (request) => {
        if (wanted(request)) requestPhases.set(request, phase);
      });
      page.on("response", (response) => {
        if (!new URL(response.url()).pathname.startsWith("/data/")) return;
        requests.push({
          phase: requestPhases.get(response.request()) ?? phase,
          path: new URL(response.url()).pathname,
          range: response.request().headers().range,
          status: response.status(),
          bytes: Number(response.headers()["content-length"]) || null,
        });
      });
      page.on("requestfailed", (request) => {
        if (new URL(request.url()).pathname.startsWith("/data/"))
          requests.push({
            phase: requestPhases.get(request) ?? phase,
            path: new URL(request.url()).pathname,
            range: request.headers().range,
            failed: request.failure()?.errorText,
          });
      });
      await page.addInitScript((coverage) => {
        const contexts = new WeakMap();
        globalThis.benchmarkContexts = contexts;
        const original = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function (type, options) {
          const webgl = /^(webgl2?|experimental-webgl)$/.test(type);
          const context = original.call(
            this,
            type,
            webgl && coverage
              ? { ...options, preserveDrawingBuffer: true }
              : options,
          );
          if (context && webgl) contexts.set(this, context);
          return context;
        };
      }, values.coverage);
      const name =
        test.id +
        "-" +
        (run + 1) +
        (settledReference ? "-settled-reference" : "");
      const record = {
        id: test.id,
        format: test.format,
        viewer: test.viewer,
        workload: test.workload,
        asset: test.asset,
        options: test,
        ...(test.viewer === "tiled-vtkjs" ? { vtkDist } : {}),
        camera: test.camera,
        run: run + 1,
        chromium: browser.version(),
        timingMode: preparingAssets
          ? "asset discovery"
          : settledReference
            ? "settled quality reference"
            : values.coverage
              ? "coverage readback"
              : values.gpu
                ? "headless hardware"
                : "headless smoke only",
        phases: {},
        requests,
        errors,
        consoleErrors,
        graphicsWarnings: [],
      };
      try {
        const url = new URL("data/" + test.asset, origin).href;
        const vtk = test.viewer === "tiled-vtkjs";
        const background = test.background ?? (vtk ? [20, 23, 28] : [0, 0, 0]);
        if (values.checkpoints) {
          record.clearColor = background;
          record.assetCorpusSha256 = assetCorpusSha256;
        }
        const entry = new URL(
          vtk ? "combined/index.html" : "index.html",
          origin,
        );
        if (vtk) {
          entry.searchParams.set("harness", "1");
          if (values.trace) entry.searchParams.set("benchmarkTrace", "1");
          if (settledReference && test.referenceSplatLimit !== undefined)
            entry.searchParams.set(
              "referenceSplatLimit",
              String(test.referenceSplatLimit),
            );
          entry.searchParams.set(
            "benchmarkCamera",
            JSON.stringify(initialBenchmarkPose(test)),
          );
          if (test.sse !== undefined)
            entry.searchParams.set("benchmarkSse", String(test.sse));
          if (test.memoryMiB !== undefined)
            entry.searchParams.set(
              "benchmarkMemoryMiB",
              String(test.memoryMiB),
            );
          if (test.maxConcurrentOperations !== undefined)
            entry.searchParams.set(
              "benchmarkOperations",
              String(test.maxConcurrentOperations),
            );
          entry.searchParams.set(
            "data",
            (test.format === "copc" ? "points-url:" : "tiles-url:") + url,
          );
        }
        const response = await page.goto(entry.href);
        if (!response.ok())
          throw new Error(
            "Viewer entry failed: " + response.status() + " " + entry.href,
          );
        if (vtk) {
          await page.addStyleTag({
            content:
              "#viewer{position:fixed!important;inset:0!important;width:100vw!important;height:100vh!important;z-index:9999!important}",
          });
          await page.waitForFunction(
            () => globalThis.pointCloudScene?.datasets().length === 1,
          );
          await page.evaluate(() => dispatchEvent(new Event("resize")));
          await page.waitForFunction(() => {
            const canvas = document.querySelector("#viewer canvas");
            return (
              globalThis.pointCloudScene?.datasets().length === 1 &&
              canvas?.width === innerWidth &&
              canvas.height === innerHeight
            );
          });
        }
        await page.evaluate(
          async (config) => {
            const module = await import("/driver.mjs");
            globalThis.benchmarkInitError = null;
            void module.initialize(config).catch((error) => {
              globalThis.benchmarkInitError = String(error.stack ?? error);
            });
          },
          {
            ...test,
            url,
            background,
            coverage: values.coverage,
            checkpoints: values.checkpoints,
            settledReference,
            trace: values.trace,
          },
        );
        await page.waitForFunction(
          () => globalThis.referenceBenchmark || globalThis.benchmarkInitError,
        );
        const environment = await page.evaluate(() => {
          if (globalThis.benchmarkInitError)
            throw new Error(globalThis.benchmarkInitError);
          return {
            gpu: globalThis.referenceBenchmark.gpu,
            canvas: globalThis.referenceBenchmark.canvas,
            initialSettledMs: globalThis.referenceBenchmark.initialSettledMs,
            navigationToReadyMs:
              globalThis.referenceBenchmark.navigationToReadyMs,
          };
        });
        Object.assign(record, environment);
        const viewport = page.viewportSize();
        if (
          environment.canvas.width !== viewport.width ||
          environment.canvas.height !== viewport.height
        )
          throw new Error(
            "Drawing buffer " +
              JSON.stringify(environment.canvas) +
              " does not match the benchmark viewport " +
              JSON.stringify(viewport),
          );
        if (
          values.gpu &&
          (!environment.gpu || isSoftwareRenderer(environment.gpu))
        )
          throw new Error(
            "Hardware timing requires a verified WebGL renderer, got " +
              environment.gpu,
          );

        const screenshot = async (label) => {
          const path = resolve(output, name + "-" + label + ".png");
          const encoded = await page
            .locator("canvas[data-benchmark-canvas]")
            .screenshot({ path });
          const { data, info } = await sharp(encoded)
            .removeAlpha()
            .raw()
            .toBuffer({ resolveWithObject: true });
          return {
            file: basename(path),
            foregroundFraction: foregroundFraction(
              data,
              info.channels,
              background,
            ),
          };
        };
        if (!test.route || values.coverage)
          record.initial = await screenshot("initial");
        if (values.checkpoints && !test.route)
          throw new Error("Checkpoint capture requires a recorded route");
        if (test.route) {
          phase = "route";
          const frames = settledReference
            ? (await page.evaluate(() =>
                globalThis.referenceBenchmark.settleReference(),
              ),
              [])
            : await page.evaluate(
                (path) => globalThis.referenceBenchmark.run(path),
                test.route.keyframes,
              );
          if (values.checkpoints) {
            const images = await page.evaluate(() =>
              globalThis.referenceBenchmark.takeCheckpoints(),
            );
            record.checkpoints = [];
            record.recovery = [];
            for (const { image, ...checkpoint } of images) {
              const file = checkpoint.recovery
                ? name +
                  "-recovery-" +
                  checkpoint.label +
                  "-" +
                  checkpoint.atMs +
                  ".png"
                : name + "-checkpoint-" + checkpoint.label + ".png";
              const encoded = Buffer.from(image.split(",")[1], "base64");
              await writeFile(resolve(output, file), encoded);
              if (settledReference) {
                const { data, info } = await sharp(encoded)
                  .removeAlpha()
                  .raw()
                  .toBuffer({ resolveWithObject: true });
                checkpoint.foregroundFraction = foregroundFraction(
                  data,
                  info.channels,
                  background,
                );
              }
              const target = checkpoint.recovery
                ? record.recovery
                : record.checkpoints;
              target.push({ ...checkpoint, file });
            }
            if (record.checkpoints.length !== test.route.checkpoints.length)
              throw new Error("Missing scheduled quality checkpoints");
          }
          if (!settledReference) {
            record.phases.route = {
              ...summary(frames),
              ...(await screenshot("route-final")),
              samples: frames,
            };
            record.routePhases = Object.fromEntries(
              test.route.phases.map(([start, label], index) => {
                const end = test.route.phases[index + 1]?.[0] ?? Infinity;
                return [
                  label,
                  summary(
                    frames.filter(
                      ({ elapsed }) => elapsed >= start && elapsed < end,
                    ),
                  ),
                ];
              }),
            );
          }
        } else {
          for (const [label, trace] of Object.entries(
            test.traces ?? builtInTraces,
          )) {
            phase = label;
            const samples = await page.evaluate(
              (path) => globalThis.referenceBenchmark.run(path),
              trace,
            );
            record.phases[label] = {
              ...summary(samples),
              ...(await screenshot(label)),
              samples,
            };
          }
        }
        if (values.trace && vtk) {
          record.detailTrace = await page.evaluate(() => ({
            events: performance
              .getEntriesByType("mark")
              .filter(({ name }) => name.startsWith("tile-detail:"))
              .map(({ name, startTime, detail }) => ({
                stage: name.slice("tile-detail:".length),
                at: startTime,
                ...detail,
              })),
            requests: performance
              .getEntriesByType("resource")
              .filter(({ name }) => new URL(name).pathname.startsWith("/data/"))
              .map(({ name, startTime, responseEnd, encodedBodySize }) => ({
                url: name,
                start: startTime,
                end: responseEnd,
                bytes: encodedBodySize,
              })),
          }));
        }
        if (
          test.format === "cog" &&
          !requests.some(({ status, range }) => status === 206 && range)
        )
          throw new Error("COG case made no successful HTTP Range reads");
        if (requests.some(({ status }) => status >= 400))
          throw new Error("Dataset request returned HTTP error");
        if (errors.length) throw new Error(errors[0]);
        if (
          ![
            record.initial,
            ...Object.values(record.phases),
            ...(record.checkpoints ?? []),
          ].some((capture) => capture?.foregroundFraction > 0.001)
        )
          throw new Error("No visible content in any checkpoint");
        const seenResponses = new Set();
        const repeated = new Set();
        for (const request of requests) {
          if (!request.status || request.status >= 400) continue;
          const key = request.path + "|" + (request.range ?? "");
          if (seenResponses.has(key)) repeated.add(request);
          seenResponses.add(key);
        }
        record.transport = Object.fromEntries(
          ["initial", ...Object.keys(record.phases)].map((label) => {
            const responses = requests.filter(
              (request) => request.phase === label && request.status,
            );
            const repeats = responses.filter((response) =>
              repeated.has(response),
            );
            return [
              label,
              {
                responses: responses.length,
                repeatedResponses: repeats.length,
                repeatedDeclaredBytes: repeats.reduce(
                  (total, response) => total + (response.bytes ?? 0),
                  0,
                ),
                failedRequests: requests.filter(
                  (request) => request.phase === label && request.failed,
                ).length,
                rangeResponses: responses.filter(
                  ({ status, range }) => status === 206 && range,
                ).length,
                declaredResponseBytes: responses.reduce(
                  (bytes, response) => bytes + (response.bytes ?? 0),
                  0,
                ),
              },
            ];
          }),
        );
        record.status = "ok";
      } catch (error) {
        record.status = "failed";
        record.failure = String(error.stack ?? error);
        process.exitCode = 1;
      } finally {
        await context.close().catch(() => {});
        record.graphicsWarnings = [...graphicsWarnings];
        await writeFile(
          resolve(output, name + ".json"),
          JSON.stringify(record, null, 2) + "\n",
        );
      }
      const { detailTrace: _trace, ...summaryRecord } = record;
      results.push({
        ...summaryRecord,
        phases: Object.fromEntries(
          Object.entries(record.phases).map(
            ([label, { samples: _samples, ...metrics }]) => [label, metrics],
          ),
        ),
      });
      console.log(
        name +
          " " +
          record.status +
          (record.failure ? " " + record.failure.split("\n")[0] : ""),
      );
    }
  }
} finally {
  if (gpuSession) await gpuSession.close();
  else await browser?.close();
  await server?.close();
  await fileServer?.close();
  await writeFile(
    resolve(
      output,
      settledReference ? "summary-settled-reference.json" : "summary.json",
    ),
    JSON.stringify(
      {
        network: {
          latencyMs: Number(values.latency),
          mbps: profiles ? mbps : null,
        },
        results,
      },
      null,
      2,
    ) + "\n",
  );
}
