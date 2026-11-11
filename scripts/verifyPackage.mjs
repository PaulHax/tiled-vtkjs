import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { createServer } from "node:http";
import {
  mkdtemp,
  mkdir,
  readFile,
  realpath,
  rm,
  stat,
  symlink,
  writeFile,
  copyFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { chromium } from "playwright";
import { build } from "vite";
import {
  FIXTURE_CODEC_TILES,
  generateTiles3dFixture,
} from "./generateTiles3dFixture.mjs";

const execute = promisify(execFile);
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const codecs = [
  "draco_wasm_wrapper.js",
  "draco_decoder.wasm",
  "basis_encoder.js",
  "basis_encoder.wasm",
];
const entrypoints = [
  ".",
  "./core",
  "./points",
  "./points/vtk",
  "./points/copc-worker",
  "./tiles3d",
  "./tiles3d/vtk",
  "./tiles3d/decode-worker",
];

const run = (command, args, cwd) =>
  execute(command, args, { cwd, maxBuffer: 16 * 1024 * 1024 });

const serve = async (directory) => {
  const server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(
        new URL(request.url ?? "/", "http://localhost").pathname,
      );
      const path = resolve(directory, `.${pathname}`);
      if (!path.startsWith(directory + sep)) throw new Error("invalid path");
      const content = await readFile(path);
      const type = path.endsWith(".js")
        ? "text/javascript"
        : path.endsWith(".wasm")
          ? "application/wasm"
          : path.endsWith(".json")
            ? "application/json"
            : "text/html";
      response.writeHead(200, { "Content-Type": type });
      response.end(content);
    } catch {
      response.writeHead(404);
      response.end();
    }
  });
  await new Promise((done) => server.listen(0, "127.0.0.1", done));
  const address = server.address();
  return {
    origin: `http://127.0.0.1:${address.port}`,
    close: () =>
      new Promise((done, reject) =>
        server.close((error) => (error ? reject(error) : done())),
      ),
  };
};

const checkIsolatedBundles = async (consumer, name) => {
  const results = {};
  for (const domain of ["root", "core", "points", "tiles3d"]) {
    const specifier = domain === "root" ? name : `${name}/${domain}`;
    const entry = join(consumer, `${domain}.mjs`);
    await writeFile(
      entry,
      `import * as api from ${JSON.stringify(specifier)}; globalThis.publicApi = api;\n`,
    );
    const modules = new Set();
    const result = await build({
      configFile: false,
      root: consumer,
      logLevel: "silent",
      plugins: [
        {
          name: "record-consumer-imports",
          moduleParsed(info) {
            modules.add(info.id);
          },
        },
      ],
      build: {
        write: false,
        sourcemap: true,
        minify: false,
        lib: { entry, formats: ["es"] },
      },
    });
    const outputs = (Array.isArray(result) ? result : [result]).flatMap(
      (output) => output.output,
    );
    const sources = outputs
      .filter((output) => output.type === "chunk")
      .flatMap((chunk) => chunk.map?.sources ?? []);
    const paths = [...modules, ...sources].map((path) =>
      path.replaceAll("\\", "/"),
    );
    const excluded =
      domain === "root" || domain === "core"
        ? [
            "/points/",
            "/tiles3d/",
            "@kitware/vtk.js",
            "@loaders.gl/",
            "/copc/",
            "/laz-perf/",
          ]
        : domain === "points"
          ? ["/tiles3d/", "@kitware/vtk.js", "@loaders.gl/", "/ktx-parse/"]
          : ["/points/", "@kitware/vtk.js", "/copc/", "/laz-perf/"];
    for (const forbidden of excluded) {
      assert(
        !paths.some((path) => path.includes(forbidden)),
        `${specifier} bundles forbidden domain ${forbidden}`,
      );
    }
    results[domain] = {
      bytes: outputs
        .filter((output) => output.type === "chunk")
        .reduce((sum, chunk) => sum + Buffer.byteLength(chunk.code), 0),
      modules: modules.size,
      sources,
    };
  }
  return results;
};

const checkBrowser = async (
  consumer,
  installed,
  manifest,
  evidenceDirectory,
) => {
  const output = join(consumer, "browser");
  const pageSource = (
    await readFile(join(root, "test/browser/packedScenePage.mjs"), "utf8")
  ).replaceAll("__PACKAGE__", manifest.name);
  const entry = join(consumer, "scene.mjs");
  await writeFile(entry, pageSource);
  await build({
    configFile: false,
    root: consumer,
    logLevel: "silent",
    build: {
      outDir: output,
      emptyOutDir: true,
      minify: false,
      lib: { entry, fileName: () => "scene.js", formats: ["es"] },
    },
  });
  await copyFile(
    join(installed, manifest.exports["./tiles3d/decode-worker"].default),
    join(output, "tiles3dDecodeWorker.classic.js"),
  );
  await mkdir(join(output, "tiles3d-codecs"));
  for (const codec of codecs)
    await copyFile(
      join(installed, "dist/tiles3d-codecs", codec),
      join(output, "tiles3d-codecs", codec),
    );
  await writeFile(
    join(output, "index.html"),
    '<!doctype html><style>html,body,#view{margin:0;width:128px;height:128px;overflow:hidden}</style><div id="view"></div><script type="module" src="scene.js"></script>',
  );
  await generateTiles3dFixture(join(output, "codec-fixture"));
  const server = await serve(output);
  let browser;
  try {
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({
      viewport: { width: 128, height: 128 },
    });
    const errors = [];
    const requests = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    page.on("request", (request) => requests.push(request.url()));
    await page.goto(`${server.origin}/index.html`);
    await page
      .locator("html[data-ready='true']")
      .waitFor({ timeout: 30_000 })
      .catch((error) => {
        throw new Error(
          errors.join("\n") || "packed fixture did not initialize",
          { cause: error },
        );
      });
    const call = (method) =>
      page.evaluate(async (method) => {
        let timer;
        try {
          return await Promise.race([
            window.packedScene[method](),
            new Promise((_, reject) => {
              timer = setTimeout(
                () =>
                  reject(
                    new Error(
                      `packed fixture stalled at ${method}: ${JSON.stringify(window.packedScene.snapshot())}`,
                    ),
                  ),
                30_000,
              );
            }),
          ]);
        } finally {
          clearTimeout(timer);
        }
      }, method);
    const rootState = await call("root");
    assert.equal(rootState.sharedScheduler, true);
    assert.equal(rootState.memory.memberCount, 2);
    assert(
      rootState.point.allocation.memoryBudgetBytes >=
        rootState.point.renderer.gpuResidentBytes,
    );
    assert(
      rootState.mesh.allocation.memoryBudgetBytes >=
        rootState.mesh.renderer.residentBytes,
    );
    assert(
      rootState.point.allocation.memoryBudgetBytes <
        rootState.memory.totalBytes / 2,
    );
    assert(
      rootState.point.allocation.memoryBudgetBytes +
        rootState.mesh.allocation.memoryBudgetBytes <=
        rootState.memory.totalBytes,
    );
    assert.deepEqual(rootState.pixels.point, [255, 0, 0, 255]);
    assert.deepEqual(rootState.pixels.mesh, [0, 255, 0, 255]);
    assert.equal(rootState.picks.point.status, "hit");
    assert.equal(rootState.picks.mesh.status, "hit");
    if (evidenceDirectory)
      await page
        .locator("canvas")
        .screenshot({ path: join(evidenceDirectory, "packed-parent.png") });

    const waiting = await call("requestRefinement");
    assert.deepEqual(waiting.mesh.renderer.drawnTileIds, ["root"]);
    assert.deepEqual(waiting.pixels.mesh, [0, 255, 0, 255]);
    const queued = await call("releaseChildren");
    assert(queued.coordinator.submissions.queuedJobs >= 3);
    const blocked = await call("paint");
    assert.equal(blocked.blockerRuns, 1);
    assert.equal(blocked.coordinator.submissions.lastFrameAdmittedJobs, 1);
    assert.equal(
      blocked.coordinator.submissions.lastFrameAdmittedBytes,
      blocked.coordinator.submissions.maxBytesPerFrame,
    );
    assert.deepEqual(blocked.mesh.renderer.drawnTileIds, ["root"]);
    const refined = await call("refined");
    assert.deepEqual(refined.mesh.renderer.drawnTileIds, ["root/0", "root/1"]);
    assert.deepEqual(refined.pixels.mesh, [0, 0, 255, 255]);
    assert.deepEqual(refined.pixels.point, [255, 0, 0, 255]);
    assert.equal(refined.picks.point.status, "hit");
    assert.equal(refined.picks.mesh.status, "hit");
    assert(
      refined.point.renderer.gpuResidentBytes +
        refined.mesh.renderer.residentBytes <=
        refined.memory.totalBytes,
    );
    assert(
      refined.coordinator.submissions.peakFrameAdmittedBytes <=
        refined.coordinator.submissions.maxBytesPerFrame,
    );
    if (evidenceDirectory)
      await page
        .locator("canvas")
        .screenshot({ path: join(evidenceDirectory, "packed-refined.png") });
    const codecResults = [];
    for (const [codec, file] of Object.entries(FIXTURE_CODEC_TILES)) {
      const decoded = await page.evaluate(
        async ({ origin, file }) => {
          const contentUrl = `${origin}/codec-fixture/content/${file}.glb`;
          const content = await (await fetch(contentUrl)).arrayBuffer();
          const worker = new Worker(`${origin}/tiles3dDecodeWorker.classic.js`);
          let timer;
          try {
            return await new Promise((done, reject) => {
              timer = setTimeout(
                () => reject(new Error("packed codec worker stalled")),
                30_000,
              );
              worker.onerror = (event) => reject(new Error(event.message));
              worker.onmessage = (event) => {
                if (event.data.kind !== "complete") {
                  reject(new Error(JSON.stringify(event.data)));
                  return;
                }
                const result = event.data.result;
                const texture = result.primitives[0]?.material.baseColorTexture;
                done({
                  primitiveCount: result.primitives.length,
                  geometryBytes: result.byteEstimate.geometry,
                  textureBytes: result.byteEstimate.textures,
                  textureKind: texture?.kind,
                  rgbaBytes: texture?.rgba?.byteLength,
                });
              };
              worker.postMessage(
                {
                  kind: "decode",
                  jobId: 1,
                  request: {
                    content,
                    contentUrl,
                    dependencyRootUrl: `${origin}/codec-fixture/`,
                    revision: "packed-codecs",
                    accumulatedTransform: [
                      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                    ],
                    tilesetToScene: [
                      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                    ],
                    textureCapabilities: {
                      capabilityKey: "packed-rgba",
                      compressedFormats: [],
                    },
                    wasm: {
                      draco: {
                        wrapperUrl: `${origin}/tiles3d-codecs/draco_wasm_wrapper.js`,
                        wasmUrl: `${origin}/tiles3d-codecs/draco_decoder.wasm`,
                      },
                      basis: {
                        encoderUrl: `${origin}/tiles3d-codecs/basis_encoder.js`,
                        wasmUrl: `${origin}/tiles3d-codecs/basis_encoder.wasm`,
                      },
                    },
                  },
                },
                [content],
              );
            });
          } finally {
            clearTimeout(timer);
            worker.terminate();
          }
        },
        { origin: server.origin, file },
      );
      assert.equal(decoded.primitiveCount, 2);
      assert(decoded.geometryBytes > 0);
      assert.equal(decoded.textureKind, "rgba");
      assert.equal(decoded.rgbaBytes, 16 * 16 * 4);
      codecResults.push({ codec, ...decoded });
    }
    for (const codec of codecs)
      assert(
        requests.some((url) => url.endsWith(`/tiles3d-codecs/${codec}`)),
        `packed codec asset ${codec} must be fetched`,
      );
    assert(
      requests.every((url) => url.startsWith(server.origin + "/")),
      "codec decoding must stay on the staged consumer origin",
    );
    const released = await call("releasePoint");
    assert.equal(released.memory.memberCount, 1);
    assert(
      released.mesh.allocation.memoryBudgetBytes >=
        released.mesh.renderer.residentBytes,
    );
    assert(
      released.mesh.allocation.memoryBudgetBytes <= released.memory.totalBytes,
    );
    assert.equal(released.memory.owners, 1);
    assert.deepEqual(released.pixels.point, [0, 0, 0, 255]);
    const disposed = await call("dispose");
    assert.equal(disposed.actors, 0);
    assert.equal(disposed.memory.memberCount, 0);
    assert.equal(disposed.memory.owners, 0);
    assert.equal(disposed.memory.rendererBytes, 0);
    assert.equal(disposed.memory.cpuRetainedBytes, 0);
    assert.equal(disposed.memory.cpuReservedBytes, 0);
    assert.equal(disposed.memory.workReservedBytes, 0);
    assert.equal(disposed.mesh.disposed, true);
    assert.equal(disposed.coordinator.submissions.queuedJobs, 0);
    assert.deepEqual(disposed.pixels.mesh, [0, 0, 0, 255]);
    assert(
      requests.some((url) => url.endsWith("/tiles3dDecodeWorker.classic.js")),
      "default worker URL must reach the staged packed worker",
    );
    assert.deepEqual(errors, []);
    return {
      root: rootState,
      waiting,
      blocked,
      refined,
      released,
      disposed,
      codecResults,
      requests,
    };
  } finally {
    await browser?.close();
    await server.close();
  }
};

export const verifyPackedPackage = async ({
  evidenceDirectory = process.env.STREAMING_PACKAGE_EVIDENCE_DIR,
  browserValidation = true,
} = {}) => {
  const directory = await mkdtemp(join(tmpdir(), "streaming-package-"));
  try {
    if (evidenceDirectory) await mkdir(evidenceDirectory, { recursive: true });
    const packed = JSON.parse(
      (
        await run(
          "npm",
          [
            "pack",
            "--ignore-scripts",
            "--json",
            "--pack-destination",
            directory,
          ],
          root,
        )
      ).stdout,
    )[0];
    assert(
      !packed.files.some((file) => /^(examples|scripts)\//.test(file.path)),
      "examples and tooling are unpublished",
    );
    const tarball = join(directory, packed.filename);
    const consumer = join(directory, "consumer");
    await mkdir(consumer);
    await writeFile(
      join(consumer, "package.json"),
      JSON.stringify({
        name: "streaming-consumer-check",
        private: true,
        type: "module",
      }),
    );
    await run(
      "npm",
      [
        "install",
        "--prefer-offline",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund",
        tarball,
      ],
      consumer,
    );
    const installed = join(consumer, "node_modules", packed.name);
    const manifest = JSON.parse(
      await readFile(join(installed, "package.json"), "utf8"),
    );
    for (const entrypoint of entrypoints) {
      const entry = manifest.exports[entrypoint];
      assert(entry, `missing package export ${entrypoint}`);
      for (const target of Object.values(entry))
        await stat(join(installed, target));
    }
    for (const codec of codecs) {
      const target =
        manifest.exports[`./tiles3d/codecs/${codec}`] ??
        manifest.exports["./tiles3d/codecs/*"];
      assert(target, `codec ${codec} must be resolvable through exports`);
      const destination = (
        typeof target === "string" ? target : target.default
      ).replace("*", codec);
      await stat(join(installed, destination));
    }
    const name = manifest.name;
    await writeFile(
      join(consumer, "consumer.ts"),
      [
        `import { createMemoryPool, createStreamedSceneCoordinator, type Bounds, type Vec3 } from ${JSON.stringify(name)};`,
        `import { createHttpTileSource, type TileSource } from ${JSON.stringify(name + "/points")};`,
        `import { createPointCloudMember } from ${JSON.stringify(name + "/points/vtk")};`,
        `import { serveCopcTileSourceWorker } from ${JSON.stringify(name + "/points/copc-worker")};`,
        `import { DecodeWorkerPool, type Tiles3dMemberServices } from ${JSON.stringify(name + "/tiles3d")};`,
        `import { createTiles3dMember } from ${JSON.stringify(name + "/tiles3d/vtk")};`,
        `const vector: Vec3 = [0, 0, 0]; const bounds: Bounds = { min: vector, max: vector };`,
        `const source: TileSource | null = null; const services: Tiles3dMemberServices | null = null;`,
        `void [createMemoryPool, createStreamedSceneCoordinator, bounds, source, services, createHttpTileSource, createPointCloudMember, serveCopcTileSourceWorker, DecodeWorkerPool, createTiles3dMember];`,
      ].join("\n"),
    );
    await writeFile(
      join(consumer, "tsconfig.json"),
      JSON.stringify({
        compilerOptions: {
          target: "ES2022",
          module: "ESNext",
          moduleResolution: "bundler",
          strict: true,
          skipLibCheck: false,
          noEmit: true,
        },
        include: ["consumer.ts"],
      }),
    );
    await run(
      join(root, "node_modules/.bin/tsc"),
      ["-p", join(consumer, "tsconfig.json")],
      consumer,
    );
    const resolver = join(consumer, "resolve.mjs");
    await writeFile(
      resolver,
      [
        'import assert from "node:assert/strict";',
        'import { stat } from "node:fs/promises";',
        'import { fileURLToPath } from "node:url";',
        `const name = ${JSON.stringify(name)};`,
        'assert.throws(() => import.meta.resolve("@kitware/vtk.js"), { code: "ERR_MODULE_NOT_FOUND" });',
        'for (const old of ["vtk", "copc-worker", "tiles3d-decode-worker"]) assert.throws(() => import.meta.resolve(`${name}/${old}`), { code: "ERR_PACKAGE_PATH_NOT_EXPORTED" });',
        "for (const entry of [name, `${name}/core`, `${name}/points`, `${name}/tiles3d`]) await import(entry);",
        `const assets = ["tiles3d/decode-worker", ...${JSON.stringify(codecs)}.map((file) => "tiles3d/codecs/" + file)];`,
        "const resolved = [];",
        'for (const asset of assets) { const path = fileURLToPath(import.meta.resolve(`${name}/${asset}`)); await stat(path); resolved.push({ asset, filename: path.split("/").at(-1) }); }',
        "console.log(JSON.stringify(resolved));",
      ].join("\n"),
    );
    const resolvedAssets = JSON.parse(
      (await run(process.execPath, [resolver], consumer)).stdout,
    );
    await mkdir(join(consumer, "node_modules/@kitware"), { recursive: true });
    await symlink(
      await realpath(join(root, "node_modules/@kitware/vtk.js")),
      join(consumer, "node_modules/@kitware/vtk.js"),
    );
    const bundles = await checkIsolatedBundles(consumer, name);
    const browser = browserValidation
      ? await checkBrowser(consumer, installed, manifest, evidenceDirectory)
      : null;
    const result = {
      package: name,
      version: manifest.version,
      tarball: packed.filename,
      tarballSha256: createHash("sha256")
        .update(await readFile(tarball))
        .digest("hex"),
      exports: Object.keys(manifest.exports),
      fileCount: packed.files.length,
      declarations: {
        skipLibCheck: false,
        vtkInstalled: false,
        imports: "all public declarations",
      },
      resolvedAssets,
      retiredExports: ["/vtk", "/copc-worker", "/tiles3d-decode-worker"],
      bundles,
      browser,
    };
    if (evidenceDirectory) {
      await copyFile(tarball, join(evidenceDirectory, packed.filename));
      await writeFile(
        join(evidenceDirectory, "packed-package.json"),
        JSON.stringify(result, null, 2) + "\n",
      );
    }
    return result;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
};

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const result = await verifyPackedPackage({
    browserValidation: !process.argv.includes("--contracts-only"),
  });
  console.log(
    JSON.stringify(
      {
        package: result.package,
        version: result.version,
        tarballSha256: result.tarballSha256,
        files: result.fileCount,
        exports: result.exports,
        bundleBytes: Object.fromEntries(
          Object.entries(result.bundles).map(([name, bundle]) => [
            name,
            bundle.bytes,
          ]),
        ),
        declarations: result.declarations,
        resolvedAssets: result.resolvedAssets,
        renderer: result.browser?.root.rendererName,
        browser: result.browser ? "passed" : "skipped",
      },
      null,
      2,
    ),
  );
}
