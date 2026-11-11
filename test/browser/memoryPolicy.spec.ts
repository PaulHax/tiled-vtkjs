import { afterAll, beforeAll, expect, it } from "vitest";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import sharp from "sharp";
import type { Page } from "playwright";
import {
  browserFor,
  closeBrowsers,
  collectPageFailures,
  EXAMPLE_DIST,
} from "./browserSession";
import { startStaticServer, type StaticServer } from "./server";

let directory: string;
let server: StaticServer;
const box = (x = 0) => ({ box: [x, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1] });
beforeAll(async () => {
  directory = await mkdtemp(join(tmpdir(), "tile-memory-"));
  for (const [name, size, color, x] of [
    ["large", 32, "red", -2.5],
    ["tiny", 4, "lime", 2.5],
    ["replacement", 8, "red", 0],
  ] as const) {
    const root = join(directory, name);
    await mkdir(root);
    const vertices = new Float32Array([
      -1.5, -1.5, 0, 1.5, -1.5, 0, 1.5, 1.5, 0, -1.5, -1.5, 0, 1.5, 1.5, 0,
      -1.5, 1.5, 0,
    ]);
    const normals = new Float32Array(
      Array.from({ length: 18 }, (_, i) => (i % 3 === 2 ? 1 : 0)),
    );
    const uv = new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]);
    await writeFile(
      join(root, "geometry.bin"),
      Buffer.concat([
        Buffer.from(vertices.buffer),
        Buffer.from(normals.buffer),
        Buffer.from(uv.buffer),
      ]),
    );
    const gltf = {
      asset: { version: "2.0" },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0, translation: [x, 0, 0] }],
      meshes: [
        {
          primitives: [
            {
              attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 },
              material: 0,
            },
          ],
        },
      ],
      buffers: [{ uri: "geometry.bin", byteLength: 192 }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: 72 },
        { buffer: 0, byteOffset: 72, byteLength: 72 },
        { buffer: 0, byteOffset: 144, byteLength: 48 },
      ],
      accessors: [
        {
          bufferView: 0,
          componentType: 5126,
          count: 6,
          type: "VEC3",
          min: [-1.5, -1.5, 0],
          max: [1.5, 1.5, 0],
        },
        { bufferView: 1, componentType: 5126, count: 6, type: "VEC3" },
        { bufferView: 2, componentType: 5126, count: 6, type: "VEC2" },
      ],
      materials: [
        {
          doubleSided: true,
          extensions: { KHR_materials_unlit: {} },
          pbrMetallicRoughness: {
            baseColorTexture: { index: 0 },
            metallicFactor: 0,
          },
        },
      ],
      extensionsUsed: ["KHR_materials_unlit"],
      textures: [{ source: 0 }],
      images: [{ uri: "image.png" }],
    };
    await writeFile(join(root, "model.gltf"), JSON.stringify(gltf));
    await writeFile(
      join(root, "image.png"),
      await sharp({
        create: { width: size, height: size, channels: 4, background: color },
      })
        .png()
        .toBuffer(),
    );
    const rootTile = {
      boundingVolume: box(x),
      geometricError: 0,
      content: { uri: "model.gltf" },
    };
    if (name === "replacement") {
      const fine = { ...gltf, images: [{ uri: "fine.png" }] };
      await writeFile(join(root, "fine.gltf"), JSON.stringify(fine));
      await writeFile(
        join(root, "fine.png"),
        await sharp({
          create: { width: 16, height: 16, channels: 4, background: "lime" },
        })
          .png()
          .toBuffer(),
      );
      Object.assign(rootTile, {
        geometricError: 100,
        refine: "REPLACE",
        children: [0, 1].map(() => ({
          boundingVolume: box(),
          geometricError: 0,
          content: { uri: "fine.gltf" },
        })),
      });
    }
    await writeFile(
      join(root, "tileset.json"),
      JSON.stringify({
        asset: { version: "1.1" },
        geometricError: rootTile.geometricError,
        root: { refine: "REPLACE", ...rootTile },
      }),
    );
  }
  server = await startStaticServer({
    "/memory-fixtures": directory,
    "/": EXAMPLE_DIST,
  });
});
afterAll(async () => {
  await closeBrowsers();
  await server?.close();
  if (directory) await rm(directory, { recursive: true, force: true });
});

// The page exposes diagnostics and controls; assertions also verify pixels and picking.
const eventually = (read: () => unknown | Promise<unknown>) =>
  expect.poll(read, { timeout: 30_000 });
const state = (page: Page): Promise<any> =>
  page.evaluate(() => (window as any).memoryMeasurement.snapshot());
const call = (page: Page, method: string, ...args: any[]) =>
  page.evaluate(
    ({ method, args }) => (window as any).memoryMeasurement[method](...args),
    { method, args },
  );
const open = async (query = "") => {
  const page = await (
    await browserFor("software")
  ).newPage({ viewport: { width: 800, height: 700 } });
  const failures = collectPageFailures(page);
  await page.goto(
    `${server.origin}/memory/index.html?budget=${6144 / 1048576}&cpu=65536&work=65536&${query}`,
  );
  await page.waitForFunction(() => !!(window as any).memoryMeasurement);
  return { page, failures };
};
const colors = async (page: Page, viewer = "#viewer") => {
  const { data, info } = await sharp(await page.locator(viewer).screenshot())
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const counts = { red: 0, green: 0, blue: 0, redX: 0, redY: 0 };
  for (let i = 0; i < data.length; i += 4) {
    const [r, g, b] = [data[i]!, data[i + 1]!, data[i + 2]!];
    if (r > 80 && r > 2 * g && r > 2 * b) {
      counts.red++;
      counts.redX += (i / 4) % info.width;
      counts.redY += Math.floor(i / 4 / info.width);
    }
    if (g > 80 && g > 2 * r && g > 2 * b) counts.green++;
    if (b > 80 && b > 2 * r && b > 2 * g) counts.blue++;
  }
  return counts;
};
const waitDrawn = async (page: Page, count: number) => {
  await eventually(async () => {
    const s = await state(page);
    return s.settled && s.drawn;
  }).toBe(count);
};

it("borrows spare capacity across live assets and preserves coverage through budget changes", async () => {
  const { page, failures } = await open("asset=large&views=2");
  try {
    await waitDrawn(page, 1);
    await page.locator("#asset").fill("tiny");
    await page.locator("#add").click();
    await waitDrawn(page, 2);
    let s = await state(page);
    expect(s.stats[0].allocation.memoryBudgetBytes).toBeGreaterThan(3072);
    expect(s.memory.rendererBytes).toBeLessThanOrEqual(6144);
    const painted = await colors(page);
    expect(painted.red).toBeGreaterThan(50);
    expect(painted.green).toBeGreaterThan(50);
    const picks = await call(
      page,
      "pick",
      painted.redX / painted.red,
      painted.redY / painted.red,
    );
    expect(picks[0].result.status).toBe("hit");

    await call(page, "setActive", "asset-0", false);
    await waitDrawn(page, 1);
    s = await state(page);
    expect(s.memory.cpuRetainedBytes).toBeGreaterThanOrEqual(4096);
    expect((await colors(page)).red).toBe(0);
    await call(page, "setActive", "asset-0", true);
    await waitDrawn(page, 2);
    await page.locator("#remove").click();
    await waitDrawn(page, 1);
    await page.locator("#budget").fill("1024");
    await page.locator("#apply-budget").click();
    await eventually(
      async () => (await state(page)).stats[0].resourceState.status,
    ).toBe("blocked");
    expect((await colors(page)).red).toBeGreaterThan(50);
    await page.locator("#budget").fill("6144");
    await page.locator("#apply-budget").click();
    await eventually(
      async () => (await state(page)).stats[0].resourceState.status,
    ).toBe("ready");
    const points = await call(page, "add", "points", 1);
    await waitDrawn(page, 2);
    expect((await colors(page, "#viewer-2")).blue).toBeGreaterThan(10);
    await call(page, "remove", points);
    await call(page, "remove", "asset-0");
    await eventually(async () => (await state(page)).memory.owners).toBe(0);
    expect((await state(page)).memory).toMatchObject({
      rendererBytes: 0,
      cpuRetainedBytes: 0,
      cpuReservedBytes: 0,
      workReservedBytes: 0,
    });
    expect(failures).toEqual([]);
    expect((await state(page)).errors).toEqual([]);
  } finally {
    await page.close();
  }
});

it("keeps canceled physical work charged after removal and admits queued work when it settles", async () => {
  const { page, failures } = await open();
  try {
    await call(page, "configure", { maxConcurrentOperations: 1 });
    const held = await call(page, "add", "delayed-points");
    await eventually(
      async () => (await state(page)).memory.workReservedBytes,
    ).toBeGreaterThan(0);
    await call(page, "remove", held);
    expect((await state(page)).memory).toMatchObject({
      memberCount: 0,
      owners: 1,
      activeOperations: 1,
    });
    const next = await call(page, "add", "points");
    await eventually(
      async () => (await state(page)).memory.queuedOperations,
    ).toBe(1);
    await call(page, "releaseDelayed");
    await waitDrawn(page, 1);
    expect((await colors(page)).blue).toBeGreaterThan(10);
    await call(page, "remove", next);
    await eventually(async () => (await state(page)).memory.owners).toBe(0);
    expect((await state(page)).memory).toMatchObject({
      activeOperations: 0,
      queuedOperations: 0,
      workReservedBytes: 0,
    });
    expect(failures).toEqual([]);
  } finally {
    await page.close();
  }
});

it("finishes competing replacement groups within shared transition headroom", async () => {
  const { page, failures } = await open("transition=1024");
  try {
    await call(page, "add", "replacement");
    await call(page, "add", "replacement");
    await eventually(async () =>
      (await state(page)).stats.map((s: any) => s.renderer.drawnTileIds),
    ).toEqual([
      ["root/0", "root/1"],
      ["root/0", "root/1"],
    ]);
    expect((await state(page)).memory.rendererBytes).toBeLessThanOrEqual(
      6144 + 1024,
    );
    expect((await colors(page)).green).toBeGreaterThan(50);
    await call(page, "remove", "asset-0");
    await call(page, "remove", "asset-1");
    await eventually(async () => (await state(page)).memory.owners).toBe(0);
    expect(failures).toEqual([]);
  } finally {
    await page.close();
  }
});

it("reports an intrinsic work refusal once and retries after its allowance increases", async () => {
  const { page, failures } = await open();
  try {
    await call(page, "configure", { workBytes: 4096 });
    await call(page, "add", "large");
    await eventually(
      async () => (await state(page)).stats[0].resourceState,
    ).toMatchObject({ status: "blocked", domain: "work", retryable: false });
    const attempts = (await state(page)).stats[0].queue.entries[0].attempt;
    for (let frame = 0; frame < 3; frame++) {
      await call(page, "render");
      await page.evaluate(
        () => new Promise<void>((done) => requestAnimationFrame(() => done())),
      );
    }
    expect((await state(page)).stats[0].queue.entries[0].attempt).toBe(
      attempts,
    );
    expect((await state(page)).errors).toEqual([]);
    await call(page, "configure", { workBytes: 65536 });
    await waitDrawn(page, 1);
    expect((await colors(page)).red).toBeGreaterThan(50);
    expect(failures).toEqual([]);
  } finally {
    await page.close();
  }
});

it("reports an unavailable coarse transition after a limit reduction without losing existing coverage", async () => {
  const { page, failures } = await open("cache=0");
  try {
    await call(page, "add", "replacement");
    await eventually(
      async () => (await state(page)).stats[0].renderer.drawnTileIds,
    ).toEqual(["root/0", "root/1"]);
    await call(page, "configure", { totalBytes: 512, transitionBytes: 128 });
    await eventually(
      async () => (await state(page)).stats[0].resourceState,
    ).toMatchObject({
      status: "blocked",
      domain: "renderer",
      reason: "transition-unavailable",
    });
    await eventually(
      async () =>
        (await state(page)).coordinator.members[0].governorInputs.work
          .operations,
    ).toBe(0);
    expect((await colors(page)).green).toBeGreaterThan(50);
    expect((await state(page)).memory.rendererBytes).toBeGreaterThan(512);
    await call(page, "configure", { totalBytes: 6144, transitionBytes: 1536 });
    await eventually(
      async () => (await state(page)).stats[0].resourceState.status,
    ).toBe("ready");
    expect((await colors(page)).green).toBeGreaterThan(50);
    expect(failures).toEqual([]);
    expect((await state(page)).errors).toEqual([]);
  } finally {
    await page.close();
  }
});

it("distinguishes allocation failure from policy refusal and releases failed renderer reservations", async () => {
  const { page, failures } = await open("asset=allocation-failure");
  try {
    await eventually(
      async () => (await state(page)).stats[0].resourceState,
    ).toMatchObject({ status: "error", errorName: "RangeError" });
    expect((await state(page)).memory).toMatchObject({
      rendererBytes: 0,
      cpuReservedBytes: 1024,
      workReservedBytes: 0,
    });
    expect((await colors(page)).red).toBe(0);
    expect((await state(page)).errors[0]).toContain("allocation failed");
    await call(page, "remove", "asset-0");
    await eventually(async () => (await state(page)).memory.owners).toBe(0);
    expect(failures).toEqual([]);
  } finally {
    await page.close();
  }
});

it("retains CPU ownership on context loss and releases it before host reopening", async () => {
  const { page, failures } = await open("asset=large");
  try {
    await waitDrawn(page, 1);
    const cpu = (await state(page)).memory.cpuRetainedBytes;
    await call(page, "loseContext");
    await eventually(async () => (await state(page)).contextLost).toBe(true);
    expect((await state(page)).memory.cpuRetainedBytes).toBe(cpu);
    expect((await state(page)).stats[0].resourceState.status).toBe("ready");
    await call(page, "remove", "asset-0");
    await eventually(async () => (await state(page)).memory.owners).toBe(0);
    expect((await state(page)).memory.cpuRetainedBytes).toBe(0);
    expect((await state(page)).errors).toEqual(["webglcontextlost"]);
    await page.reload();
    await page.waitForFunction(() => !!(window as any).memoryMeasurement);
    await waitDrawn(page, 1);
    expect((await colors(page)).red).toBeGreaterThan(50);
    expect((await state(page)).errors).toEqual([]);
    expect(failures).toEqual([]);
  } finally {
    await page.close();
  }
});
