import { cp, mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, test } from "vitest";
import { hashAssetCorpus } from "./asset-corpus.mjs";

test("quality inputs identify nested payload bytes and paths across asset roots", async ({
  onTestFinished,
}) => {
  const work = await mkdtemp(join(tmpdir(), "reference-corpus-"));
  onTestFinished(() => rm(work, { recursive: true, force: true }));
  const original = join(work, "original");
  const copy = join(work, "copy");
  await mkdir(join(original, "tiles"), { recursive: true });
  await writeFile(
    join(original, "tileset.json"),
    '{"root":{"content":{"uri":"tiles/root.glb"}}}',
  );
  const payload = join(copy, "tiles", "root.glb");
  await writeFile(join(original, "tiles", "root.glb"), "original content");
  await cp(original, copy, { recursive: true });
  const expected = await hashAssetCorpus(original);

  expect(await hashAssetCorpus(copy)).toBe(expected);
  await writeFile(payload, "modified content");
  expect(await hashAssetCorpus(copy)).not.toBe(expected);
  await writeFile(payload, "original content");
  await rename(payload, join(copy, "tiles", "moved.glb"));
  expect(await hashAssetCorpus(copy)).not.toBe(expected);
});
