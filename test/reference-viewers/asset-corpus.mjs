import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readdir, realpath, stat } from "node:fs/promises";
import { join } from "node:path";

export async function hashAssetCorpus(root) {
  const corpus = createHash("sha256");
  const directories = new Set();

  async function visit(directory, prefix) {
    const physicalPath = await realpath(directory);
    if (directories.has(physicalPath))
      throw new Error("Asset corpus contains a directory cycle at " + prefix);
    directories.add(physicalPath);
    const names = (await readdir(directory)).sort();
    for (const name of names) {
      const path = join(directory, name);
      const relative = prefix + name;
      const info = await stat(path);
      if (info.isDirectory()) {
        await visit(path, relative + "/");
      } else if (info.isFile()) {
        const file = createHash("sha256");
        for await (const chunk of createReadStream(path)) file.update(chunk);
        corpus.update(JSON.stringify([relative, file.digest("hex")]) + "\n");
      } else {
        throw new Error("Unsupported asset corpus entry " + relative);
      }
    }
    directories.delete(physicalPath);
  }

  await visit(root, "");
  return corpus.digest("hex");
}
