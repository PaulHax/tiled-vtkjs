import { expect, it } from "vitest";
import { createMemoryPool } from "./memoryPool";
import { ResourcePolicyError } from "./memoryResources";
import { readMemoryResponse } from "./memoryResponse";

it("holds the work reservation until an early body refusal acknowledges cancellation", async () => {
  const pool = createMemoryPool({
    totalBytes: 100,
    workBytes: 8,
    maxConcurrentOperations: 1,
  });
  const owner = pool.createOwner();
  const work = await owner.work(undefined, 2);
  let acknowledge!: () => void;
  let cancelling!: () => void;
  const started = new Promise<void>((resolve) => {
    cancelling = resolve;
  });
  const body = new ReadableStream<Uint8Array>({
    cancel() {
      cancelling();
      return new Promise<void>((resolve) => {
        acknowledge = resolve;
      });
    },
  });
  const read = readMemoryResponse(
    {
      headers: { get: () => "16" },
      body,
      arrayBuffer: async () => {
        throw new Error("refused body must not be buffered");
      },
    },
    work,
  ).finally(() => work.release());
  const rejected = expect(read).rejects.toBeInstanceOf(ResourcePolicyError);
  await started;
  owner.close();
  expect(pool.stats()).toMatchObject({
    activeOperations: 1,
    workReservedBytes: 2,
    owners: 1,
  });
  acknowledge();
  await rejected;
  expect(pool.stats()).toMatchObject({
    activeOperations: 0,
    workReservedBytes: 0,
    owners: 0,
  });
});
