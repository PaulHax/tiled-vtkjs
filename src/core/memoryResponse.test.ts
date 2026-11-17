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
  const read = pool.admission.run({
    signal: new AbortController().signal,
    stage: {
      lane: "network",
      async start(signal) {
        const work = owner.work(signal, 2);
        try {
          const value = await readMemoryResponse(
            {
              headers: { get: () => "16" },
              body,
              arrayBuffer: async () => {
                throw new Error("refused body must not be buffered");
              },
            },
            work,
            signal,
          );
          return { value };
        } finally {
          work.release();
        }
      },
    },
  });
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
