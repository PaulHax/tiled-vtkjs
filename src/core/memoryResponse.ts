import type { MemoryWork } from "./memoryResources";

export type BufferedResponse = {
  readonly headers?: { get(name: string): string | null };
  readonly body?: ReadableStream<Uint8Array> | null;
  arrayBuffer(): Promise<ArrayBuffer>;
};

/** Charge overlapping streamed chunks and their contiguous destination before concatenation. */
export const readMemoryResponse = async (
  response: BufferedResponse,
  work?: MemoryWork,
  signal?: AbortSignal,
): Promise<ArrayBuffer> => {
  const stated = Number(response.headers?.get("content-length"));
  try {
    signal?.throwIfAborted();
    if (work && Number.isFinite(stated) && stated > 0)
      work.resize(Math.max(work.bytes(), stated * 2));
  } catch (error) {
    await response.body?.cancel().catch(() => {});
    throw error;
  }
  if (!work || !response.body) {
    const bytes = await response.arrayBuffer();
    work?.resize(Math.max(work.bytes(), bytes.byteLength * 2));
    return bytes;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      signal?.throwIfAborted();
      const next = await reader.read();
      if (next.done) break;
      length += next.value.byteLength;
      work.resize(Math.max(work.bytes(), length * 2));
      chunks.push(next.value);
    }
    const output = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      output.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return output.buffer;
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
};
