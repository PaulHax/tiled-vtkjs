import { describe, expect, it, vi } from "vitest";

import {
  createGaussianDistanceSorter,
  createGaussianSortController,
  createGaussianSortWorkerBackend,
  type GaussianSortJob,
  type GaussianSortPresentation,
  type GaussianSortRange,
  type GaussianSortWorkerLike,
  type GaussianSortWorkerRequest,
} from "./gaussianSort";

const placement = (x = 0, y = 0, z = 0): Float64Array =>
  new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);

const range = (
  id: string,
  positions: number[],
  modelToScene = placement(),
  generation = 0,
): GaussianSortRange => ({
  id,
  generation,
  positions: new Float32Array(positions),
  modelToScene,
});

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
};

const controlledBackend = () => {
  const started: ReturnType<typeof deferred<GaussianSortJob>>[] = [];
  const results: ReturnType<typeof deferred<Uint32Array>>[] = [];
  const call = (index: number) => {
    started[index] ??= deferred<GaussianSortJob>();
    results[index] ??= deferred<Uint32Array>();
    return { started: started[index]!, result: results[index]! };
  };
  const sort = vi.fn((job: GaussianSortJob) => {
    const next = call(sort.mock.calls.length - 1);
    next.started.resolve(job);
    return next.result.promise;
  });
  return { backend: { sort, dispose: vi.fn() }, call };
};

describe("Gaussian camera-distance order", () => {
  it("globally interleaves overlapping ranges by distance rather than view depth", () => {
    const sorter = createGaussianDistanceSorter();
    // The off-axis point has smaller view depth (z), but greater distance.
    const order = sorter.sort(
      [
        range("tile-a", [0, 0, 6, 0, 0, 2]),
        range("tile-b", [8, 0, 4, 0, 0, 5]),
      ],
      [0, 0, 0],
    );
    expect([...order]).toEqual([2, 0, 3, 1]);
  });

  it("retains stable input order for equal distances and repeated sorts", () => {
    const sorter = createGaussianDistanceSorter();
    const ranges = [range("a", [1, 0, 0, -1, 0, 0]), range("b", [0, 1, 0])];
    const first = sorter.sort(ranges, [0, 0, 0]);
    expect([...first]).toEqual([0, 1, 2]);
    expect([...sorter.sort(ranges, [2, 0, 0])]).toEqual([1, 2, 0]);
    expect([...first]).toEqual([0, 1, 2]);
    expect([...sorter.sort([], [0, 0, 0])]).toEqual([]);
  });

  it("keeps submeter local offsets and Float64 distance distinctions at ECEF scale", () => {
    const sorter = createGaussianDistanceSorter();
    const matrix = placement(6378137, 20, -30);
    // Both squared distances round to the same Float32 key.
    const ranges = [range("ecef", [1, 0, 0, 1, 0.0001, 0], matrix)];
    expect([...sorter.sort(ranges, [6378137, 20, -30])]).toEqual([1, 0]);
    expect([...ranges[0]!.positions]).toEqual([
      1,
      0,
      0,
      1,
      Math.fround(0.0001),
      0,
    ]);
  });

  it("applies rotation and nonuniform placement scale to centers", () => {
    const sorter = createGaussianDistanceSorter();
    const matrix = new Float64Array([
      0, 2, 0, 0, -3, 0, 0, 0, 0, 0, 4, 0, 10, 0, 0, 1,
    ]);
    const ranges = [range("placed", [1, 0, 0, 0, 1, 0], matrix)];
    expect([...sorter.sort(ranges, [10, 0, 0])]).toEqual([1, 0]);
  });

  it("matches an independent numeric reference for all distance exponents", () => {
    const sorter = createGaussianDistanceSorter();
    const values = [0, 0.001, 7, 0.01, 1e10, 1e-10, 1e20, 7, -7, 1e-20];
    const positions = values.flatMap((value) => [value, 0, 0]);
    const expected = values
      .map((value, id) => ({ id, d: Math.fround(value) ** 2 }))
      .sort((a, b) => b.d - a.d || a.id - b.id)
      .map(({ id }) => id);
    expect([...sorter.sort([range("powers", positions)], [0, 0, 0])]).toEqual(
      expected,
    );
  });

  it("rejects invalid identities, placement and distances with actionable errors", () => {
    const sorter = createGaussianDistanceSorter();
    expect(() => sorter.sort([range("bad", [NaN, 0, 0])], [0, 0, 0])).toThrow(
      "bad has a nonfinite camera distance",
    );
    expect(() =>
      sorter.sort([range("a", [0, 0, 0]), range("a", [])], [0, 0, 0]),
    ).toThrow("identities must be unique");
    expect(() =>
      sorter.sort([range("a", [0, 0, 0], new Float64Array(16))], [0, 0, 0]),
    ).toThrow("finite affine placement");
    expect(() => sorter.sort([], [Infinity, 0, 0])).toThrow("camera position");
  });
});

describe("Gaussian sort presentation controller", () => {
  it("coalesces camera updates and publishes only the latest requested order", async () => {
    const controlled = controlledBackend();
    const presentations: GaussianSortPresentation[] = [];
    const controller = createGaussianSortController({
      backend: controlled.backend,
      onOrder: (value) => presentations.push(value),
      onError: vi.fn(),
    });
    controller.setScene([range("tile", [0, 0, 1, 0, 0, 2])]);
    controller.request([0, 0, 0]);
    expect((await controlled.call(0).started.promise).cameraPosition).toEqual([
      0, 0, 0,
    ]);
    controller.request([1, 0, 0]);
    controller.request([2, 0, 0]);
    controller.request([3, 0, 0]);
    expect(controlled.backend.sort).toHaveBeenCalledTimes(1);
    controlled.call(0).result.resolve(new Uint32Array([1, 0]));
    expect((await controlled.call(1).started.promise).cameraPosition).toEqual([
      3, 0, 0,
    ]);
    expect(presentations).toEqual([]);
    controlled.call(1).result.resolve(new Uint32Array([0, 1]));
    await controller.whenIdle();
    expect(presentations).toHaveLength(1);
    expect([...presentations[0]!.order]).toEqual([0, 1]);
    // Camera orientation is deliberately absent from the request contract.
    controller.request([3, 0, 0]);
    await controller.whenIdle();
    expect(controlled.backend.sort).toHaveBeenCalledTimes(2);
    controller.dispose();
  });

  it("rejects stale scene generations and copies placement before dispatch", async () => {
    const controlled = controlledBackend();
    const presentations: GaussianSortPresentation[] = [];
    const controller = createGaussianSortController({
      backend: controlled.backend,
      onOrder: (value) => presentations.push(value),
      onError: vi.fn(),
    });
    const matrix = placement();
    controller.setScene([range("tile", [0, 0, 1], matrix, 1)]);
    controller.request([0, 0, 0]);
    matrix[12] = 100;
    expect(
      (await controlled.call(0).started.promise).ranges[0]!.modelToScene[12],
    ).toBe(0);
    controller.setScene([range("tile", [0, 0, 2], placement(10), 2)]);
    controlled.call(0).result.resolve(new Uint32Array([0]));
    const replacement = await controlled.call(1).started.promise;
    expect(replacement.sceneRevision).toBe(2);
    expect(presentations).toHaveLength(0);
    controlled.call(1).result.resolve(new Uint32Array([0]));
    await controller.whenIdle();
    expect(presentations[0]!.ranges).toEqual([
      { id: "tile", generation: 2, offset: 0, count: 1 },
    ]);
    expect(presentations[0]!.sceneRevision).toBe(2);
    controller.dispose();
  });

  it("retains the last coherent presentation during a replacement", async () => {
    const controlled = controlledBackend();
    const presentations: GaussianSortPresentation[] = [];
    const controller = createGaussianSortController({
      backend: controlled.backend,
      onOrder: (value) => presentations.push(value),
      onError: vi.fn(),
    });
    controller.setScene([range("parent", [0, 0, 1])]);
    controller.request([0, 0, 0]);
    await controlled.call(0).started.promise;
    controlled.call(0).result.resolve(new Uint32Array([0]));
    await controller.whenIdle();
    controller.setScene([
      range("child-a", [0, 0, 1]),
      range("child-b", [0, 0, 2]),
    ]);
    await controlled.call(1).started.promise;
    expect(presentations).toHaveLength(1);
    expect(presentations[0]!.ranges[0]!.id).toBe("parent");
    controlled.call(1).result.resolve(new Uint32Array([1, 0]));
    await controller.whenIdle();
    expect(presentations[1]!.ranges).toEqual([
      { id: "child-a", generation: 0, offset: 0, count: 1 },
      { id: "child-b", generation: 0, offset: 1, count: 1 },
    ]);
    controller.dispose();
  });

  it("reports current errors, skips obsolete errors and allows subsequent work", async () => {
    const controlled = controlledBackend();
    const onError = vi.fn();
    const controller = createGaussianSortController({
      backend: controlled.backend,
      onOrder: vi.fn(),
      onError,
    });
    controller.request([0, 0, 0]);
    await controlled.call(0).started.promise;
    controller.request([1, 0, 0]);
    controlled.call(0).result.reject(new Error("obsolete"));
    await controlled.call(1).started.promise;
    expect(onError).not.toHaveBeenCalled();
    controlled.call(1).result.reject(new Error("current"));
    await controller.whenIdle();
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ message: "current" }),
    );
    controller.request([2, 0, 0]);
    await controlled.call(2).started.promise;
    controlled.call(2).result.resolve(new Uint32Array());
    await controller.whenIdle();
    controller.dispose();
  });

  it("disposes its backend and prevents late publication and dispatch", async () => {
    const controlled = controlledBackend();
    const onOrder = vi.fn();
    const onError = vi.fn();
    const controller = createGaussianSortController({
      backend: controlled.backend,
      onOrder,
      onError,
    });
    controller.request([0, 0, 0]);
    await controlled.call(0).started.promise;
    controller.request([1, 0, 0]);
    const idle = controller.whenIdle();
    controller.dispose();
    await idle;
    controlled.call(0).result.resolve(new Uint32Array());
    await controlled.call(0).result.promise;
    controller.request([2, 0, 0]);
    controller.dispose();
    expect(controlled.backend.dispose).toHaveBeenCalledTimes(1);
    expect(controlled.backend.sort).toHaveBeenCalledTimes(1);
    expect(onOrder).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });

  it("does not dispatch a queued microtask after immediate disposal", async () => {
    const controlled = controlledBackend();
    const controller = createGaussianSortController({
      backend: controlled.backend,
      onOrder: vi.fn(),
      onError: vi.fn(),
    });
    controller.request([0, 0, 0]);
    controller.dispose();
    await controller.whenIdle();
    expect(controlled.backend.sort).not.toHaveBeenCalled();
  });

  it("contains throwing observers while continuing queued work and disposal", async () => {
    const controlled = controlledBackend();
    let controller!: ReturnType<typeof createGaussianSortController>;
    const onOrder = vi.fn(() => {
      controller.request([1, 0, 0]);
      throw new Error("order observer failed");
    });
    const onError = vi.fn(() => {
      controller.request([2, 0, 0]);
      throw new Error("error observer failed");
    });
    controller = createGaussianSortController({
      backend: controlled.backend,
      onOrder,
      onError,
    });
    controller.request([0, 0, 0]);
    await controlled.call(0).started.promise;
    controlled.call(0).result.resolve(new Uint32Array());
    expect((await controlled.call(1).started.promise).cameraPosition).toEqual([
      1, 0, 0,
    ]);
    controlled.call(1).result.reject(new Error("sort failed"));
    expect((await controlled.call(2).started.promise).cameraPosition).toEqual([
      2, 0, 0,
    ]);
    controller.dispose();
    controlled.call(2).result.resolve(new Uint32Array());
    await controller.whenIdle();
    expect(onOrder).toHaveBeenCalledOnce();
    expect(onError).toHaveBeenCalledOnce();
    expect(controlled.backend.dispose).toHaveBeenCalledOnce();
  });
});

describe("Gaussian sort worker backend", () => {
  const fakeWorker = () => {
    const messages: GaussianSortWorkerRequest[] = [];
    const worker: GaussianSortWorkerLike = {
      onmessage: null,
      onerror: null,
      postMessage: (message) =>
        messages.push(message as GaussianSortWorkerRequest),
      terminate: vi.fn(),
    };
    return { worker, messages };
  };

  it("sends attributes only on scene revision changes and ignores late replies", async () => {
    const { worker, messages } = fakeWorker();
    const backend = createGaussianSortWorkerBackend(worker);
    const job: GaussianSortJob = {
      sceneRevision: 1,
      ranges: [range("tile", [0, 0, 1])],
      cameraPosition: [0, 0, 0],
    };
    const first = backend.sort(job);
    expect(messages.map(({ kind }) => kind)).toEqual(["scene", "sort"]);
    worker.onmessage?.({
      data: { kind: "order", jobId: 1, order: new Uint32Array([0]) },
    } as MessageEvent);
    expect([...(await first)]).toEqual([0]);
    const second = backend.sort({ ...job, cameraPosition: [1, 0, 0] });
    expect(messages.map(({ kind }) => kind)).toEqual(["scene", "sort", "sort"]);
    worker.onmessage?.({
      data: { kind: "order", jobId: 1, order: new Uint32Array([99]) },
    } as MessageEvent);
    worker.onmessage?.({
      data: { kind: "order", jobId: 2, order: new Uint32Array([0]) },
    } as MessageEvent);
    expect([...(await second)]).toEqual([0]);
    const third = backend.sort({ ...job, sceneRevision: 2 });
    expect(messages.map(({ kind }) => kind)).toEqual([
      "scene",
      "sort",
      "sort",
      "scene",
      "sort",
    ]);
    backend.dispose();
    await expect(third).rejects.toThrow("disposed");
    expect(worker.terminate).toHaveBeenCalledOnce();
    expect(worker.onmessage).toBe(null);
  });

  it("contains a worker crash and rejects future dispatch", async () => {
    const { worker } = fakeWorker();
    const backend = createGaussianSortWorkerBackend(worker);
    const job: GaussianSortJob = {
      sceneRevision: 1,
      ranges: [],
      cameraPosition: [0, 0, 0],
    };
    const active = backend.sort(job);
    worker.onerror?.({ message: "worker crashed" } as ErrorEvent);
    await expect(active).rejects.toThrow("worker crashed");
    await expect(backend.sort(job)).rejects.toThrow("disposed");
    backend.dispose();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it("contains malformed worker messages and rejects invalid matching replies", async () => {
    const { worker } = fakeWorker();
    const backend = createGaussianSortWorkerBackend(worker);
    const job: GaussianSortJob = {
      sceneRevision: 1,
      ranges: [],
      cameraPosition: [0, 0, 0],
    };
    const active = backend.sort(job);
    expect(() =>
      worker.onmessage?.({ data: null } as MessageEvent),
    ).not.toThrow();
    worker.onmessage?.({
      data: { kind: "order", jobId: 1, order: [0] },
    } as MessageEvent);
    await expect(active).rejects.toThrow("invalid response");
    backend.dispose();
  });
});
