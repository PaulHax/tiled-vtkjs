import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_MEMORY_BUDGET_BYTES,
  createMemoryPool,
  defaultMemoryBudgetBytes,
} from "./memoryPool";
import { ResourcePolicyError } from "./memoryResources";

const MiB = 1024 * 1024;

describe("createMemoryPool", () => {
  it("gives a lone member the whole budget and splits evenly as members join", () => {
    const pool = createMemoryPool({ totalBytes: 900 });
    const a = pool.register();
    expect(a.budgetBytes()).toBe(900);
    const b = pool.register();
    expect(a.budgetBytes()).toBe(450);
    expect(b.budgetBytes()).toBe(450);
    const c = pool.register();
    expect(a.budgetBytes()).toBe(300);
    c.release();
    expect(a.budgetBytes()).toBe(450);
  });

  it("notifies existing members on join and leave, but never the joiner during register", () => {
    const pool = createMemoryPool({ totalBytes: 100 });
    const aChanges = vi.fn();
    const bChanges = vi.fn();
    pool.register(aChanges);
    expect(aChanges).not.toHaveBeenCalled();
    const b = pool.register(bChanges);
    expect(aChanges).toHaveBeenCalledTimes(1);
    expect(bChanges).not.toHaveBeenCalled();
    b.release();
    expect(aChanges).toHaveBeenCalledTimes(2);
  });

  it("release is idempotent and zeroes the released member", () => {
    const pool = createMemoryPool({ totalBytes: 100 });
    const aChanges = vi.fn();
    pool.register(aChanges);
    const b = pool.register();
    b.release();
    b.release();
    expect(b.budgetBytes()).toBe(0);
    expect(aChanges).toHaveBeenCalledTimes(2); // join + one release
    expect(pool.memberCount()).toBe(1);
  });

  it("grants feasible minimum coverage before sharing optional demand", () => {
    const pool = createMemoryPool({ totalBytes: 512 });
    const large = pool.register();
    large.update({ minimumBytes: 320, desiredBytes: 400, residentBytes: 320 });
    const tiny = pool.register();
    tiny.update({ minimumBytes: 1, desiredBytes: 1, residentBytes: 0 });
    expect(large.budgetBytes()).toBe(400);
    expect(tiny.budgetBytes()).toBe(1);
    const impossible = pool.register();
    impossible.update({
      minimumBytes: 256,
      desiredBytes: 256,
      residentBytes: 0,
    });
    expect(impossible.budgetBytes()).toBe(0);
    tiny.release();
    large.release();
    expect(impossible.budgetBytes()).toBe(256);
  });

  it("deduplicates retained buffers independently of scheduling and shared handles", () => {
    const pool = createMemoryPool({ totalBytes: 100, cpuBytes: 100 });
    const active = pool.register();
    const a = pool.createOwner();
    const b = pool.createOwner();
    const buffer = new ArrayBuffer(80);
    const first = a.retain([buffer, buffer]);
    const alias = first.share();
    const second = b.retain([buffer]);
    active.release();
    a.close();
    expect(pool.stats()).toMatchObject({
      memberCount: 0,
      owners: 2,
      cpuRetainedBytes: 80,
    });
    first.release();
    second.release();
    b.close();
    expect(pool.stats()).toMatchObject({ owners: 1, cpuRetainedBytes: 80 });
    alias.release();
    expect(pool.stats()).toMatchObject({ owners: 0, cpuRetainedBytes: 0 });
  });

  it("admits whole replacement groups and rolls back refused peaks", () => {
    const pool = createMemoryPool({ totalBytes: 100, transitionBytes: 20 });
    const a = pool.createOwner();
    const b = pool.createOwner();
    const old = a.reserveRenderer(40);
    const sibling = b.reserveRenderer(40);
    expect(() =>
      a.reserveRendererGroup(
        [
          { rendererBytes: 30, cpuBytes: 0 },
          { rendererBytes: 30, cpuBytes: 0 },
        ],
        [old],
      ),
    ).toThrow(ResourcePolicyError);
    expect(pool.stats().rendererBytes).toBe(80);
    sibling.release();
    const group = a.reserveRendererGroup(
      [
        { rendererBytes: 30, cpuBytes: 0 },
        { rendererBytes: 30, cpuBytes: 0 },
      ],
      [old],
    );
    expect(pool.stats().rendererBytes).toBe(100);
    group[0]!.release();
    expect(() => a.reserveRenderer(1, { replacing: [old] })).toThrow(
      /unclaimed/,
    );
    group[1]!.release();
    old.release();
    a.close();
    b.close();
    expect(pool.stats().owners).toBe(0);
  });

  it("reports replacement eligibility for the complete composite lease lifetime and shared handles", () => {
    const pool = createMemoryPool({ totalBytes: 100, transitionBytes: 40 });
    const owner = pool.createOwner();
    const old = owner.reserveRenderer(40);
    expect(owner.replacementAvailable(old)).toBe(true);
    const [mesh, arena] = owner.reserveRendererGroup(
      [
        { rendererBytes: 30, cpuBytes: 0 },
        { rendererBytes: 30, cpuBytes: 0 },
      ],
      [old],
    );
    const retainedArena = arena!.share();
    expect(owner.replacementAvailable(old)).toBe(false);
    expect(() => owner.reserveRenderer(1, { replacing: [old] })).toThrow(
      /unclaimed/,
    );
    mesh!.release();
    expect(owner.replacementAvailable(old)).toBe(false);
    arena!.shrinkRenderer(0);
    arena!.release();
    expect(pool.stats().rendererBytes).toBe(40);
    expect(owner.replacementAvailable(old)).toBe(false);
    retainedArena.release();
    expect(owner.replacementAvailable(old)).toBe(true);
    old.release();
    owner.close();
    expect(pool.stats()).toMatchObject({ rendererBytes: 0, owners: 0 });
  });

  it("keeps released, foreign and unknown replacement allocations as hard errors", () => {
    const pool = createMemoryPool({ totalBytes: 100 });
    const owner = pool.createOwner();
    const other = pool.createOwner();
    const foreign = other.reserveRenderer(20);
    const released = owner.reserveRenderer(20);
    released.release();
    const unknown = {} as Parameters<typeof owner.replacementAvailable>[0];
    for (const lease of [foreign, released, unknown]) {
      expect(() => owner.replacementAvailable(lease)).toThrow(
        /replacement must own/,
      );
      expect(() => owner.reserveRenderer(1, { replacing: [lease] })).toThrow(
        /replacement must own/,
      );
    }
    foreign.release();
    owner.close();
    other.close();
    expect(pool.stats()).toMatchObject({ rendererBytes: 0, owners: 0 });
  });

  it("retains reserved work bytes independently of admission and owner closure", () => {
    const pool = createMemoryPool({ totalBytes: 100, workBytes: 100 });
    const a = pool.createOwner();
    const b = pool.createOwner();
    const signal = new AbortController();
    const operation = a.work(signal.signal, 80);
    expect(() => b.work(undefined, 40)).toThrow(ResourcePolicyError);
    signal.abort();
    a.close();
    expect(pool.stats()).toMatchObject({
      activeOperations: 0,
      queuedOperations: 0,
      workReservedBytes: 80,
      owners: 2,
    });
    expect(() => b.work(signal.signal, 10)).toThrow(/aborted/i);
    const before = pool.availability();
    operation.release();
    expect(pool.availability()).toBeGreaterThan(before);
    const resumed = b.work(undefined, 40);
    expect(pool.stats()).toMatchObject({
      activeOperations: 0,
      workReservedBytes: 40,
      owners: 1,
    });
    resumed.release();
    b.close();
    expect(pool.stats()).toMatchObject({
      activeOperations: 0,
      workReservedBytes: 0,
      owners: 0,
    });
  });

  it("rechecks buffer identity after reclamation before accepting CPU ownership", () => {
    const pool = createMemoryPool({ totalBytes: 100, cpuBytes: 100 });
    let cached:
      | ReturnType<ReturnType<typeof pool.createOwner>["retain"]>
      | undefined;
    const owner = pool.createOwner({ reclaim: () => cached?.release() });
    const buffer = new ArrayBuffer(80);
    cached = owner.retain([buffer]);
    expect(() => owner.retain([buffer], 30)).toThrow(ResourcePolicyError);
    expect(pool.stats()).toMatchObject({
      cpuRetainedBytes: 0,
      cpuReservedBytes: 0,
    });
    owner.close();
  });

  it("stops reclaiming once the live CPU deficit is covered", () => {
    const pool = createMemoryPool({ totalBytes: 100, cpuBytes: 100 });
    let first: ReturnType<ReturnType<typeof pool.createOwner>["retain"]>;
    const reclaimed = vi.fn();
    const a = pool.createOwner({
      reclaim: (domain, deficit) => {
        expect(domain).toBe("cpu");
        expect(deficit()).toBe(10);
        first.release();
      },
    });
    const b = pool.createOwner({ reclaim: reclaimed });
    first = a.retain([new ArrayBuffer(30)]);
    const other = b.retain([new ArrayBuffer(60)]);
    const next = b.retain([new ArrayBuffer(20)]);
    expect(reclaimed).not.toHaveBeenCalled();
    expect(pool.stats()).toMatchObject({
      cpuRetainedBytes: 80,
      cpuReclaimPasses: 1,
      cpuReclaimedBytes: 30,
    });
    pool.configure({ totalBytes: 110, cpuBytes: 110 });
    expect(reclaimed).not.toHaveBeenCalled();
    other.release();
    next.release();
    a.close();
    b.close();
  });

  it("counts replacement allocation identity once when multiple handles share it", () => {
    const pool = createMemoryPool({ totalBytes: 100, transitionBytes: 100 });
    const owner = pool.createOwner();
    const old = owner.reserveRenderer(50);
    const alias = old.share();
    expect(() =>
      owner.reserveRenderer(120, { replacing: [old, alias] }),
    ).toThrow(ResourcePolicyError);
    expect(pool.stats().rendererBytes).toBe(50);
    const next = owner.reserveRenderer(100, { replacing: [old, alias] });
    old.release();
    alias.release();
    expect(pool.stats().rendererBytes).toBe(100);
    next.release();
    owner.close();
    expect(pool.stats()).toMatchObject({ rendererBytes: 0, owners: 0 });
  });
});

describe("defaultMemoryBudgetBytes", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("falls back to the fixed default without a deviceMemory signal", () => {
    vi.stubGlobal("navigator", {});
    expect(defaultMemoryBudgetBytes()).toBe(DEFAULT_MEMORY_BUDGET_BYTES);
  });

  it("takes an eighth of reported device memory, clamped to [256 MiB, 1 GiB]", () => {
    vi.stubGlobal("navigator", { deviceMemory: 8 });
    expect(defaultMemoryBudgetBytes()).toBe(1024 * MiB);
    vi.stubGlobal("navigator", { deviceMemory: 4 });
    expect(defaultMemoryBudgetBytes()).toBe(512 * MiB);
    vi.stubGlobal("navigator", { deviceMemory: 0.5 });
    expect(defaultMemoryBudgetBytes()).toBe(256 * MiB);
  });
});
