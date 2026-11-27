import { describe, expect, it } from "vitest";
import { createGaussianStorage } from "./gaussianStorage";

describe("Gaussian resident tuple ranges", () => {
  it("preserves displayed identities and data through admission and storage growth", () => {
    const arena = createGaussianStorage();
    const parent = arena.allocate("parent:1", 3);
    arena.positions.set([1, 2, 3, 4, 5, 6, 7, 8, 9], parent.offset * 3);
    const accepted = arena.positions;
    const child = arena.allocate("child:1", 3);
    expect(child.offset).toBeGreaterThanOrEqual(parent.offset + parent.count);
    expect(
      Array.from(accepted.slice(parent.offset * 3, (parent.offset + 3) * 3)),
    ).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(arena.positions).not.toBe(accepted);
    expect(arena.allocate("parent:1", 3)).toMatchObject({
      offset: parent.offset,
      fresh: false,
    });
  });

  it("reuses a retired range only after the presentation commits its retirement", () => {
    const arena = createGaussianStorage();
    const parent = arena.allocate("parent:1", 3);
    arena.allocate("child:1", 3);
    arena.retireExcept(new Set(["child:1"]));
    const replacement = arena.allocate("other:2", 3);
    expect(replacement.offset).toBe(parent.offset);
    expect(arena.capacity).toBe(8);
    arena.clear();
    expect(arena.capacity).toBe(0);
    expect(arena.bytes).toBe(0);
  });
});

it("plans the same coalesced growth ranges that allocation will use", () => {
  const arena = createGaussianStorage();
  arena.allocate("a", 4);
  arena.allocate("temporary", 1);
  arena.retireExcept(new Set(["a"]));
  const projected = arena.requiredCapacity([
    { key: "b", count: 5 },
    { key: "c", count: 6 },
  ]);
  arena.allocate("b", 5);
  arena.allocate("c", 6);
  expect(projected).toBe(16);
  expect(projected).toBe(arena.capacity);
});

it("reports dirty texture ranges for packed live tuples", () => {
  const arena = createGaussianStorage();
  arena.grow(8);
  const parent = arena.allocate("parent", 3);
  const child = arena.allocate("child", 3);
  arena.markPacked(
    parent.revision,
    [0, 1, 0, 1, 0, 1],
    parent.offset,
    parent.count,
  );
  arena.markPartial(child.revision, 1, [0, 1, 0, 1, 0, 1]);
  expect(arena.takePackedUpdates()).toEqual([[parent.offset, parent.count]]);
  arena.markPacked(
    child.revision,
    [0, 1, 0, 1, 0, 1],
    child.offset,
    child.count,
  );
  const retired = arena.allocate("retired", 2);
  arena.markPacked(
    retired.revision,
    [0, 1, 0, 1, 0, 1],
    retired.offset,
    retired.count,
  );
  arena.retireExcept(new Set(["parent", "child"]));
  expect(arena.takePackedUpdates()).toEqual([[child.offset, child.count]]);
});

it("charges only the missing tail space and bounds growth to the available capacity", () => {
  const arena = createGaussianStorage();
  arena.grow(65_536);
  arena.allocate("occupied", 60_000);
  const requests = [{ key: "new", count: 6_000 }];
  expect(arena.requiredCapacity(requests)).toBe(73_728);
  expect(arena.requiredCapacity(requests, 67_000)).toBe(67_000);
  expect(arena.requiredCapacity(requests, 64_000)).toBe(66_000);

  arena.grow(arena.requiredCapacity(requests, 67_000));
  expect(arena.allocate("new", 6_000).offset).toBe(60_000);
  expect(arena.capacity).toBe(67_000);
});
