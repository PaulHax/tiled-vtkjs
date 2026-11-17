import { describe, expect, it } from "vitest";
import { createGaussianStorage } from "./gaussianStorage";

describe("Gaussian resident tuple ranges", () => {
  it("preserves displayed identities and data through admission and storage growth", () => {
    const arena = createGaussianStorage();
    const parent = arena.allocate("parent:1", 3);
    arena.positions.set([1, 2, 3, 4, 5, 6, 7, 8, 9], parent.offset * 3);
    const accepted = arena.positions;
    arena.updates();
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
    expect(arena.updates()).toMatchObject({
      grew: true,
      ranges: [[child.offset, 3]],
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
