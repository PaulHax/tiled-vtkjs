/** Stable tuple ranges; retirement is committed only after a draw-order swap. */
export const createGaussianStorage = () => {
  let capacity = 0;
  let positions: Float32Array = new Float32Array();
  let scales: Float32Array = new Float32Array();
  let rotations: Float32Array = new Float32Array();
  let colors: Float32Array = new Float32Array();
  let origin: [number, number, number] | null = null;
  const allocations = new Map<string, { offset: number; count: number }>();
  let free: { offset: number; count: number }[] = [];
  let pending: [number, number][] = [];
  let grew = false;

  const coalesce = (): void => {
    free.sort((left, right) => left.offset - right.offset);
    const merged: typeof free = [];
    for (const range of free) {
      const previous = merged.at(-1);
      if (previous && previous.offset + previous.count === range.offset)
        previous.count += range.count;
      else merged.push({ ...range });
    }
    free = merged;
  };

  const api = {
    get positions() {
      return positions;
    },
    get scales() {
      return scales;
    },
    get rotations() {
      return rotations;
    },
    get colors() {
      return colors;
    },
    get capacity() {
      return capacity;
    },
    get bytes() {
      return capacity * 14 * Float32Array.BYTES_PER_ELEMENT;
    },
    get origin(): [number, number, number] {
      return origin ?? [0, 0, 0];
    },
    setOrigin(value: [number, number, number]): void {
      origin ??= value;
    },
    requiredCapacity(
      requests: readonly { key: string; count: number }[],
    ): number {
      let projected = capacity;
      const ranges = free.map((entry) => ({ ...entry }));
      for (const request of requests) {
        if (allocations.has(request.key)) continue;
        let range = ranges.find((entry) => entry.count >= request.count);
        if (!range) {
          const old = projected;
          projected = Math.max(1, projected);
          while (projected < old + request.count) projected *= 2;
          range = { offset: old, count: projected - old };
          ranges.push(range);
        }
        range.offset += request.count;
        range.count -= request.count;
      }
      return projected;
    },
    allocate(
      key: string,
      count: number,
    ): { offset: number; count: number; fresh: boolean } {
      const existing = allocations.get(key);
      if (existing) return { ...existing, fresh: false };
      let range = free.find((entry) => entry.count >= count);
      if (!range) {
        const oldCapacity = capacity;
        const needed = oldCapacity + count;
        capacity = Math.max(1, oldCapacity);
        while (capacity < needed) capacity *= 2;
        const grow = (old: Float32Array, components: number): Float32Array => {
          const next = new Float32Array(capacity * components);
          next.set(old);
          return next;
        };
        positions = grow(positions, 3);
        scales = grow(scales, 3);
        rotations = grow(rotations, 4);
        colors = grow(colors, 4);
        free.push({ offset: oldCapacity, count: capacity - oldCapacity });
        coalesce();
        grew = true;
        range = free.find((entry) => entry.count >= count)!;
      }
      const allocation = { offset: range.offset, count };
      range.offset += count;
      range.count -= count;
      free = free.filter((entry) => entry.count > 0);
      allocations.set(key, allocation);
      pending.push([allocation.offset, count]);
      return { ...allocation, fresh: true };
    },
    /** Obsolete sort results must already be invalidated before releasing ranges. */
    retireExcept(live: ReadonlySet<string>): void {
      for (const [key, allocation] of allocations)
        if (!live.has(key)) {
          allocations.delete(key);
          free.push(allocation);
        }
      coalesce();
    },
    updates(): { grew: boolean; ranges: [number, number][] } {
      const result = { grew, ranges: pending };
      pending = [];
      grew = false;
      return result;
    },
    mark(offset: number, count: number): void {
      pending.push([offset, count]);
    },
    clear(): void {
      capacity = 0;
      positions = new Float32Array();
      scales = new Float32Array();
      rotations = new Float32Array();
      colors = new Float32Array();
      origin = null;
      allocations.clear();
      free = [];
      pending = [];
      grew = true;
    },
  };
  return api;
};
