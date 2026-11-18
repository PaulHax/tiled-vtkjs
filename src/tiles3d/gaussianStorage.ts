export const GAUSSIAN_ATTRIBUTE_BYTES_PER_TUPLE =
  14 * Float32Array.BYTES_PER_ELEMENT;

export type GaussianBounds = [number, number, number, number, number, number];

/** Stable tuple ranges; retirement is committed only after a draw-order swap. */
export const createGaussianStorage = () => {
  let bufferRevision = 0;
  let allocationRevision = 0;
  let capacity = 0;
  let positions: Float32Array = new Float32Array();
  let scales: Float32Array = new Float32Array();
  let rotations: Float32Array = new Float32Array();
  let colors: Float32Array = new Float32Array();
  let origin: [number, number, number] | null = null;
  const allocations = new Map<
    string,
    { offset: number; count: number; revision: number }
  >();
  let free: { offset: number; count: number }[] = [];
  let pending: [number, number][] = [];
  const packed = new Map<number, GaussianBounds>();
  const progress = new Map<number, { count: number; bounds: GaussianBounds }>();

  const grow = (target: number): void => {
    if (target <= capacity) return;
    positions = new Float32Array(target * 3);
    scales = new Float32Array(target * 3);
    rotations = new Float32Array(target * 4);
    colors = new Float32Array(target * 4);
    bufferRevision++;
    free.push({ offset: capacity, count: target - capacity });
    capacity = target;
    coalesce();
    packed.clear();
    progress.clear();
    pending = [];
  };

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
    identity: Symbol(),
    get bufferRevision() {
      return bufferRevision;
    },
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
      return capacity * GAUSSIAN_ATTRIBUTE_BYTES_PER_TUPLE;
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
          // Allocation joins the growth tail to an adjacent free range first.
          ranges.sort((left, right) => left.offset - right.offset);
          for (let index = ranges.length - 1; index > 0; index--) {
            const previous = ranges[index - 1]!;
            const current = ranges[index]!;
            if (previous.offset + previous.count === current.offset) {
              previous.count += current.count;
              ranges.splice(index, 1);
            }
          }
          range = ranges.find((entry) => entry.count >= request.count)!;
        }
        range.offset += request.count;
        range.count -= request.count;
      }
      return projected;
    },
    allocate(
      key: string,
      count: number,
    ): { offset: number; count: number; revision: number; fresh: boolean } {
      const existing = allocations.get(key);
      if (existing) return { ...existing, fresh: false };
      let range = free.find((entry) => entry.count >= count);
      if (!range) {
        const needed = capacity + count;
        let target = Math.max(1, capacity);
        while (target < needed) target *= 2;
        grow(target);
        range = free.find((entry) => entry.count >= count)!;
      }
      const allocation = {
        offset: range.offset,
        count,
        revision: ++allocationRevision,
      };
      range.offset += count;
      range.count -= count;
      free = free.filter((entry) => entry.count > 0);
      allocations.set(key, allocation);
      return { ...allocation, fresh: true };
    },
    grow,
    isPacked: (revision: number): boolean => packed.has(revision),
    packedBounds: (revision: number) => packed.get(revision),
    packingProgress: (revision: number) => progress.get(revision),
    markPacked(
      revision: number,
      bounds: GaussianBounds,
      offset: number,
      count: number,
    ): void {
      progress.delete(revision);
      packed.set(revision, bounds);
      pending.push([offset, count]);
    },
    markPartial(revision: number, count: number, bounds: GaussianBounds): void {
      progress.set(revision, { count, bounds });
    },
    owns(key: string, revision: number): boolean {
      return allocations.get(key)?.revision === revision;
    },
    /** Obsolete sort results must already be invalidated before releasing ranges. */
    retireExcept(live: ReadonlySet<string>): void {
      for (const [key, allocation] of allocations)
        if (!live.has(key)) {
          allocations.delete(key);
          packed.delete(allocation.revision);
          progress.delete(allocation.revision);
          free.push(allocation);
        }
      coalesce();
    },
    takePackedUpdates(): [number, number][] {
      const ranges = pending;
      pending = [];
      return ranges;
    },
    clear(): void {
      bufferRevision++;
      capacity = 0;
      positions = new Float32Array();
      scales = new Float32Array();
      rotations = new Float32Array();
      colors = new Float32Array();
      origin = null;
      allocations.clear();
      packed.clear();
      progress.clear();
      free = [];
      pending = [];
    },
  };
  return api;
};
