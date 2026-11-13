/**
 * Where each decoded point tile lives once its read has delivered it: in
 * renderer residency, or in a byte-bounded CPU cache for cheap reselection.
 * A payload has one owner at a time. A second copy would double-count
 * decoded bytes and spend cache capacity on a tile the renderer already
 * holds, so every move goes through `hold`, `releaseExcept`, `park` and
 * `promote`.
 *
 * The consumer sees residency as deltas: `flush` hands it the difference
 * between what is resident now and what it was last handed, so a key made
 * resident and released again between two flushes never reaches it, and no
 * batch both adds and removes one key.
 */

import { createLruCache } from "../core/lru";
import { keyFromString, type VoxelKey } from "./octree";
import { tileBytes, type TileData } from "./tileSource";
import {
  ResourcePolicyError,
  type MemoryOwner,
  type MemoryLease,
} from "../core/memoryResources";

export type TileBatch = {
  readonly added: readonly {
    key: VoxelKey;
    tile: TileData;
    memory?: MemoryLease;
  }[];
  readonly removed: readonly VoxelKey[];
};

export type ResidencyTotals = {
  readonly residentTiles: number;
  readonly residentPoints: number;
  readonly residentBytes: number;
  readonly cachedTiles: number;
  readonly cachedBytes: number;
};

export type PayloadResidency = {
  /** Make a payload resident; the cache gives up any copy of the key. */
  hold(key: string, tile: TileData): boolean;
  /** Move every resident payload the set does not name into the cache. */
  releaseExcept(keep: ReadonlySet<string>): void;
  /** Cache a payload, unless residency already owns the key. */
  park(key: string, tile: TileData): void;
  /** Make a cached payload resident; false when the cache has none. */
  promote(key: string): boolean;
  isResident(key: string): boolean;
  /** Whether a payload for the key is resident or cached. */
  isDecoded(key: string): boolean;
  residentKeys(): IterableIterator<string>;
  /** What the consumer was last handed. */
  submitted(): ReadonlyMap<string, TileData>;
  /**
   * The change since the last flush, or null when there is none. The
   * consumer is then taken to hold exactly what is resident.
   */
  flush(): TileBatch | null;
  totals(): ResidencyTotals;
  /**
   * Drop every payload. The consumer keeps what it was handed until the next
   * flush takes it back.
   */
  clear(): void;
  /** Stop tracking what the consumer holds, and return it for removal. */
  takeBack(): VoxelKey[];
  trimCache(bytes: number): void;
  retryBlocked(): void;
  blocked(key: string): boolean;
};

export const createPayloadResidency = (options: {
  readonly cacheBytes: number;
  readonly memory?: MemoryOwner;
  readonly reserveRenderer?: (
    key: string,
    tile: TileData,
  ) => MemoryLease | undefined;
  readonly onPolicyRefusal?: (error: ResourcePolicyError) => void;
}): PayloadResidency => {
  const allocations = new Map<
    TileData,
    { memory: MemoryLease; references: number }
  >();
  const rendererMemory = new Map<string, MemoryLease>();
  const blocked = new Set<string>();
  const holdCpu = (tile: TileData): void => {
    if (!options.memory) return;
    const entry = allocations.get(tile);
    if (entry) {
      entry.references += 1;
      return;
    }
    allocations.set(tile, {
      memory: options.memory.retain(
        [
          tile.positions.buffer as ArrayBuffer,
          ...(tile.rgb ? [tile.rgb.buffer as ArrayBuffer] : []),
        ],
        64,
      ),
      references: 1,
    });
  };
  const releaseCpu = (tile: TileData): void => {
    const entry = allocations.get(tile);
    if (!entry) return;
    entry.references -= 1;
    if (entry.references === 0) {
      entry.memory.release();
      allocations.delete(tile);
    }
  };
  const resident = new Map<string, TileData>();
  let residentPoints = 0;
  let residentBytes = 0;
  const cache = createLruCache<string, TileData>({
    maxBytes: options.cacheBytes,
    onEvict: (_key, tile) => releaseCpu(tile),
  });
  let submitted = new Map<string, TileData>();

  const park = (key: string, tile: TileData): void => {
    if (resident.has(key)) return;
    holdCpu(tile);
    cache.delete(key);
    cache.set(key, tile, tileBytes(tile));
  };
  const hold = (key: string, tile: TileData): boolean => {
    if (resident.get(key) === tile) return true;
    if (resident.has(key))
      throw new Error("resident tile key already owns a different payload");
    let rendererLease: MemoryLease | undefined;
    try {
      rendererLease = options.reserveRenderer?.(key, tile);
      holdCpu(tile);
    } catch (error) {
      rendererLease?.release();
      if (!(error instanceof ResourcePolicyError)) throw error;
      blocked.add(key);
      options.onPolicyRefusal?.(error);
      try {
        park(key, tile);
      } catch (cacheError) {
        if (!(cacheError instanceof ResourcePolicyError)) throw cacheError;
      }
      return false;
    }
    cache.delete(key);
    if (rendererLease) rendererMemory.set(key, rendererLease);
    resident.set(key, tile);
    residentPoints += tile.pointCount;
    residentBytes += tileBytes(tile);
    blocked.delete(key);
    return true;
  };

  return {
    hold,

    releaseExcept(keep) {
      // Deleting the key being visited is safe while iterating a Map.
      for (const [key, tile] of resident) {
        if (keep.has(key)) continue;
        resident.delete(key);
        residentPoints -= tile.pointCount;
        residentBytes -= tileBytes(tile);
        rendererMemory.get(key)?.release();
        rendererMemory.delete(key);
        // Transfer the resident CPU reference into the reuse cache.
        cache.delete(key);
        cache.set(key, tile, tileBytes(tile));
      }
    },

    park(key, tile) {
      try {
        park(key, tile);
      } catch (error) {
        if (!(error instanceof ResourcePolicyError)) throw error;
        blocked.add(key);
        options.onPolicyRefusal?.(error);
      }
    },

    promote(key) {
      const cached = cache.get(key);
      if (cached === undefined) return false;
      return hold(key, cached);
    },

    isResident: (key) => resident.has(key),

    isDecoded: (key) => resident.has(key) || cache.has(key),

    residentKeys: () => resident.keys(),

    submitted: () => submitted,

    flush() {
      const added: { key: VoxelKey; tile: TileData; memory?: MemoryLease }[] =
        [];
      const removed: VoxelKey[] = [];
      for (const [key, tile] of resident) {
        // A key whose payload was replaced is an addition, never a
        // remove/add pair.
        if (submitted.get(key) === tile) continue;
        added.push({
          key: keyFromString(key),
          tile,
          ...(rendererMemory.has(key)
            ? { memory: rendererMemory.get(key)!.share() }
            : {}),
        });
      }
      for (const key of submitted.keys()) {
        if (!resident.has(key)) removed.push(keyFromString(key));
      }
      if (added.length === 0 && removed.length === 0) return null;
      for (const [key, tile] of resident)
        if (submitted.get(key) !== tile) holdCpu(tile);
      for (const [key, tile] of submitted)
        if (resident.get(key) !== tile) releaseCpu(tile);
      submitted = new Map(resident);
      return { added, removed };
    },

    totals: () => ({
      residentTiles: resident.size,
      residentPoints,
      residentBytes,
      cachedTiles: cache.count(),
      cachedBytes: cache.totalBytes(),
    }),

    clear() {
      for (const tile of resident.values()) releaseCpu(tile);
      for (const memory of rendererMemory.values()) memory.release();
      rendererMemory.clear();
      resident.clear();
      residentPoints = 0;
      residentBytes = 0;
      cache.clear();
      blocked.clear();
    },

    takeBack() {
      const keys = [...submitted.keys()].map(keyFromString);
      for (const tile of submitted.values()) releaseCpu(tile);
      submitted = new Map();
      return keys;
    },
    trimCache: (bytes) => cache.trimTo(bytes),
    retryBlocked() {
      blocked.clear();
    },
    blocked: (key) => blocked.has(key),
  };
};
