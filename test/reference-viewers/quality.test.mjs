import { expect, test } from "vitest";
import {
  measureCheckpoint,
  measureRecovery,
  passesQuality,
} from "./quality.mjs";

const size = 32;
const limits = {
  minimumRecall: 0.95,
  minimumPrecision: 0.95,
  maximumColorError: 0.08,
  maximumEdgeError: 0.15,
};
const image = (pixel) =>
  Uint8Array.from({ length: size * size * 3 }, (_, i) =>
    pixel(Math.floor(i / 3) % size, Math.floor(i / (3 * size))),
  );
const pattern = (x, y) =>
  x >= 8 && x < 24 && y >= 8 && y < 24 ? ((x + y) % 2 === 0 ? 90 : 210) : 0;
const reference = image(pattern);
const score = (candidate) =>
  measureCheckpoint(candidate, reference, size, size, [0, 0, 0]);

test("a stable exact-pose capture passes all quality gates", () => {
  expect(passesQuality(score(reference), limits)).toBe(true);
});
test("exact crop comparison counts changed RGB pixels including the background", () => {
  const original = image(() => 0);
  const changed = original.slice();
  changed[0] = 1;
  changed[(8 * size + 8) * 3 + 2] = 1;
  expect(
    measureCheckpoint(
      changed,
      original,
      size,
      size,
      [0, 0, 0],
      [0.25, 0.25, 0.75, 0.75],
    ),
  ).toMatchObject({
    pixels: 256,
    differentPixels: 1,
  });
});
test("quality gates reject blank, blurred, coarsened, and stale appearances", () => {
  const variants = {
    blank: image(() => 0),
    blurred: image((x, y) =>
      Math.round((pattern(x, y) + pattern(x + 1, y)) / 2),
    ),
    coarsened: image((x, y) => (pattern(x, y) ? 150 : 0)),
    stale: image((x, y) => pattern(x - 5, y)),
  };
  expect(score(variants.coarsened).coverageRecall).toBe(1);
  expect(score(variants.coarsened).coveragePrecision).toBe(1);
  for (const [label, variant] of Object.entries(variants)) {
    expect(passesQuality(score(variant), limits), label).toBe(false);
  }
});

test("recovery needs a sustained passing streak reaching the expected terminal sample", () => {
  const samples = [0, 250, 500, 750, 1000].map((scheduledMs) => ({
    scheduledMs,
    atMs: scheduledMs + 20,
    passed: scheduledMs >= 500,
  }));
  expect(measureRecovery(samples, 1000, 600)).toMatchObject({
    recoveredMs: 520,
    passed: true,
  });
  expect(measureRecovery(samples, 1000, 400).passed).toBe(false);
  expect(
    measureRecovery(samples.slice(0, -1), 1000, 600).recoveredMs,
  ).toBeNull();
  expect(
    measureRecovery(
      samples.filter(({ scheduledMs }) => scheduledMs !== 750),
      1000,
      600,
    ).recoveredMs,
  ).toBeNull();
  expect(measureRecovery([], 1000, 600).passed).toBe(false);
  const missing = [
    { scheduledMs: 0, atMs: 120, passed: true },
    { scheduledMs: 500, atMs: 505, passed: true },
    { scheduledMs: 750, atMs: 755, passed: true },
  ];
  expect(measureRecovery(missing, 750, 200).passed).toBe(false);
});
