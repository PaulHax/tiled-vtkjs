const linear = Float64Array.from({ length: 256 }, (_, byte) => {
  const s = byte / 255;
  return s < 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
});

/** All images have the same pose, viewport, renderer and clear color. */
export function measureCheckpoint(
  actual,
  reference,
  width,
  height,
  background,
  crop = [0, 0, 1, 1],
) {
  if (
    actual.length !== width * height * 3 ||
    actual.length !== reference.length
  )
    throw new Error("Quality inputs must be equal-size RGB images");
  if (
    crop.length !== 4 ||
    crop.some((value) => !Number.isFinite(value) || value < 0 || value > 1) ||
    crop[0] >= crop[2] ||
    crop[1] >= crop[3]
  )
    throw new Error("Quality crop must be a nonempty normalized rectangle");
  let expectedPixels = 0,
    actualPixels = 0,
    sharedPixels = 0,
    unionPixels = 0;
  let rgbError = 0,
    edgeEnergy = 0,
    edgeError = 0;
  const foreground = (image, p) =>
    background.some(
      (value, channel) => Math.abs(image[p + channel] - value) > 12,
    );
  const luminance = (image, p) =>
    0.2126 * linear[image[p]] +
    0.7152 * linear[image[p + 1]] +
    0.0722 * linear[image[p + 2]];
  const x0 = Math.floor(crop[0] * width);
  const y0 = Math.floor(crop[1] * height);
  const x1 = Math.ceil(crop[2] * width);
  const y1 = Math.ceil(crop[3] * height);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const p = (y * width + x) * 3;
      const expected = foreground(reference, p);
      const drawn = foreground(actual, p);
      expectedPixels += Number(expected);
      actualPixels += Number(drawn);
      sharedPixels += Number(expected && drawn);
      if (expected || drawn) {
        unionPixels++;
        for (let c = 0; c < 3; c++)
          rgbError += Math.abs(
            linear[actual[p + c]] - linear[reference[p + c]],
          );
      }
      for (const stride of [x + 1 < x1 ? 3 : 0, y + 1 < y1 ? width * 3 : 0]) {
        if (!stride) continue;
        const referenceEdge =
          luminance(reference, p + stride) - luminance(reference, p);
        const actualEdge = luminance(actual, p + stride) - luminance(actual, p);
        edgeEnergy += Math.abs(referenceEdge);
        edgeError += Math.abs(referenceEdge - actualEdge);
      }
    }
  }
  return {
    coverageRecall: expectedPixels ? sharedPixels / expectedPixels : null,
    coveragePrecision: actualPixels ? sharedPixels / actualPixels : 0,
    meanLinearRgbError: unionPixels ? rgbError / (unionPixels * 3) : null,
    edgeErrorRatio: edgeEnergy > 0 ? edgeError / edgeEnergy : null,
  };
}

export function passesQuality(metrics, limits) {
  return Object.entries({
    coverageRecall: (value) => value >= limits.minimumRecall,
    coveragePrecision: (value) => value >= limits.minimumPrecision,
    meanLinearRgbError: (value) => value <= limits.maximumColorError,
    edgeErrorRatio: (value) => value <= limits.maximumEdgeError,
  }).every(
    ([metric, check]) =>
      Number.isFinite(metrics[metric]) && check(metrics[metric]),
  );
}

const recoveryIntervalMs = 250;

export const recoveryOffsets = (durationMs) =>
  Array.from(
    { length: Math.ceil((durationMs - 100) / recoveryIntervalMs) },
    (_, index) => index * recoveryIntervalMs,
  );

export function measureRecovery(samples, lastScheduledMs, maximumRecoveryMs) {
  let first;
  let last;
  let lastScheduled;
  for (const { atMs, scheduledMs, passed } of samples) {
    if (
      !passed ||
      (last !== undefined && atMs - last > 400) ||
      (lastScheduled !== undefined &&
        scheduledMs - lastScheduled > recoveryIntervalMs)
    )
      first = undefined;
    if (passed) first ??= atMs;
    last = atMs;
    lastScheduled = scheduledMs;
  }
  const observedEnd = samples.at(-1)?.scheduledMs === lastScheduledMs;
  const recoveredMs = observedEnd && last - first >= 500 ? first : null;
  return {
    recoveredMs,
    passed: recoveredMs !== null && recoveredMs <= maximumRecoveryMs,
    lastObservedMs: last ?? null,
  };
}
