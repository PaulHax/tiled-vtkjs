import { recoveryOffsets } from "./quality.mjs";
import { foregroundFraction } from "./visual.mjs";
import { viewers } from "./viewers.mjs";

const interpolate = (from, to, fraction) =>
  Array.isArray(from)
    ? from.map((value, index) => interpolate(value, to[index], fraction))
    : from + (to - from) * fraction;

const poseAt = (config, state) => {
  const camera = config.camera;
  if (config.route) {
    const [radius, heading, elevation, ...target] = state;
    const horizontal = radius * Math.cos(elevation);
    return {
      target,
      up: [0, 0, 1],
      position: [
        target[0] + horizontal * Math.cos(heading),
        target[1] + horizontal * Math.sin(heading),
        target[2] + radius * Math.sin(elevation),
      ],
    };
  }
  if (config.camera.type === "map") {
    const zoom = interpolate(camera.farZoom, camera.nearZoom, state[0]);
    return {
      center: [
        camera.center[0] + (camera.panDegrees ?? 0) * state[1],
        camera.center[1],
      ],
      zoom,
      resolution:
        camera.farResolution === undefined
          ? undefined
          : camera.farResolution * 2 ** (camera.farZoom - zoom),
    };
  }
  const angle = state[1] * Math.PI;
  const direction = camera.direction ?? [0, -1, 0.55];
  const axis = camera.up ?? [0, 0, 1];
  const up = axis.map((value) => value / Math.hypot(...axis));
  const [x, y, z] = direction.map((value) => value / Math.hypot(...direction));
  const [a, b, c] = up;
  const cross = [b * z - c * y, c * x - a * z, a * y - b * x];
  const dot = a * x + b * y + c * z;
  const distance = interpolate(
    camera.farDistance,
    camera.nearDistance,
    state[0],
  );
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const target = camera.target.map(
    (value, axis) => value + (camera.panOffset?.[axis] ?? 0) * (state[2] ?? 0),
  );
  return {
    target,
    up,
    position: [x, y, z].map(
      (value, index) =>
        target[index] +
        distance *
          (value * cosine +
            cross[index] * sine +
            up[index] * dot * (1 - cosine)),
    ),
  };
};

export const initialBenchmarkPose = (config) =>
  poseAt(config, config.route?.keyframes[0].slice(1) ?? [0, 0]);

export async function initialize(config) {
  const viewer = await viewers[config.viewer](config);
  const samples = [];
  let lastState;
  const frame = (state) => {
    if (
      !lastState ||
      !state.every((value, index) => value === lastState[index])
    ) {
      viewer.pose(poseAt(config, state));
      lastState = state;
    }
    viewer.draw?.();
    return viewer.stats();
  };

  const nextFrame = () => new Promise(requestAnimationFrame);
  const ready = async () => {
    for (;;) {
      await nextFrame();
      if (frame(config.route?.keyframes[0].slice(1) ?? [0, 0]).ready) return;
    }
  };
  await ready();

  const canvas =
    typeof viewer.canvas === "function" ? viewer.canvas() : viewer.canvas;
  if (!canvas) throw new Error("No viewer canvas");
  canvas.dataset.benchmarkCanvas = "";
  const gl = globalThis.benchmarkContexts.get(canvas);
  const extension = gl?.getExtension("WEBGL_debug_renderer_info");
  const gpu = extension
    ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL)
    : null;

  const probe =
    config.coverage && !config.checkpoints
      ? new OffscreenCanvas(96, 72).getContext("2d", {
          willReadFrequently: true,
        })
      : null;
  if (config.coverage && !gl?.getContextAttributes()?.preserveDrawingBuffer)
    throw new Error("Coverage pass requires a preserved WebGL drawing buffer");
  const coverage = () => {
    probe.clearRect(0, 0, 96, 72);
    probe.drawImage(canvas, 0, 0, 96, 72);
    return foregroundFraction(
      probe.getImageData(0, 0, 96, 72).data,
      4,
      config.background,
    );
  };

  const checkpoints = [];
  const capture = (label, atMs, elapsed, exact, recovery) => {
    checkpoints.push({
      label,
      atMs,
      elapsed,
      pose: poseAt(config, exact),
      ...(recovery ? { recovery } : {}),
      image: createImageBitmap(canvas),
    });
  };
  const exactPose = (keyframes, at) => {
    const checkpoint = keyframes.find(([time]) => time === at);
    if (!checkpoint)
      throw new Error("Checkpoint must coincide with a route keyframe");
    return checkpoint.slice(1);
  };
  const scheduled =
    config.checkpoints && config.route
      ? config.route.checkpoints
          .flatMap(([poseAtMs, label]) => {
            const index = config.route.phases.findIndex(
              ([, phase]) => phase === label,
            );
            if (!["inspect-a", "inspect-b", "overview-return"].includes(label))
              return [{ at: poseAtMs, label, poseAtMs }];
            const [start] = config.route.phases[index];
            const end =
              config.route.phases[index + 1]?.[0] ??
              config.route.keyframes.at(-1)[0];
            const recovery = recoveryOffsets(end - start).map((offset) => ({
              at: start + offset,
              label,
              poseAtMs,
              recovery: { phaseStartMs: start },
            }));
            return [...recovery, { at: poseAtMs, label, poseAtMs }];
          })
          .sort((a, b) => a.at - b.at)
      : [];
  globalThis.referenceBenchmark = {
    async takeCheckpoints() {
      const output = document.createElement("canvas");
      output.width = canvas.width;
      output.height = canvas.height;
      const renderer = output.getContext("bitmaprenderer");
      const captures = [];
      for (const { image, ...checkpoint } of checkpoints) {
        renderer.transferFromImageBitmap(await image);
        captures.push({ ...checkpoint, image: output.toDataURL() });
      }
      return captures;
    },
    gpu,
    navigationToReadyMs: performance.now(),
    canvas: { width: canvas.width, height: canvas.height },
    async settleReference() {
      for (const [at, label] of config.route.checkpoints) {
        const exact = exactPose(config.route.keyframes, at);
        const start = performance.now();
        let settledSince;
        let progressAt = start;
        let progress = "";
        for (;;) {
          const now = await nextFrame();
          const snapshot = frame(exact);
          const { ready, settled } = snapshot;
          const current = JSON.stringify([
            snapshot.selected,
            snapshot.readyRequests,
            snapshot.gaussianSorts,
            snapshot.frontierRebuilds,
            snapshot.packingUploadBytes,
          ]);
          if (current !== progress) {
            progress = current;
            progressAt = now;
          }
          if (ready && settled && (settledSince ??= now) + 500 <= now) {
            capture(label, at, now - start, exact);
            break;
          }
          if (!ready || !settled) settledSince = undefined;
          if (now - progressAt > 30000 || now - start > 180000) {
            const state = { ...snapshot };
            delete state.selected;
            delete state.desired;
            delete state.requested;
            throw new Error(
              "Reference did not settle at " +
                label +
                " " +
                JSON.stringify(state),
            );
          }
        }
      }
    },
    async run(keyframes) {
      samples.length = 0;
      let segment = 1;
      let checkpointIndex = 0;
      const start = await nextFrame();
      if (config.trace)
        performance.mark("tile-detail:route-start", { startTime: start });
      const duration = keyframes.at(-1)[0];
      let last = start;
      for (;;) {
        const now = await nextFrame();
        const elapsed = Math.min(now - start, duration);
        while (keyframes[segment][0] < elapsed) segment++;
        const [at, ...from] = keyframes[segment - 1];
        const [end, ...to] = keyframes[segment];
        const fraction = (elapsed - at) / (end - at);
        const state = config.route
          ? interpolate(from, to, fraction)
          : interpolate(
              [from[0], from[1], from[2] ?? 0],
              [to[0], to[1], to[2] ?? 0],
              fraction,
            );
        const begin = performance.now();
        const snapshot = frame(state);
        samples.push({
          elapsed,
          intervalMs: now - last,
          mainThreadMs: performance.now() - begin,
          ...(probe ? { foregroundFraction: coverage() } : {}),
          ...snapshot,
        });
        let due;
        // Skipped recovery samples stay missing when the page stalls.
        while ((due = scheduled[checkpointIndex]) && elapsed >= due.at) {
          checkpointIndex++;
          if (now - start - due.at >= 125) {
            if (due.recovery) continue;
            throw new Error("Missed scheduled quality checkpoint " + due.label);
          }
          const exact = exactPose(keyframes, due.poseAtMs);
          frame(exact);
          if (!viewer.draw) {
            await nextFrame();
            await nextFrame();
          }
          const capturedAt = performance.now() - start;
          if (capturedAt - due.at >= 125) {
            if (due.recovery) continue;
            throw new Error("Missed scheduled quality checkpoint " + due.label);
          }
          capture(due.label, due.at, capturedAt, exact, due.recovery);
        }
        last = now;
        if (elapsed === duration) return samples;
      }
    },
  };
}
