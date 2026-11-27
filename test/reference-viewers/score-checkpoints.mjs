import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import sharp from "sharp";
import {
  measureCheckpoint,
  measureRecovery,
  passesQuality,
  recoveryOffsets,
} from "./quality.mjs";

const [runFile, oracleFile, policyFile] = process.argv.slice(2);
if (!runFile || !oracleFile || !policyFile)
  throw new Error(
    "Usage: node score-checkpoints.mjs <quality run> <converged oracle> <calibrated policy>",
  );
const json = async (path) => JSON.parse(await readFile(path, "utf8"));
const [run, oracle, policy] = await Promise.all(
  [runFile, oracleFile, policyFile].map(json),
);
if (
  run.status !== "ok" ||
  oracle.status !== "ok" ||
  run.asset !== oracle.asset ||
  !run.assetCorpusSha256 ||
  run.assetCorpusSha256 !== oracle.assetCorpusSha256 ||
  JSON.stringify(run.clearColor) !== JSON.stringify(policy.background) ||
  JSON.stringify(run.clearColor) !== JSON.stringify(oracle.clearColor) ||
  run.viewer !== oracle.viewer ||
  run.gpu !== oracle.gpu ||
  JSON.stringify(run.options.route) !== JSON.stringify(oracle.options.route) ||
  run.timingMode !== "coverage readback" ||
  oracle.timingMode !== "settled quality reference"
)
  throw new Error(
    "Checkpoints require successful same-engine quality runs of the same frozen asset corpus",
  );
const decode = (file, image) =>
  sharp(resolve(dirname(file), image))
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
const compare = async (label, candidate, reference, { crop, ...limits }) => {
  if (
    !candidate ||
    !reference ||
    JSON.stringify(candidate.pose) !== JSON.stringify(reference.pose)
  )
    throw new Error("Missing or mismatched checkpoint pose " + label);
  if (run.phases?.route?.file === candidate.file)
    throw new Error(
      "Checkpoint was overwritten by the route screenshot: " + label,
    );
  const [a, b] = await Promise.all([
    decode(runFile, candidate.file),
    decode(oracleFile, reference.file),
  ]);
  if (a.info.width !== b.info.width || a.info.height !== b.info.height)
    throw new Error("Mismatched viewport at " + label);
  const metrics = measureCheckpoint(
    a.data,
    b.data,
    a.info.width,
    a.info.height,
    policy.background,
    crop,
  );
  const full = crop
    ? measureCheckpoint(
        a.data,
        b.data,
        a.info.width,
        a.info.height,
        policy.background,
      )
    : metrics;
  return {
    label,
    ...metrics,
    fullCoverageRecall: full.coverageRecall,
    fullCoveragePrecision: full.coveragePrecision,
    passed:
      passesQuality(metrics, limits) &&
      full.coverageRecall !== null &&
      full.coverageRecall >= limits.minimumRecall &&
      full.coveragePrecision >= limits.minimumPrecision,
  };
};
const checkpoints = [];
const recovery = [];
const rules = Object.entries(policy.checkpoints);
if (!rules.length || rules.length !== run.checkpoints?.length)
  throw new Error("Quality policy must cover every captured checkpoint");
for (const [label, rule] of rules) {
  const candidate = run.checkpoints.find((c) => c.label === label);
  const reference = oracle.checkpoints.find((c) => c.label === label);
  checkpoints.push(await compare(label, candidate, reference, rule));
  const samples = [];
  for (const frame of run.recovery ?? []) {
    if (frame.label !== label) continue;
    const { passed } = await compare(label, frame, reference, rule);
    const { phaseStartMs } = frame.recovery;
    samples.push({
      atMs: frame.elapsed - phaseStartMs,
      scheduledMs: frame.atMs - phaseStartMs,
      passed,
    });
  }
  if (
    !samples.length &&
    !["inspect-a", "inspect-b", "overview-return"].includes(label)
  )
    continue;
  if (!Number.isFinite(rule.maximumRecoveryMs) || rule.maximumRecoveryMs < 0)
    throw new Error("Set maximumRecoveryMs for " + label);
  const { phases, keyframes } = run.options.route;
  const phaseIndex = phases.findIndex(([, phase]) => phase === label);
  const duration =
    (phases[phaseIndex + 1]?.[0] ?? keyframes.at(-1)[0]) -
    phases[phaseIndex][0];
  const expectedEnd = recoveryOffsets(duration).at(-1);
  recovery.push({
    label,
    ...measureRecovery(samples, expectedEnd, rule.maximumRecoveryMs),
    samples,
  });
}
console.log(JSON.stringify({ checkpoints, recovery }, null, 2));
if ([...checkpoints, ...recovery].some(({ passed }) => !passed))
  process.exitCode = 1;
