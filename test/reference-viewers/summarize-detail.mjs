import { readFile } from "node:fs/promises";

const [file, ...regions] = process.argv.slice(2);
if (!file)
  throw new Error("Usage: summarize-detail.mjs <trace report> [tile path ...]");
const report = JSON.parse(await readFile(file, "utf8"));
const trace = report.detailTrace;
if (report.status !== "ok" || !trace)
  throw new Error("Expected a successful tiled-vtkjs --trace run");
const starts = trace.events.filter(({ stage }) => stage === "route-start");
if (starts.length !== 1)
  throw new Error("Expected a single recorded route or fixed-pose trace");
const origin = starts[0].at;
const relative = (time) => (time == null ? null : Math.round(time - origin));
const costs = new Map();
const counts = new Map();
const targets = new Map();
const results = new Map();
const preparations = new Map();
const decodes = new Map();
const draws = new Map();
let requested = new Map();
let desired = new Map();
const changes = [];
const keyFor = (event) => JSON.stringify([event.revision, event.camera]);
const updateSelection = (ids, previous, at) =>
  new Map(ids.map((id) => [id, previous.get(id) ?? at]));

for (const event of trace.events) {
  const { stage, at, revision } = event;
  counts.set(stage, (counts.get(stage) ?? 0) + 1);
  switch (stage) {
    case "selection":
      requested = updateSelection(event.requested, requested, at);
      desired = updateSelection(event.desired, desired, at);
      break;
    case "decode-start":
      decodes.set(event.tile, { start: at, url: event.url });
      break;
    case "decode-end": {
      const decode = decodes.get(event.tile);
      if (decode) decode.end = at;
      break;
    }
    case "sort-target":
      targets.set(revision, event);
      break;
    case "result":
      results.set(keyFor(event), at);
      break;
    case "prepare":
      preparations.set(keyFor(event), at);
      break;
    case "preparation-cost":
      for (const [name, metrics] of Object.entries(event.stages)) {
        const sum = costs.get(name) ?? {
          activeMs: 0,
          waitMs: 0,
          jobs: 0,
          bytes: 0,
        };
        for (const field of Object.keys(sum)) sum[field] += metrics[field];
        costs.set(name, sum);
      }
      break;
    case "paint": {
      const target = targets.get(revision);
      if (!target)
        throw new Error("Missing target for painted revision " + revision);
      for (const region of regions) {
        const covering = target.tiles.filter(
          (tile) => tile === region || region.startsWith(tile + "/"),
        );
        const tile = covering.sort((a, b) => b.length - a.length)[0] ?? null;
        if (draws.has(region) && draws.get(region) === tile) continue;
        draws.set(region, tile);
        const decode = decodes.get(tile);
        const request =
          decode &&
          trace.requests.findLast(
            ({ url, end }) => url === decode.url && end <= decode.start,
          );
        changes.push({
          region,
          tile,
          revision,
          desiredSinceMs: relative(desired.get(tile)),
          requestedSinceMs: relative(requested.get(tile)),
          fetchStartMs: relative(request?.start),
          fetchEndMs: relative(request?.end),
          decodeStartMs: relative(decode?.start),
          decodeEndMs: relative(decode?.end),
          targetMs: relative(target.at),
          sortResultMs: relative(results.get(keyFor(event))),
          prepareMs: relative(preparations.get(keyFor(event))),
          firstPaintMs: relative(at),
        });
      }
      break;
    }
  }
}
console.log(
  JSON.stringify(
    {
      events: Object.fromEntries(counts),
      stages: Object.fromEntries(costs),
      changes,
    },
    null,
    2,
  ),
);
