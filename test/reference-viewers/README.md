# Reference viewer benchmark

The benchmark gives each viewer an explicit camera path and serves local assets from the same HTTP Range server. It saves the raw browser samples, per-case summaries, request and Range logs, and canvas screenshots. The generated assets are small correctness fixtures. Use local captures for performance comparisons.

| Workload               | Default cases                                                                       | Scope                                                                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Standard mesh 3D Tiles | `tiled-vtkjs-mesh-tiles`, `cesium-mesh-tiles`, `three-tiles-mesh-tiles`             | Same three-level GLB tileset                                                                                                              |
| SPZ Gaussian 3D Tiles  | `tiled-vtkjs-gaussian-tiles`, `cesium-gaussian-tiles`, `three-tiles-gaussian-tiles` | Same three-level tileset. Three.js uses NASA-AMMOS 3DTilesRendererJS with Gaussian Splat Lite and the Gaussian splat plugin.              |
| Standalone mesh        | `tiled-vtkjs-mesh`, `three-mesh`                                                    | Same GLB mesh. tiled-vtkjs loads it as one tile.                                                                                          |
| Point clouds           | `tiled-vtkjs-points`, `potree-points`                                               | COPC and Potree 2 fixtures. For a performance comparison, convert the same source capture to both formats.                                |
| Native splat LOD       | `spark-splats`, `playcanvas-splats`                                                 | Shared source splats. Spark generates LOD from PLY in the smoke case and accepts paged RAD in custom cases. PlayCanvas uses Streamed SOG. |
| Raster zoom            | `maplibre-raster`, `openlayers-cog`                                                 | XYZ raster tiles and an internally tiled COG with overviews. OpenLayers reads the COG by byte range.                                      |

The [Svirnas roof comparison](./svirnas/README.md) includes matched crops, repeated close-up scores and the recorded camera path used to measure time to detail.

## Run the fixtures

Run from the tiled-vtkjs worktree with Node 22.18 or later, the repository dependencies, and Chromium installed for Playwright. The fixture generator needs Python with NumPy and rasterio. Set `BENCHMARK_PYTHON` to the Python executable if those packages are in a virtual environment.

```sh
npm ci --prefix test/reference-viewers
npm run build
npm run example:build
npm --prefix test/reference-viewers run bench
```

Use `--case ID` repeatedly to select cases. The default is a headless smoke run. The opt-in hardware run also stays headless and verifies the viewer's actual WebGL renderer. Under WSL, the runner uses native Windows Chrome and connects over a local DevTools relay. Install Chrome in the standard Windows location and leave WSL localhost forwarding enabled. Use the renderer and browser recorded in each report when comparing runs. The runner verifies that each viewer starts at the requested drawing-buffer size. Timed recorded routes omit the initial screenshot so the browser begins the route as soon as the initial view is drawable. Use a separate run for pixel sampling, which preserves the drawing buffer and reads a 96 x 72 image on each animation callback during the trace.

```sh
npm --prefix test/reference-viewers run bench -- \
  --gpu --repeats 5 --latency 40 --mbps 80 --out artifacts/reference-hardware
npm --prefix test/reference-viewers run bench -- \
  --coverage --latency 40 --case tiled-vtkjs-gaussian-tiles --out artifacts/reference-coverage
```

The trace first approaches from the coarse view, then reverses zoom on the same page with the viewer's caches warm. It also oscillates near an LOD boundary and orbits at fixed distance. Thresholds vary by asset and renderer. Adjust a capture's camera keyframes to cross the intended boundary. Browser HTTP caching is disabled by the asset server so the warm test measures the viewer's own reuse. The server applies delay and transfer pacing per request.

Compare the screenshots and settled detail alongside frame intervals and request volume. Animation intervals are browser callback cadence. The host timer covers work performed synchronously by the benchmark adapter. Selected tile IDs are available for some viewers. An empty selection, a blank canvas, and a localized coverage hole each measure different behavior. The coverage pass flags globally blank frames and records foreground area, so use image inspection or a dataset-specific mask to assess smaller holes. Use `summary.json` for case summaries and the per-case JSON for samples. The settled-reference pass writes
`summary-settled-reference.json`. Frame timings from software smoke runs and coverage readback runs are diagnostic.

## Run a local capture

Keep the manifest adjacent to its asset root, with paths in each case relative to that root. For direct 3D Tiles comparisons, point all three viewers at the same tileset. For native formats, use the same source scan and keep the axis, origin, framing, and settled detail comparable. Use `viewer: "three-mesh"` for a Y-up GLB and `viewer: "spark", format: "rad"` for paged splats. Raster cases use a map camera with a longitude/latitude center. COG cases also need a starting resolution in the COG projection's map units per pixel.

```json
{
  "assetRoot": "datasets",
  "cases": [
    {
      "id": "three-local-splats",
      "viewer": "three-tiles",
      "workload": "local-scan-gaussian",
      "format": "gaussian-3d-tiles",
      "asset": "scan/tileset.json",
      "sse": 8,
      "camera": {
        "target": [0, 0, 0],
        "farDistance": 350,
        "nearDistance": 25,
        "direction": [0, -1, 0.55],
        "up": [0, 0, 1]
      }
    }
  ]
}
```

```sh
npm --prefix test/reference-viewers run bench -- \
  --manifest /path/to/manifest.json --gpu --repeats 5 --out artifacts/local-capture
```

Use `--vtk-dist /path/to/examples/vtk/dist` to compare locally built tiled-vtkjs revisions that support the benchmark camera and SSE controls with the same manifest. Choose a different output directory per revision. Cases can override `traces` with arrays of `[milliseconds, normalizedZoom, halfTurns]` keyframes starting at zero. Zoom values zero and one select the far and near camera endpoints. For raster, the third value pans horizontally using the configured `panDegrees`.

Add `--trace` to a tiled-vtkjs diagnostic run to include the selection, decode,
sort, CPU preparation, texture writes, commit and first rendered frame in
`detailTrace`. Events and resource timings share the browser performance clock.
The route-start marker aligns them with the authored camera phases. Follow tile
IDs from selection into sort-target revisions, and match the revision and camera
position for each result and publication. Preparation costs separate time spent
running each step from time waiting for admission. Texture times measure CPU
submission. Each geometry-upload event records the portion of a completed texture write within one tile allocation, including writes made before a preparation is canceled. Sum bytes across events for transferred geometry. Capture frame timing in a separate run with tracing disabled. To inspect the fallback covering a
particular explicit leaf tile, run
`node test/reference-viewers/summarize-detail.mjs <report.json> <tile-path>`.
Times in the coverage history are relative to route start. A missing timestamp
means that operation was unobserved for the reported revision or tile.
Stage totals include startup and can overlap between preparations.

The pressure duration sums frame
intervals with an observed pressure flag. Repeated response bytes use the
declared length for each repeated path and byte range. A response cancelled
after headers can contribute its full declared length.

For a settled Gaussian reference that reaches the splat-count ceiling, set
`referenceSplatLimit` in its manifest case and record the value alongside
`memoryMiB`. This override applies only to the settled pass. Reference capture
requires 500 ms of reported convergence, with a 30-second no-progress guard
and a three-minute bound per pose.

## Memory pressure and LOD churn

Set `memoryMiB` on a 3D Tiles case to sweep the engine's own limit. For
tiled-vtkjs it sets the explorer's renderer budget and associated shared
CPU/work budgets. Complete replacement groups can use the pool's existing
temporary transition headroom. For Cesium it specifies the sum of the tile
cache and optional `overflowMiB`. For 3DTilesRendererJS it sets the cache high
watermark, with `retentionFraction` controlling the low watermark. Verify
actual pressure per case and interpret values in the context of each viewer's
memory accounting. The reported bytes represent engine estimates or resource
ledgers. The vtk.js samples report the configured and governor-adjusted SSE
separately.

For 3D camera traces, `camera.panOffset` is a world-coordinate displacement.
An optional fourth keyframe value scales that displacement, which lets the
same replay leave one neighborhood, visit another, and return. A custom
`traces` object supplies the full sequence in insertion order. Preserve
an approach and dwell before benchmarking a warm reversal.

Reports include tile additions, removals and re-entries, peak reported
residency, pressure samples and repeat-response counts. `measureDemand`
also records tiled-vtkjs traversal target and request sets. Compare those
sets with the published tile set to distinguish streaming work from visible
churn. Response byte counts use declared content lengths. Requests canceled
after headers can appear in both response and failure counts.

## Recorded two-POI route

`recorded-route.mjs` adds the same 37.3-second camera route to every compatible
Z-up 3D case in a manifest. Set `camera.poiA` to the first point of interest
(defaults to `camera.target`). The second point is `target + panOffset`.
`farDistance`, `nearDistance` and `transferDistance` select the camera distances.
The approach wheel bursts and two orbit tracks come from the original
Luxembourg recording. The POI locations, transfer and overview return are
authored for the asset. This keeps the path identical across engines. For
original input event replay in the native tiled-vtkjs viewer, see the
[gesture replay instructions](../recordings/README.md).

```sh
node test/reference-viewers/recorded-route.mjs scene.json route.json
npm --prefix test/reference-viewers run bench -- \
  --manifest route.json --gpu --latency 40 --mbps 80 --repeats 5
```

A manifest with `assetSource: { "directory": "city-full", "url": "https://…/" }`
can populate missing GLBs during an explicit `--prepare-assets` discovery run.
Place the root tileset JSON in the asset root first. Discover with every
compared viewer and cap, then use the frozen local corpus for timed runs.
A cache miss fails the measured run. Each repeat gets a fresh browser context.
The two POIs share that context, so viewer cache reuse and re-fetches are part
of the workload. Record the source hierarchy and the bytes served on the route
separately.

The full city profile uses the published Jastrzębia Góra scan. The preparer
checks the source manifest hash, keeps all 8,784 payload references, and removes
the root georeferencing transform so all three cameras operate in local
coordinates. Use 256 MiB for the main pressure pass and 24 MiB for a severe
stress probe. [Source scan](https://superspl.at/scene/221ee167)
[Tiled distribution](https://github.com/WilliamLiu-1997/3D-Tiles-RendererJS-3DGS-Plugin)

```sh
curl -fL https://3d-tiles-rendererjs-3dgs-plugin.vercel.app/api/blob/JastrebiaGora/tileset.json -o city-source.json
node test/reference-viewers/prepare-city.mjs city-source.json artifacts/city
node test/reference-viewers/recorded-route.mjs artifacts/city/city-full.json artifacts/city/route.json
```

### Metrics for tuning

1. **Gate on visual quality at every checkpoint.** Compare exact-pose candidate
   and settled same-engine images at the same viewport and clear color.
   Foreground recall catches missing coverage. Foreground precision catches
   stale detail or spill into previously empty regions. Mean linear RGB error
   measures appearance changes. Gradient residual measures loss of fine detail
   within labeled POI crops. Check separate crops for the close POIs, the
   transfer and the coarse return. A failure at one checkpoint fails the run's
   quality gate.
2. **Rank recovery and frame pacing after passing quality.** The recovery target
   is milliseconds from the last camera motion to the first accepted view
   followed by at least 500 ms of passing samples. The quality pass takes
   samples roughly every 250 ms during the two close holds and overview return.
   A stalled or unobserved recovery fails the gate. Cold first entry still reports
   viewer readiness as a diagnostic.
   Compare p95 frame intervals and total time above 16.7 ms and 33.3 ms by
   route phase using the hardware timing pass. Keep cold arrival and warm
   overview return separate.
3. **Use memory and churn to explain the result.** Report peak engine-accounted
   residency against the configured cap, time under pressure, tile re-entry,
   repeat payload responses and bytes. Use view coverage and quality to judge
   whether retention helped. Browser and engine memory estimates have different
   scopes. Publish the budget, SSE, and pressure evidence with a comparison.

Capture candidate checkpoints with `--coverage --checkpoints`. For the
reference, use the same route and viewer with tighter detail settings and
enough residency to let the selected LOD settle, and add `--settled-reference`.
The two passes default to separate output directories under
`artifacts/reference-viewers/`. Settled-reference filenames include
`-settled-reference` so they can also share an explicit `--out` directory.
Quality scoring requires identical SHA-256 fingerprints of the local asset
roots, including the files behind any tileset references. Keep the asset
root limited to the frozen corpus and save the benchmark outputs elsewhere.
Validate the reference by repeating it with tighter settings and inspecting
the two close POIs and overview. Match the intended detail across viewers
before ranking them. Do the image runs separately from timing.

```sh
node test/reference-viewers/score-checkpoints.mjs candidate.json reference.json quality-policy.json
```

The policy supplies `background` and a `checkpoints` entry for every saved
label. Each entry has a top-left-origin normalized `crop` of `[x0,y0,x1,y1]`,
`minimumRecall`, `minimumPrecision`, `maximumColorError`, and
`maximumEdgeError`. Also set `maximumRecoveryMs` for `inspect-a`,
`inspect-b` and `overview-return`. Freeze the reference captures and thresholds using stable
repeat runs and reviewed degraded images before using them to choose an LOD
policy. Keep held-out crops or routes for confirmation.
