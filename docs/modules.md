# Module and worker contracts

One repository and one npm package contain `src/core`, `src/points` and
`src/tiles3d`. Each format imports core. Core has no format or renderer imports,
including type-only imports; the formats do not import each other. Source and
packed-bundle checks enforce these boundaries.

Core owns `Vec3` and `Bounds`, camera/matrix math, neutral `StreamedMember`
contracts, the member factory registry, coordinator, view quality governor,
page memory pool, request admission and submission scheduler. The package
root and `/core` expose the neutral API. The formats retain different
selection algorithms:
points use additive parent-closed draw prefixes, while meshes use replacement
frontiers and sibling admission barriers. They share resource accounting,
quality allocation and scheduling rather than traversal policy.

`/points` exposes PCT1/COPC sources, the point controller, progressive ordering
and picking contracts. `/points/vtk` exposes `createPointCloudMember` and
`createRendererAdapter`. `/tiles3d` exposes profile errors/configuration,
`Tiles3dMemberServices`, decoder pools/capabilities and coordinate helpers;
`/tiles3d/vtk` exposes `createTiles3dMember` and `createMeshAdapter`.

These are internal modules with explicit public subpaths, not three workspace
packages. This keeps one coordinated API/version while reducing bundle imports.
Install dependencies remain combined: a points-only installation still installs
the glTF/Draco dependencies, though a points bundle excludes them. Independently
published workspaces would be justified by an actual install-size or reuse
requirement; any such split should initially synchronize versions and keep
examples/fixtures/tooling unpublished.

## Format services and lifetime

A coordinator context carries the renderer handle, shared memory pool,
submission scheduler, render/work callbacks and device pixel ratio. It carries
no decoder or texture capabilities. The host constructs format services and
binds them into its tiles factory:

```js
import {
  createMemoryPool,
  createStreamedSceneCoordinator,
} from "tiled-vtkjs/core";
import { DecodeWorkerPool } from "tiled-vtkjs/tiles3d";
import { createTiles3dMember } from "tiled-vtkjs/tiles3d/vtk";

const memory = createMemoryPool(); // Share across views using the same GPU.
const coordinator = createStreamedSceneCoordinator({ memory, scheduleRender });
const workers = new DecodeWorkerPool({ workerUrl });
const services = { workers, textureCapabilities };
const mesh = createTiles3dMember(
  coordinator.context(renderer),
  config,
  services,
);
coordinator.register(mesh);
// Register point members with the same coordinator and memory pool.
// Before each paint:
coordinator.prepareFrame(frameSerial);
// At teardown:
coordinator.dispose();
workers.dispose();
```

The host owns the decoder pool and can share it across tiles members. Member
disposal cancels that member's work; it never disposes the host's pool.
Coordinator disposal releases registrations and members. The page memory pool
can outlive an individual view. Active source demand, optional CPU cache
retention, GPU residency and presented coverage have independent lifetimes.
Views can share live decoded payloads for compatible immutable sources.
The decoder capabilities and placement inputs are part of that identity.

The submission scheduler shares a time and byte allowance across views. CPU
packing and dirty-range GPU transfers can run on the owning live view through
`prepareWork(pageFrame)`. Use `prepareFrame(frameSerial, pageFrame)` immediately
before rendering to commit staged coverage. Pass the same page frame to sibling
coordinators that share an allowance. The VTK adapters use live OpenGL render
windows to stage point ranges, mesh geometry, and Gaussian attribute and
order ranges. Mesh texture payloads are prepared in scheduler jobs and
uploaded by VTK during paint. Some allocations are indivisible, including
texture storage allocation. Hosts without a live window resolver use each
mapper's paint-time uploads.

## Memory policy

Share one `MemoryPool` between every participating coordinator that uses the
same page resource allowance. Point and mesh members report minimum useful
coverage and desired capacity. The pool grants feasible minima in registration
order, then distributes spare capacity up to each member's demand. A small
member therefore leaves capacity available to a larger one. Registration order
is the priority policy when minimum representations cannot all fit.

```js
const memory = createMemoryPool({
  totalBytes: 512 * 1024 ** 2,
  cpuBytes: 1024 * 1024 ** 2,
  workBytes: 1024 * 1024 ** 2,
  transitionBytes: 128 * 1024 ** 2,
  maxConcurrentOperations: 2,
});
memory.configure({ totalBytes: 768 * 1024 ** 2 });
const usage = memory.stats();
```

`totalBytes` bounds estimated managed renderer residency. `cpuBytes` separately
admits retained CPU buffers and reserved realization arrays; shared backing
buffers count once. `workBytes` admits conservative estimates of overlapping
fetch, input buffers, decoder workspace and output. One keyed admission service
chooses ready work using current demand priorities and fairness across owners.
The configured `maxConcurrentOperations` bounds physical network and decode work.
Sorting has one independent grant so camera movement can update a sortable
view while tile content is pending. These stages all use the same work-byte
budget. Stats report the configurable background limit as
`maxConcurrentOperations` and `maxConcurrentBackgroundOperations`. Use
`maxConcurrentTotalOperations` for capacity across all stages, including sorting,
and `activeOperations` for their combined active count.
While a view has sortable content, its own fetch backlog leaves a background
slot for decoding or a member without a fetch in flight. Persistent workers
have separate limits. Waiting for a memory charge or worker does not occupy
a running slot. Cancelled work retains its charges until completion,
acknowledgement or worker termination. Finished output transfers to retained
ownership before renderer jobs release their references.

The renderer default retains the existing device-memory heuristic and 512 MiB
fallback. CPU and work limits default to twice that renderer allowance;
transition headroom defaults to one quarter. `configure` changes only the
specified limits. Lowering a limit does not instantly free committed resources.
Per-format payload caches and actor reuse remain soft, reclaimable storage
under these aggregate admission limits.

Resource ownership is independent of active scheduling. Hiding a member removes
its scheduling grant while its caches, renderer reuse and unsettled operations
remain charged. Disposal releases caches and renderer objects immediately;
uncancellable operations keep their work charges until they settle.

Mesh replacement reserves the complete new group before realization, including
old/new overlap. Its final frontier must fit `totalBytes`, and its overlapping
renderer estimate must fit `totalBytes + transitionBytes`. A refusal rolls back
partial preparation rather than waiting while holding incomplete groups. If a
lower limit cannot admit a coarse replacement, the member reports an inability
to transition and preserves existing coverage. Hosts can increase the relevant
limit, hide/remove lower-priority content, or supply a cheaper representation.

Member `stats().resourceState` and `onResourceState` distinguish `blocked` policy
refusals from `error` failures; mesh members also report `stalled`. Blocked states
include the domain, requested estimate, configured limit and whether competing
ownership can make the request feasible. Limit changes retry blocked work;
intrinsically oversized work does not repeatedly decode under unchanged limits.
Live demand and accepted coverage maintain their own references to immutable
payloads. Evicting optional cache entries only affects later reuse. Errors retain
their names and are delivered to `onError`. For WebGL context changes, hosts
provide the current render window before draining staged work. Retained arrays
remain available for reconstructing graphics resources in a new context.

These counters are accounting estimates, not total browser memory or physical
VRAM measurements. Standard raster dimensions, accessor ranges, COPC records
and streamed bodies provide early checks. Opaque custom sources are reconciled
after they return; decoder runtimes, browser image internals, uncertain codec
workspace and delayed driver reclamation cannot be bounded exactly. Source
initialization performed before member creation remains host-owned. The work
concurrency limit provides a coarse bound where sizes are unknown. Malformed
range validation remains independent of resource policy.

## Workers and assets

`/points/copc-worker` exports `serveCopcTileSourceWorker`, an endpoint function.
A host creates a module-worker wrapper importing and invoking it, supplies that
worker factory to `createCopcWorkerTileSource`, and passes `lazPerfWasmUrl`.
The source owns and disposes the workers it creates. `laz-perf.wasm` is resolved
from the laz-perf dependency and staged by the host.

`/tiles3d/decode-worker` resolves to the classic executable IIFE. It is an asset,
not an importable ES module. The decoder pool can locate the adjacent worker in
the flat package output. Bundled applications should stage it explicitly and
pass `workerUrl` so asset bases and content security policy belong to the host.

The codec exports are:

- `/tiles3d/codecs/draco_wasm_wrapper.js`
- `/tiles3d/codecs/draco_decoder.wasm`
- `/tiles3d/codecs/basis_encoder.js`
- `/tiles3d/codecs/basis_encoder.wasm`

Resolve these exports in Node with `import.meta.resolve` and copy their bytes
to the host's static asset directory. A bundler can resolve them as URL assets
(for example Vite's `?url`). Inject the resulting same-origin URLs through the
member configuration's `wasm: DecodeWasmUrls`. The worker disables remote codec
defaults; Draco and Basis payloads need those explicit URLs. Uncompressed GLB
decoding needs no codec assets. Codec files and the worker must be served from
the same verified package version, without substituting a stale built worker.

## Exports, declarations and qualification

Every JavaScript API export has a matching declaration target. Declarations
retain the core/points/tiles3d structure, while runtime library chunks remain
flat to preserve the decoder pool's adjacent-worker URL. The package contains
source/declarations/maps, the workers/codecs, the VTK revision file and the
source-contract fixture. Examples, generated fixtures and replay tooling do
not become published workspace packages.

The former root format exports move to `/points` and `/tiles3d`; the former
`/vtk` splits between `/points/vtk` and `/tiles3d/vtk`. The old `/copc-worker`
moves to `/points/copc-worker`, and `/tiles3d-decode-worker` moves to
`/tiles3d/decode-worker`. No compatibility aliases are provided.

`npm run verify:boundaries` checks source imports, including type-only
references. `npm run verify:package` packs the built artifact, installs it into
a separate consumer, checks public declarations and bundle graphs, and renders
with the pinned VTK build in headless Chromium. The fixture exercises point
rendering/picking, controlled mesh replacement, shared budgets/scheduling,
actual GLB worker decoding, and disposal. It uses software WebGL for correctness;
it is not GPU performance evidence.

The mesh reader remains limited to the [documented supported 3D Tiles
profile](../README.md#supported-3d-tiles-profile). Scene-streaming naming does
not expand format support.
