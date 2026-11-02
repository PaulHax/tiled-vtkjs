# Module and worker contracts

One repository and one npm package contain `src/core`, `src/points` and
`src/tiles3d`. Each format imports core. Core has no format or renderer imports,
including type-only imports; the formats do not import each other. Source and
packed-bundle checks enforce these boundaries.

Core owns `Vec3` and `Bounds`, camera/matrix math, neutral `StreamedMember`
contracts, the member factory registry, coordinator, view quality governor,
page memory pool and submission scheduler. The package root and `/core` expose
this neutral surface. The formats retain different selection algorithms:
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
can outlive an individual view. Decoded CPU cache, GPU residency and active draw
remain separate lifecycles. Point draw-plan changes preserve the existing
prefix upload behavior; extraction does not add staged GPU upload admission.

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
