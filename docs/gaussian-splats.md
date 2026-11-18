# Gaussian 3D Tiles

**Status: experimental.** Gaussian loading and rendering are not production-ready.

The tiles member accepts Gaussian glTF content alongside triangle meshes.
`createGaussianContentAdapterFactory` supplies one shared Gaussian presentation
per VTK renderer, using the existing tile hierarchy, queues and residency
governor. The [dataset explorer](../examples/vtk/) streams an
existing city reconstruction; its [data record](../examples/vtk/scene/GAUSSIAN_DATA.md)
documents the source, supplied placement, license record and hosting.

## Decoded content

`DecodedPrimitive.kind` distinguishes `mesh` and `gaussian`. Gaussian data retains
primitive-local Float32 centers, XYZW rotations, linear standard deviations,
linear opacity, and all supplied degree 0 through 3 spherical harmonics. RGB
coefficients remain floating point without color clamping. `modelToScene` is the
composed Float64 node, glTF axis, tile and scene transform; Gaussian consumers
must not add the mesh `content.origin` to it.

The uncompressed decoder accepts the `ellipse` kernel, `perspective` projection,
`cameraDistance` sorting and both Rec.709 display color spaces. The Gaussian
adapter advertises only `srgb_rec709_display`, and the mesh adapter advertises
an empty list. The streamed member passes this capability to the decoder, which
checks Gaussian primitives in the active scene before reserving decoded output
or loading glTF dependencies. Unsupported required extensions and appearance
properties produce named errors across the decode worker boundary.

The uncompressed accessor profile is non-indexed POINTS with non-sparse
attributes. Skinning and morph targets are rejected. The selected city's draft
`KHR_gaussian_splatting_compression_spz_2` profile is also supported through
`@adobe/spz`, including SPZ version 3 and the producer's optional
`EXT_splat_opacity` power kernel. The codec contains its WASM and requires no
external codec URL. These draft compressed inputs are distinct from ratified
uncompressed Gaussian attributes. The SPZ codec preserves the packed numeric
coordinate frame; the authored glTF node and tile matrices place the centers,
rotations and harmonics without another axis conversion.

The existing document-wide codec checks still require codec URLs
when Draco or Basis extensions are declared, including declarations associated
with an ignored Gaussian fallback material.

The decoded geometry estimate includes all owned Gaussian attribute buffers and
the placement matrix. Transfers deduplicate backing buffers. The decoded cache
is bounded separately from renderer residency. Neither estimate measures peak
parser allocations, the codec runtime heap or driver overhead.

## Distance ordering

`createGaussianDistanceSorter` produces a stable global back-to-front order over
all supplied ranges, including ranges from different tilesets. It uses Float64
camera-distance keys and subtracts camera translation before adding local
offsets. The order indexes the concatenation of the input ranges.

`createGaussianSortController` keeps one running job and coalesces upcoming
work to the current camera and scene. The shared renderer can publish a
completed order while camera or desired coverage keeps changing. Tile
generations, visibility, placement and live tuple ranges must still match.
The renderer accepts snapshots in revision order. Repeated positions on an
unchanged scene reuse the current order. Rotation alone leaves the distance
keys unchanged.

The shared renderer acquires a work slot before capturing the scene and camera
for a sort. Changes while admission is queued coalesce into that job. Its latest
workspace estimate must fit the shared work-byte allowance before worker dispatch;
the slot remains owned until the scheduled work settles, including on disposal.

Preparing off-frontier content preserves sorting for unchanged visible ranges.
Storage updates can reuse a retained order when range identities, generations,
positions, placement and camera match. A completed order arriving during a queued
storage refresh is retained for that refresh to publish. The shared CPU ledger
owns this retained order until its visible scene changes or the renderer is disposed.

Use `createGaussianSortWorkerBackend` with a module worker resolved from
`tiled-vtkjs/tiles3d/gaussian-sort-worker`. The backend owns and terminates that
worker. Scene attributes are cloned once per scene revision. Subsequent jobs
for that scene send the camera position. Borrowed position arrays must remain
immutable until every referring job completes. The renderer retries a failed
worker dispatch once. If that attempt fails, it reports the error and preserves
the last compatible presentation.

## Tile realization

`Tiles3dMemberServices.createContentAdapter` supplies view-owned realization below
the existing tile selector, queues and residency governor. A shared implementation
can coordinate multiple members through this factory. The contract carries
camera, placement, visibility, admission, replacement groups and disposal.
`pickableMeshTiles()` exposes only triangles to mesh picking.

The default `createMeshContentAdapter` rejects Gaussian content. Opt in by
passing the same `createGaussianContentAdapterFactory` to the members sharing a
view. When the member renderer is wrapped for per-layer mesh appearance, set
`TileContentAdapterOptions.realizationRenderer` to the canonical physical renderer.
Meshes continue to use their layer renderer while Gaussian attachment and global
ordering share the physical renderer. The adapter supports complete SH degrees
0 through 3, affine covariance transformation and explicit XYZW-to-mapper
quaternion conversion. Gaussian picking is unavailable.

Install `createGaussianAlphaPass` as the host ForwardPass's translucent pass.
Opaque geometry populates depth first. The shared Gaussian actor uses
premultiplied alpha and depth testing without depth writes. The pass preserves
straight-alpha blending for other actors. It does not depth-sort those actors
together with individual splats. The shared tile adapter rejects `BLEND`
triangle content when it would coexist with Gaussians.

Attributes occupy stable, reusable ranges. The view's submission scheduler
slices covariance reconstruction, order remapping, SH color evaluation and
dirty-range texture transfers. With a live `getOpenGLRenderWindow` resolver,
bind the context before draining work and call the factory's `noteContext` for
that renderer. The adapter stages geometry in one texture bank per atlas and
stages color and order for the next draw separately from the accepted draw.
Degree-zero camera updates skip geometry and reuse colors already written to
that draw bank. A new atlas or context requires writing every range in its
first presentation. The VTK mapper handles texture updates during paint when
the host omits the live window resolver.

A completed transfer publishes coverage, the order and mesh picking together
at paint. A compatible region can advance while another region retains its
predecessor. Overlapping parent and child coverage is never drawn together for
a REPLACE transition. Tuple slots remain unavailable for reuse while an
accepted draw or pending work still references them. Growth and required
replacement compaction hold the old allocation until publication. Cancelling
a transition releases its unpublished reservations and transfers.

`maximumVisibleSplats` bounds complete visible representations. Its count
refusals are distinct from byte limits. Members pass their shared
`MemoryPool` through `TileContentAdapterOptions.memoryPool`. The ledger
charges source buffers, storage capacity, overlapping presentations, worker
copies and sort workspace. Per-member grants constrain demand. The factory's
`resourceBudgetBytes` configures its pool when used without an injected pool.
Final residency and available transition headroom can require different
coverage choices. The logical byte estimates exclude texture allocator and
driver overhead.

The sort endpoint is an ES module with shared module dependencies. Hosts bundling
the library must stage its complete emitted module graph at the worker URL, or
provide `createSortBackend` using their bundler's worker support. The example uses
Vite's module worker entry so those dependencies are emitted together.

## Running the city example

Build and link the exact renderer in `vtkjs-fork.env`, then run:

```sh
npm run build
npm run example -- --host 127.0.0.1 --port 5194
```

Open `/` and choose **Add dataset → 3D Tiles → Jastrzębia Góra**. The
city is a normal dataset in the main explorer, with the shared camera, view
governor, visibility, screen-space error control, source switcher and diagnostics.
A `?tiles=<tileset URL>` link opens a tileset directly, including on the root page.
The city preset uses the example server's fixed same-origin proxy because its
upstream redirect does not support browser CORS. Other URLs need a CORS-enabled
endpoint. The preset derives its local ENU frame from the supplied ECEF root
transform.

The tiny test GLB is synthetic and unlocated. It verifies standard attribute and
worker handling, not independent geographic alignment or producer compatibility.
