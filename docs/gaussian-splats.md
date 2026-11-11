# Gaussian 3D Tiles

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

The accepted appearance properties are `ellipse`, `perspective`,
`cameraDistance`, and either `srgb_rec709_display` or `lin_rec709_display`.
Accepting both color spaces for decoding does not establish a policy for blending
them together. Unsupported required extensions and appearance properties produce
named errors, including across the decode worker boundary.

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

`createGaussianSortController` coalesces camera changes behind one running job.
Changing scene membership or placement through `setScene` advances its revision;
obsolete results are discarded. Repeating a camera position does not sort again.
Rotation alone is not a sorting input. The last published presentation remains
the consumer's responsibility until the next one is ready.

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
worker. Scene attributes are cloned once per scene revision; subsequent camera
jobs send only the camera position. Position arrays are borrowed and must remain
immutable until all jobs referring to them complete.

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
ordering share the physical renderer. It supports sRGB reconstruction, complete SH degrees 0 through 3, affine
covariance transformation and explicit XYZW-to-mapper quaternion conversion.
The renderer rejects linear reconstruction color space until a corresponding
resolve/composition policy is available. Gaussian picking remains disabled.

Install `createGaussianAlphaPass` as the host ForwardPass's translucent pass.
Opaque geometry populates depth first; the shared Gaussian actor then uses
premultiplied alpha and depth testing without depth writes. General translucent
triangle interleaving is not supported by this pass.

Attributes occupy stable, reusable ranges in persistent capacity arrays and GPU
textures. Tile admission and placement changes update affected ranges; camera
movement changes only order for degree-zero content. Directional SH also updates
color ranges. Capacity growth and context restoration require full uploads.
Retired ranges become reusable only after a coherent presentation swap. Parent
and child representations do not draw together during replacement.

`maximumVisibleSplats` constrains complete representations. Members pass their
shared `MemoryPool` through `TileContentAdapterOptions.memoryPool`; its ledger
owns source buffers, storage capacity, overlapping presentations, worker copies
and sort workspace. Per-member grants still constrain demand. The factory's
`resourceBudgetBytes` configures its pool when used without an injected pool. The selector retains a coarser representation when a replacement
cannot fit. Diagnostics separate source/staging attributes, GPU attributes,
worker attributes, sort workspace and replacement storage. Texture row padding
and runtime allocations can add overhead beyond logical payload estimates.

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
transform. Application bridge and map view integration remain separate work.

The tiny test GLB is synthetic and unlocated. It verifies standard attribute and
worker handling, not independent geographic alignment or producer compatibility.
