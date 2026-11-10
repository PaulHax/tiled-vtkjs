# Gaussian content foundations

The decoder accepts uncompressed `KHR_gaussian_splatting` ellipse primitives
alongside triangle meshes. This is a decoding and sorting API, not yet a complete
Gaussian renderer for streamed tiles.

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

The initial accessor profile is non-indexed POINTS with non-sparse attributes.
Skinning and morph targets are rejected. Gaussian compression extensions are not
implemented. The existing document-wide codec checks still require codec URLs
when Draco or Basis extensions are declared, including declarations associated
with an ignored Gaussian fallback material.

The decoded geometry estimate includes all owned Gaussian attribute buffers and
the placement matrix. Transfers deduplicate backing buffers. This estimate does
not include peak parser allocations, worker clones, sort workspace or future GPU
storage; it is not an end-to-end memory bound.

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

Use `createGaussianSortWorkerBackend` with a module worker resolved from
`tiled-vtkjs/tiles3d/gaussian-sort-worker`. The backend owns and terminates that
worker. Scene attributes are cloned once per scene revision; subsequent camera
jobs send only the camera position. Position arrays are borrowed and must remain
immutable until all jobs referring to them complete. Range generations identify
their lifetime but do not allocate or retire GPU storage.

## Tile realization

`Tiles3dMemberServices.createContentAdapter` supplies view-owned realization below
the existing tile selector, queues and residency governor. A shared implementation
can coordinate multiple members through this factory. The contract carries
camera, placement, visibility, admission, replacement groups and disposal.
`pickableMeshTiles()` exposes only triangles to mesh picking.

The default `createMeshContentAdapter` rejects a Gaussian tile or replacement
group with `TileContentRealizationError` before admitting partial mesh content.
It preserves an already displayed parent when its replacement cannot be
realized. Gaussian rendering still requires shared GPU range storage,
appearance evaluation, host depth and blending integration, resource budgets,
and coherent presentation swaps. The Python host must keep Gaussian admission
disabled until those parts are available.

The tiny test GLB is synthetic and unlocated. It verifies standard attribute and
worker handling, not independent geographic alignment or producer compatibility.
