# Gaussian tiles demo source data

The city preset loads an existing tiled version of Jastrzębia Góra, Poland.
The producer supplies the hierarchy, coarse representations, geometric errors
and ECEF placement. This demo does not produce or retile the reconstruction.

## Attribution and license records

- Original reconstruction: **Andrii Shramko, Teleportour**.
- [Creator's SuperSplat scene](https://superspl.at/scene/221ee167): “20K Photo
  106Mspats 4x2KM Andrii Shramko Poland Jastrzebia Gora.” The scene page lists
  **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** and 105.9 million
  splats. This license permits reuse with attribution and notice of changes.
- Tiled version: hosted by William Liu's
  [3D Tiles Gaussian Splat plugin demo](https://3d-tiles-rendererjs-3dgs-plugin.vercel.app/).
  Its [build summary](https://3d-tiles-rendererjs-3dgs-plugin.vercel.app/api/blob/JastrebiaGora/build_summary.json)
  identifies `3DGS-PLY-3DTiles-Converter` as the producer and records filtering,
  simplification and coverage compensation.
- The [plugin repository](https://github.com/WilliamLiu-1997/3D-Tiles-RendererJS-3DGS-Plugin)
  is Apache-2.0. That software license is not a separate license declaration
  for the city data. The tiled host supplies no separate attribution or data
  license record alongside this tileset; the original creator's CC BY 4.0
  scene is the available source license record.

The demo streams the externally hosted data and does not distribute a city
archive in this repository. Preserve the creator, source link, license link and
tiled-version credit when presenting or redistributing the reconstruction.

## Existing tileset

Canonical manifest:

<https://3d-tiles-rendererjs-3dgs-plugin.vercel.app/api/blob/JastrebiaGora/tileset.json>

The manifest and build summary inspected on 2026-11-10 describe:

| Property                                         | Value                                |
| ------------------------------------------------ | ------------------------------------ |
| Input reconstruction                             | 106,447,647 splats                   |
| Retained reconstruction after producer filtering | 103,518,942 splats                   |
| Tiles with content                               | 8,784                                |
| Explicit hierarchy levels                        | 13                                   |
| Refinement                                       | `REPLACE`                            |
| Bounds                                           | Oriented boxes                       |
| Geographic placement                             | Supplied root transform to ECEF      |
| Payload                                          | Gaussian glTF/GLB with SPZ version 3 |
| Harmonic degree                                  | 0                                    |
| Reconstruction color space                       | `srgb_rec709_display`                |
| Root tile                                        | 25,274 splats, 540,328 bytes         |

The retained count describes the reconstruction, not the sum of every stored
LOD representation or the number simultaneously displayed. The root transform
places the origin near 18.322779° E, 54.829515° N at 197.227 m ellipsoidal
height. This is the declared placement; it has not been independently surveyed
against control points.

The payload uses the draft
`KHR_gaussian_splatting_compression_spz_2` child extension, plus the producer's
optional `EXT_splat_opacity` version 1. The latter stores a nonlinear opacity
encoding used by coarse merged splats. The decoder exposes its power-kernel
semantics explicitly; it does not interpret values above one as ordinary
linear opacity. This input profile is distinct from uncompressed, ratified
`KHR_gaussian_splatting` attributes.

## Local serving

The canonical endpoint redirects to expiring Vercel Blob URLs. Its redirect
response does not provide the CORS headers required by a browser request from
localhost. The example's Vite development and preview servers expose
`/gaussian-city/` as a same-origin, read-only proxy for this one preset.
Relative content URLs continue to resolve beneath that route.

The proxy follows redirects server-side and streams response bodies. It accepts
only `GET` and `HEAD` for the manifest, build summary, and numeric GLB tile
paths, and does not accept arbitrary upstream URLs. It stores no whole-city
cache. An independently hosted static build needs equivalent routing or a
tileset endpoint whose redirect chain supports CORS.

## Inspected content hashes

These hashes identify the inspected responses, not an immutable hosting
guarantee. The public host can replace files at the same URLs.

| Response             |     Bytes | SHA-256                                                            |
| -------------------- | --------: | ------------------------------------------------------------------ |
| `tileset.json`       | 2,210,041 | `510fa249994ee65aa69b408fc92aa5b6e1aca1079c45f97c03d47495f9fbdd65` |
| `build_summary.json` |     4,627 | `0e626c3f5037a18ccb089be07b0cda0cf88d5334eb345f2e002697a3c8780834` |
| `tiles/0/0.glb`      |   540,328 | `af744865113c12bec15901a571177574b52ec473790040e00a68838f60769b05` |
