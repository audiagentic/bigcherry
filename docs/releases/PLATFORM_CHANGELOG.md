# Changelog

## 1.0.0 (2026-10-09)


### New and changed

* engine registry for per-engine patch, overlay and vendor locations ([#87](https://github.com/audiagentic/bigcherry/issues/87)) ([c9a8c58](https://github.com/audiagentic/bigcherry/commit/c9a8c58eab5e86b5e6122e5d62c17a799949b8e4))
* engine serve specification and engine-neutral server runner (MEN03) ([#90](https://github.com/audiagentic/bigcherry/issues/90)) ([b72b150](https://github.com/audiagentic/bigcherry/commit/b72b1504e06656e86c89ca3dc730a0cad136d230))
* engine-bench measures any declared engine with one result record (MEN03) ([#99](https://github.com/audiagentic/bigcherry/issues/99)) ([f5176c1](https://github.com/audiagentic/bigcherry/commit/f5176c1d9e1926605fa41f5680e27df2cefef019))
* **lab:** cross-engine quality check through the OpenAI API (MEN05) ([#89](https://github.com/audiagentic/bigcherry/issues/89)) ([4968569](https://github.com/audiagentic/bigcherry/commit/4968569537be30e0baa5faa36ef8092011915141))
* **lab:** radiance source build script, MEN01 source findings ([#82](https://github.com/audiagentic/bigcherry/issues/82)) ([7e85160](https://github.com/audiagentic/bigcherry/commit/7e851605808502c8678721170d32737bd68a9a67))
* **patch:** add QFP41 meta dispatch workers ([#56](https://github.com/audiagentic/bigcherry/issues/56)) ([df52fe6](https://github.com/audiagentic/bigcherry/commit/df52fe60850f9af9397c14af7edc906d7e39bc30))
* **patch:** fuse QFP35 HC post gate ([#54](https://github.com/audiagentic/bigcherry/issues/54)) ([4707efe](https://github.com/audiagentic/bigcherry/commit/4707efedefad20aeef6b11aab8421fe5da1f4bc0))
* **patch:** promote host-timing, MTP prompt timing and fusion-bisect diagnostics (1319, 1320, 1325, 1342, 1346) ([#8](https://github.com/audiagentic/bigcherry/issues/8)) ([1248ec4](https://github.com/audiagentic/bigcherry/commit/1248ec4b46f56f5236ea4001a5775c5048796772))
* per-engine release lines (bc-llamacpp, bc-platform) with readable engine-build tags (MEN09) ([#91](https://github.com/audiagentic/bigcherry/issues/91)) ([f9288eb](https://github.com/audiagentic/bigcherry/commit/f9288ebe26e60ff4a0efdb2b32c9e9bd91e6bcb5))
* promotion gates accept the profile-evidence tier; patch-promote writes its record and runs the gates in the slice ([#98](https://github.com/audiagentic/bigcherry/issues/98)) ([3b3e08e](https://github.com/audiagentic/bigcherry/commit/3b3e08e471f589c13c14588e543da126b26ff1fc))


### Fixes

* **lab:** cross-engine quality run gives each arm its own port ([#95](https://github.com/audiagentic/bigcherry/issues/95)) ([8a784c2](https://github.com/audiagentic/bigcherry/commit/8a784c239a25c110bef3e7c58eef29773949a39e))
* patch-promote keeps the comments in the production patch list ([#96](https://github.com/audiagentic/bigcherry/issues/96)) ([dc1949c](https://github.com/audiagentic/bigcherry/commit/dc1949cda38c75e4532d982d28d73c1a9c02b984))
* **patch:** 1293 superseded by 1326 (neutral at b11474); reference-lane results, ledger events ([#80](https://github.com/audiagentic/bigcherry/issues/80)) ([958da27](https://github.com/audiagentic/bigcherry/commit/958da27d67d768819a299b9442f64b42aa30d5d4))

## Changelog: BigCherry platform

Shared tooling (patch engine, lab queue, CI, slices, planning). Engine release lines keep their own changelog
under `engines/<engine>/`. Releases before the split are in `engines/llamacpp/CHANGELOG.md`.
