# Changelog

## [1.1.0](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.0.1...bc-platform-1.1.0) (2026-10-10)


### New and changed

* **patch:** 1330 keeps decode-sized batches on the dense path; evaluated, ub1024 not adopted at ctx 245760 ([#77](https://github.com/audiagentic/bigcherry/issues/77)) ([456d7eb](https://github.com/audiagentic/bigcherry/commit/456d7ebfd8f225d01abca667c7599ca207bc6728))
* **patch:** evict only the stale Meta split-state cache entry (1358); reject 1328 expert offload ([#93](https://github.com/audiagentic/bigcherry/issues/93)) ([aadc5a3](https://github.com/audiagentic/bigcherry/commit/aadc5a3338eb622dd56ed1033ea87930cc443f7a))
* **patch:** keep one prompt batch queued ahead of the cards with an MTP drafter, default off (1359) ([#135](https://github.com/audiagentic/bigcherry/issues/135)) ([94736f1](https://github.com/audiagentic/bigcherry/commit/94736f1b6b1ebdbb799c4b5f04eee5794c7cb46f))
* **patch:** lock counters for the dispatch workers (1356), standalone HIP probes and prefill trace tools ([#129](https://github.com/audiagentic/bigcherry/issues/129)) ([82bdc03](https://github.com/audiagentic/bigcherry/commit/82bdc0300f765b0ba2098675d8219f028fad1609))
* **patch:** promote 1355_hc_post_gate_fuse ([#112](https://github.com/audiagentic/bigcherry/issues/112)) ([0a1a75a](https://github.com/audiagentic/bigcherry/commit/0a1a75aafb17e2c464e2b590884360386eeeb6b4))
* **patch:** promote 1356_meta_dispatch_workers ([#116](https://github.com/audiagentic/bigcherry/issues/116)) ([b1330d8](https://github.com/audiagentic/bigcherry/commit/b1330d8305d3b6c58085d32731a67f3547203c69))
* **patch:** promote 1357_moe_router_splitk ([#122](https://github.com/audiagentic/bigcherry/issues/122)) ([c5e8d6b](https://github.com/audiagentic/bigcherry/commit/c5e8d6bed639276db0040678b12cc29a6d8853c0))
* **patch:** promote 1358_meta_split_cache_local_evict ([#107](https://github.com/audiagentic/bigcherry/issues/107)) ([197ca50](https://github.com/audiagentic/bigcherry/commit/197ca504530f3f94f4b39e1169f5501278afb6e9))
* **patch:** promote 1359_prefill_pipeline ([#138](https://github.com/audiagentic/bigcherry/issues/138)) ([a9ebc43](https://github.com/audiagentic/bigcherry/commit/a9ebc432072ee1504d6482845efbd794c45c1b21))
* **patch:** split-K router GEMM for the MoE router, default off (1357) ([#92](https://github.com/audiagentic/bigcherry/issues/92)) ([ad84e71](https://github.com/audiagentic/bigcherry/commit/ad84e71594f1541cd84df59698f3bdd8e4f800af))
* **patch:** the prefill pipeline defaults on; its flag is an off switch (1359) ([#140](https://github.com/audiagentic/bigcherry/issues/140)) ([a976e32](https://github.com/audiagentic/bigcherry/commit/a976e32f7acdafcf18c5ced3dc35f71fd823cf9a))
* **profile:** Flash-Next switches on the split-K MoE router (1357) ([#126](https://github.com/audiagentic/bigcherry/issues/126)) ([1042e84](https://github.com/audiagentic/bigcherry/commit/1042e84d1fe0e943851c970eb28ca489b04bafad))
* **qfp43:** add DFlash acceptance input trace ([#78](https://github.com/audiagentic/bigcherry/issues/78)) ([a200912](https://github.com/audiagentic/bigcherry/commit/a200912123b2e4f11f13cbfc3047ce392b84f33a))


### Fixes

* **lab:** Flash-Next timing run sends its request through the chat template ([#124](https://github.com/audiagentic/bigcherry/issues/124)) ([ddefe0d](https://github.com/audiagentic/bigcherry/commit/ddefe0dd2f6a3a64393f33a7db2df3413827ee19))
* **patch:** demote 1356_meta_dispatch_workers to evaluated (output race at 98K) ([#130](https://github.com/audiagentic/bigcherry/issues/130)) ([1ce33a5](https://github.com/audiagentic/bigcherry/commit/1ce33a5dbd43dee43d847129b524d876ac5c4b4a))

## [1.0.1](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.0.0...bc-platform-1.0.1) (2026-10-09)


### Fixes

* engine-bench stops a production llama.cpp build with SIGINT and reports an unclean stop ([#102](https://github.com/audiagentic/bigcherry/issues/102)) ([b65e52f](https://github.com/audiagentic/bigcherry/commit/b65e52faf1f34bb053e45674ccf25be3dc3bbe13))

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
