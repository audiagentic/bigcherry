# Changelog

## [11474.7.0](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11474.6.0...bc-llamacpp-11474.7.0) (2026-10-10)


### New and changed

* **patch:** keep one prompt batch queued ahead of the cards with an MTP drafter, default off (1359) ([#135](https://github.com/audiagentic/bigcherry/issues/135)) ([94736f1](https://github.com/audiagentic/bigcherry/commit/94736f1b6b1ebdbb799c4b5f04eee5794c7cb46f))
* **patch:** lock counters for the dispatch workers (1356), standalone HIP probes and prefill trace tools ([#129](https://github.com/audiagentic/bigcherry/issues/129)) ([82bdc03](https://github.com/audiagentic/bigcherry/commit/82bdc0300f765b0ba2098675d8219f028fad1609))
* **patch:** promote 1359_prefill_pipeline ([#138](https://github.com/audiagentic/bigcherry/issues/138)) ([a9ebc43](https://github.com/audiagentic/bigcherry/commit/a9ebc432072ee1504d6482845efbd794c45c1b21))
* **profile:** Flash-Next switches on the split-K MoE router (1357) ([#126](https://github.com/audiagentic/bigcherry/issues/126)) ([1042e84](https://github.com/audiagentic/bigcherry/commit/1042e84d1fe0e943851c970eb28ca489b04bafad))


### Fixes

* **patch:** demote 1356_meta_dispatch_workers to evaluated (output race at 98K) ([#130](https://github.com/audiagentic/bigcherry/issues/130)) ([1ce33a5](https://github.com/audiagentic/bigcherry/commit/1ce33a5dbd43dee43d847129b524d876ac5c4b4a))

## [11474.6.0](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11474.5.0...bc-llamacpp-11474.6.0) (2026-10-10)


### New and changed

* **patch:** promote 1357_moe_router_splitk ([#122](https://github.com/audiagentic/bigcherry/issues/122)) ([c5e8d6b](https://github.com/audiagentic/bigcherry/commit/c5e8d6bed639276db0040678b12cc29a6d8853c0))
* **patch:** split-K router GEMM for the MoE router, default off (1357) ([#92](https://github.com/audiagentic/bigcherry/issues/92)) ([ad84e71](https://github.com/audiagentic/bigcherry/commit/ad84e71594f1541cd84df59698f3bdd8e4f800af))
* **qfp43:** add DFlash acceptance input trace ([#78](https://github.com/audiagentic/bigcherry/issues/78)) ([a200912](https://github.com/audiagentic/bigcherry/commit/a200912123b2e4f11f13cbfc3047ce392b84f33a))

## [11474.5.0](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11474.4.0...bc-llamacpp-11474.5.0) (2026-10-09)


### New and changed

* **patch:** promote 1355_hc_post_gate_fuse ([#112](https://github.com/audiagentic/bigcherry/issues/112)) ([0a1a75a](https://github.com/audiagentic/bigcherry/commit/0a1a75aafb17e2c464e2b590884360386eeeb6b4))
* **patch:** promote 1356_meta_dispatch_workers ([#116](https://github.com/audiagentic/bigcherry/issues/116)) ([b1330d8](https://github.com/audiagentic/bigcherry/commit/b1330d8305d3b6c58085d32731a67f3547203c69))

## [11474.4.0](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11474.3.0...bc-llamacpp-11474.4.0) (2026-10-09)


### New and changed

* **patch:** evict only the stale Meta split-state cache entry (1358); reject 1328 expert offload ([#93](https://github.com/audiagentic/bigcherry/issues/93)) ([aadc5a3](https://github.com/audiagentic/bigcherry/commit/aadc5a3338eb622dd56ed1033ea87930cc443f7a))
* **patch:** promote 1358_meta_split_cache_local_evict ([#107](https://github.com/audiagentic/bigcherry/issues/107)) ([197ca50](https://github.com/audiagentic/bigcherry/commit/197ca504530f3f94f4b39e1169f5501278afb6e9))

## [11474.3.0](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11474.2.0...bc-llamacpp-11474.3.0) (2026-10-09)


### New and changed

* engine registry for per-engine patch, overlay and vendor locations ([#87](https://github.com/audiagentic/bigcherry/issues/87)) ([c9a8c58](https://github.com/audiagentic/bigcherry/commit/c9a8c58eab5e86b5e6122e5d62c17a799949b8e4))
* engine serve specification and engine-neutral server runner (MEN03) ([#90](https://github.com/audiagentic/bigcherry/issues/90)) ([b72b150](https://github.com/audiagentic/bigcherry/commit/b72b1504e06656e86c89ca3dc730a0cad136d230))
* **patch:** add QFP41 meta dispatch workers ([#56](https://github.com/audiagentic/bigcherry/issues/56)) ([df52fe6](https://github.com/audiagentic/bigcherry/commit/df52fe60850f9af9397c14af7edc906d7e39bc30))
* **patch:** fuse QFP35 HC post gate ([#54](https://github.com/audiagentic/bigcherry/issues/54)) ([4707efe](https://github.com/audiagentic/bigcherry/commit/4707efedefad20aeef6b11aab8421fe5da1f4bc0))
* **patch:** promote host-timing, MTP prompt timing and fusion-bisect diagnostics (1319, 1320, 1325, 1342, 1346) ([#8](https://github.com/audiagentic/bigcherry/issues/8)) ([1248ec4](https://github.com/audiagentic/bigcherry/commit/1248ec4b46f56f5236ea4001a5775c5048796772))
* per-engine release lines (bc-llamacpp, bc-platform) with readable engine-build tags (MEN09) ([#91](https://github.com/audiagentic/bigcherry/issues/91)) ([f9288eb](https://github.com/audiagentic/bigcherry/commit/f9288ebe26e60ff4a0efdb2b32c9e9bd91e6bcb5))

## [11474.2.0](https://github.com/audiagentic/bigcherry/compare/bc-11474.1.0...bc-11474.2.0) (2026-10-08)


### New and changed

* add slice worktree lifecycle commands ([#34](https://github.com/audiagentic/bigcherry/issues/34)) ([bbce6b6](https://github.com/audiagentic/bigcherry/commit/bbce6b6eb9dcf9559051e9c02074a0451a6279e5))
* automate qualified patch promotion ([#11](https://github.com/audiagentic/bigcherry/issues/11)) ([bd80a8d](https://github.com/audiagentic/bigcherry/commit/bd80a8d824697e963e58617d8ceb43b0ab3af183))
* execute slice lab runs in detached queued worktrees ([#35](https://github.com/audiagentic/bigcherry/issues/35)) ([a03fd83](https://github.com/audiagentic/bigcherry/commit/a03fd83d232f49ffc54268632e1aa54ec4f0aab0))
* **lab:** reference lane run of llama-server on the R9700 alone with a 4-bit 27B ([#75](https://github.com/audiagentic/bigcherry/issues/75)) ([ef52141](https://github.com/audiagentic/bigcherry/commit/ef5214128379aecf38a2182530ea8b50f15f7892))
* **patch:** add Q8_0 few-tile MMQ Stream-K qualification ([#22](https://github.com/audiagentic/bigcherry/issues/22)) ([4b1dfc4](https://github.com/audiagentic/bigcherry/commit/4b1dfc4854d168b3d250b7435f3d2bb9c272f52c))
* share primary checkout state across worktrees ([#33](https://github.com/audiagentic/bigcherry/issues/33)) ([cfc8d9b](https://github.com/audiagentic/bigcherry/commit/cfc8d9b1167c8aacd961d16261495d81ed440f2d))


### Fixes

* **ci:** scope the experiment audit to changed experiments ([#71](https://github.com/audiagentic/bigcherry/issues/71)) ([e76f349](https://github.com/audiagentic/bigcherry/commit/e76f349cb29521710d6ee473f18c4fc5bf3f4f9e))
* enforce LF line endings for tracked text ([#32](https://github.com/audiagentic/bigcherry/issues/32)) ([ff6bc5a](https://github.com/audiagentic/bigcherry/commit/ff6bc5ad3c5c4f6671882df5c6c0de6b0af4b140))
* **lab:** queue finds built binaries where the build publishes them ([#72](https://github.com/audiagentic/bigcherry/issues/72)) ([3dc67c6](https://github.com/audiagentic/bigcherry/commit/3dc67c64275cc4e621da43c8581881f609f0abec))
* make slice finish resumable and carry primary changes ([#58](https://github.com/audiagentic/bigcherry/issues/58)) ([ab8feab](https://github.com/audiagentic/bigcherry/commit/ab8feabade9a50f7046b93647bebaf61a8813c19))
* **patch:** 1328 aux expert backend works at b11474 (split-state cache, mirrored merge) ([#61](https://github.com/audiagentic/bigcherry/issues/61)) ([64c1ca7](https://github.com/audiagentic/bigcherry/commit/64c1ca71f3238171a3b76f9e1b011153efe9ef42))
* **patch:** 1340 arena plans are keyed by the output set, OUTPUT flags reach the per-device tensors ([#63](https://github.com/audiagentic/bigcherry/issues/63)) ([64c15de](https://github.com/audiagentic/bigcherry/commit/64c15dec117d680e8e5555c31b3c6bdf042b6923))
* **patch:** reconcile remaining experiment audit after BPB01 merge ([#59](https://github.com/audiagentic/bigcherry/issues/59)) ([9b8ec87](https://github.com/audiagentic/bigcherry/commit/9b8ec879fb4c4be030e82e2af0dd7a2fbce6c1a8))
* **patch:** reject adaptive MTP depth (1255, 1268) at b11474 ([#51](https://github.com/audiagentic/bigcherry/issues/51)) ([d5fb657](https://github.com/audiagentic/bigcherry/commit/d5fb657a5075581df0e65d42c967d39a8ffbd63a))
* slice commands read git output as UTF-8; --carry fails instead of carrying nothing ([#74](https://github.com/audiagentic/bigcherry/issues/74)) ([fca841a](https://github.com/audiagentic/bigcherry/commit/fca841a2b3af317a54eb1b236fa2d94f2da47208))
* **test:** mechanics tests read the pinned vendor source, not deleted lab fixtures ([#69](https://github.com/audiagentic/bigcherry/issues/69)) ([52c3eb5](https://github.com/audiagentic/bigcherry/commit/52c3eb5443b08239602b8f40471a12ee2c72c2d7))

## [11474.1.0](https://github.com/audiagentic/bigcherry/compare/bc-11474.0.0...bc-11474.1.0) (2026-10-08)


### New and changed

* **patch:** defer MTP prompt catch-up ([#13](https://github.com/audiagentic/bigcherry/issues/13)) ([b43235a](https://github.com/audiagentic/bigcherry/commit/b43235a62822dff785db221962806533d095bd7b))
* **patch:** promote MTP look-ahead (1321, 1322), deferred catch-up in the production set (1348), diagnostics 1316/1329/1331; reject 1210/1252/1337/1338, retire 1330 ([#27](https://github.com/audiagentic/bigcherry/issues/27)) ([e240483](https://github.com/audiagentic/bigcherry/commit/e240483439312202926d3064597d25cbd4cb374e))
* **patch:** promote MTP tracing diagnostics (1315, 1317, 1318) ([#23](https://github.com/audiagentic/bigcherry/issues/23)) ([6cf9f5c](https://github.com/audiagentic/bigcherry/commit/6cf9f5ce0d173b64ae74b7ebda1424fd6b3b3d44))


### Fixes

* **patch:** make 1316 node hash safe on Meta buffers ([#24](https://github.com/audiagentic/bigcherry/issues/24)) ([1e1c0da](https://github.com/audiagentic/bigcherry/commit/1e1c0da93ceab530c2d67c2763c65f7dc15e4d0f))

## [11474.0.0](https://github.com/audiagentic/bigcherry/compare/bc-11402.0.0...bc-11474.0.0) (2026-10-07)


### Maintenance

* release notes bc-11474.0.0 ([45ab24c](https://github.com/audiagentic/bigcherry/commit/45ab24c808fd79f7119d9c437009621dae88629d))

## [11402.0.0](https://github.com/audiagentic/bigcherry/compare/bc-11126.0.0...bc-11402.0.0) (2026-10-05)


### Performance

* **patching:** avoid duplicate overlay target reads ([cdac46e](https://github.com/audiagentic/bigcherry/commit/cdac46e9e300728fea4dc385bf743363879e6222))


### Maintenance

* release notes bc-b11402 ([1e0b1da](https://github.com/audiagentic/bigcherry/commit/1e0b1da3f996a89456fdf61c14c4f9b43e9f15ef))
