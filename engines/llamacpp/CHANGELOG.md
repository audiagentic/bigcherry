# Changelog

## [11552.3.0](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11552.2.1...bc-llamacpp-11552.3.0) (2026-10-21)


### New and changed

* **build:** engine-aware build context, GPU-target option and default binary (RAD01 PR 4) ([#220](https://github.com/audiagentic/bigcherry/issues/220)) ([7a88895](https://github.com/audiagentic/bigcherry/commit/7a88895c504fac8d5bebfa1f65d2b2761b4fa46e))
* **campaign:** RAM-only CPU smoke test per engine, first phase of pin-bump (MEN15) ([#258](https://github.com/audiagentic/bigcherry/issues/258)) ([f8b514b](https://github.com/audiagentic/bigcherry/commit/f8b514b0fd1311748faf18e670f4d905bfec2c77))
* **campaign:** server A/B names its engine; first Radiance A/B through ab-benchmark (RAD01 PR 8c) ([#226](https://github.com/audiagentic/bigcherry/issues/226)) ([2052122](https://github.com/audiagentic/bigcherry/commit/2052122aed40f2009c971103ec192a65e84810a9))
* **config:** each engine declares its recipes; Radiance recipes (RAD01 PR 3) ([#219](https://github.com/audiagentic/bigcherry/issues/219)) ([e8ee8e5](https://github.com/audiagentic/bigcherry/commit/e8ee8e5f2c8e5fe739f87b792b1bd2dee9f2fb6f))
* **libr3:** expert width and every remaining expert-count site are runtime (RCS09-05b, 07) ([#328](https://github.com/audiagentic/bigcherry/issues/328)) ([f792f51](https://github.com/audiagentic/bigcherry/commit/f792f51e183abc63242effa3e07ac870780ee301))
* **patch:** engine-declared patch ID prefix; radiance patches become rad0001-rad0003 ([#232](https://github.com/audiagentic/bigcherry/issues/232)) ([3527b6a](https://github.com/audiagentic/bigcherry/commit/3527b6afae7e9e60e6f0168a81e6d2eac50590c4))
* **radiance:** f16 WMMA MoE GEMM in the helpers (R3_MOE_FP16, default on) + replay sweep harness (RAD06) ([#276](https://github.com/audiagentic/bigcherry/issues/276)) ([5fea8be](https://github.com/audiagentic/bigcherry/commit/5fea8be8b0a5e8ebef13a3fa0a06083ad8bc0440))
* **radiance:** helper request capture/replay + GEMV_LARGE_ROWS (default off) (RAD06) ([#274](https://github.com/audiagentic/bigcherry/issues/274)) ([cec1316](https://github.com/audiagentic/bigcherry/commit/cec1316175b920fe8b4dc37cc136dc7cab17232e))
* **radiance:** long-run tile split (R3_MOE_HELPER_MOE_LONG_ROWS, default off, evaluated) + counter/lab scripts (RAD06) ([#275](https://github.com/audiagentic/bigcherry/issues/275)) ([abaa59f](https://github.com/audiagentic/bigcherry/commit/abaa59fe9d2fea4d875058830ed1c6567f044876))
* **radiance:** port to radiance 1.7.0 - thin-hook rad0002, diagnostics in rad0004, plugins on ABI 17, natives ported (MEN12) ([#261](https://github.com/audiagentic/bigcherry/issues/261)) ([2d374d1](https://github.com/audiagentic/bigcherry/commit/2d374d183770cee668a240e2138a3b3cda886571))
* **radiance:** R9700 engine with two XTX expert helpers - GPU mailbox protocol, placement, decode GEMV (RAD02-RAD06, RAD09) ([#236](https://github.com/audiagentic/bigcherry/issues/236)) ([63cc952](https://github.com/audiagentic/bigcherry/commit/63cc952f5df67093a70e038c67a9d6185aac20bb))
* **release:** radiance release line with upstream-as-major versions (MEN13) ([#252](https://github.com/audiagentic/bigcherry/issues/252)) ([1bd9b91](https://github.com/audiagentic/bigcherry/commit/1bd9b914f386ecff3100480b6ca067b689d1a87b))


### Fixes

* **patch:** 1340 meta-arena-graph-owned anchor ends at the accessor, not end of file ([#217](https://github.com/audiagentic/bigcherry/issues/217)) ([8014255](https://github.com/audiagentic/bigcherry/commit/8014255302746a61e7bfd0d0896880c3b76c5e32))

## [11552.2.1](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11552.2.0...bc-llamacpp-11552.2.1) (2026-10-12)


### Fixes

* **patch:** 1361 puts its second-arena helpers in 1340's own header (the anchor moved there with the 1340 refactor) ([bf01c91](https://github.com/audiagentic/bigcherry/commit/bf01c9116adaeabef8d297787bb33224d8592d38))

## [11552.2.0](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11552.1.2...bc-llamacpp-11552.2.0) (2026-10-12)


### New and changed

* **patch:** keep the shelved 1361 and 1356 code on main (evaluated, default off) ([71d6f49](https://github.com/audiagentic/bigcherry/commit/71d6f498ef45c0423ca24664d677b028f6643784))
* **patch:** keep the shelved code on main - 1361 as an evaluated patch, 1356 with its lock attempt, harness draft-model mode ([8db3f9a](https://github.com/audiagentic/bigcherry/commit/8db3f9ab325de4637e0aafe52b09bcb354c89eca))


### Fixes

* **patch:** correct owned-file guard regexes ([ebf5c71](https://github.com/audiagentic/bigcherry/commit/ebf5c71b1f5288cc14ec216beacf985ac64a2043))

## [11552.1.2](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11552.1.1...bc-llamacpp-11552.1.2) (2026-10-11)


### Fixes

* **patch:** shelve 1356 and 1360 as evaluated (owner decision); QFP41 and QFP49 closed; main ruleset prepared for the owner (PA46) ([ff10025](https://github.com/audiagentic/bigcherry/commit/ff10025d48e79e516c85e97c75a0bacc374e1fb9))
* **patch:** shelve 1356 and 1360 as evaluated; close QFP41 and QFP49 ([5d9c163](https://github.com/audiagentic/bigcherry/commit/5d9c1631c6989e6a85202a9fe17e6c64d2b8e3fa))

## [11552.1.1](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11552.1.0...bc-llamacpp-11552.1.1) (2026-10-11)


### Fixes

* **patch:** 1362 pads the pooled-block count too, so Qwen4Exp prompt batches reuse the graph ([30d3f1b](https://github.com/audiagentic/bigcherry/commit/30d3f1b2d6ef2b0ad5ed7e85109ed3125d07ac21))
* **patch:** reject 1362_kv_pad_prompt - graph reuse brings no prefill gain ([9109acb](https://github.com/audiagentic/bigcherry/commit/9109acb7463f250db8dea45c7122423fdea04560))
* **patch:** reject 1362_kv_pad_prompt - the graph is reused but prefill does not get faster (QFP53) ([351b18f](https://github.com/audiagentic/bigcherry/commit/351b18f107f729da98b00215a508cb2a1f9da121))
* **patch:** stabilize Qwen4Exp k-pool padding in 1362 ([0c18af2](https://github.com/audiagentic/bigcherry/commit/0c18af263cb1cc3a55b842d8f6e501283b656bdd))

## [11552.1.0](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11552.0.0...bc-llamacpp-11552.1.0) (2026-10-11)


### New and changed

* **patch:** add opt-in prompt KV padding (1362) ([4d690b5](https://github.com/audiagentic/bigcherry/commit/4d690b5013183de60b3d93eac899388e39ad1a85))
* **patch:** opt-in KV padding so prompt batches reuse the graph (1362) ([215424f](https://github.com/audiagentic/bigcherry/commit/215424f62e0dd19dc1e3487bd114e09dcb097305))


### Fixes

* **patch:** 1281 - the MoE kernel edit's guard keeps its indent (without it the guard matches upstream's own line and the edit is skipped); test checks the barrier after the clear ([9e6be0a](https://github.com/audiagentic/bigcherry/commit/9e6be0a7168bc35dc14004fd264c0dd205371b49))

## [11552.0.0](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11474.9.0...bc-llamacpp-11552.0.0) (2026-10-11)


### Fixes

* **patch:** b11552 - 0300's autotune edit takes the first of four can_execute openings ([34322c1](https://github.com/audiagentic/bigcherry/commit/34322c154d53f1cad4c88f4cf6b64fa8638de060))
* **patch:** b11552 - 0300's MoE native call sits inside the should_use_mmq branch ([fd926c0](https://github.com/audiagentic/bigcherry/commit/fd926c0a310be81e9ab0773a1a9613914dfde1ef))
* **patch:** b11552 - 1294 follows top-k's chunked row loop; 1333 anchors on the has_mixed decision alone ([9477827](https://github.com/audiagentic/bigcherry/commit/94778276bd17485a481f04f1ed4328f73309b0d1))
* **patch:** b11552 - 1321 and 1348 use set_embd_state for hidden-state rows; 1307's unary edit follows the contiguous branch; legacy-part tool ([142a0b8](https://github.com/audiagentic/bigcherry/commit/142a0b80866bb41c770937be43ecb18cf998ad11))
* **patch:** b11552 - 1337 state superseded, MoE cache experiments removed, 1338 test skipped until its rebuild (MET07) ([9fa5026](https://github.com/audiagentic/bigcherry/commit/9fa502648e4f00ea334d4f783750417edd8f275d))
* **patch:** b11552 - 1346's process timing edits take the second match (1348's snapshot repeats the code) ([cfb7b63](https://github.com/audiagentic/bigcherry/commit/cfb7b633f6df52ca8ba8dc1854994e2d304b9957))

## [11474.9.0](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11474.8.0...bc-llamacpp-11474.9.0) (2026-10-11)


### New and changed

* **patch:** cross-card sum on a transfer stream, default off (1360) ([#145](https://github.com/audiagentic/bigcherry/issues/145)) ([1542986](https://github.com/audiagentic/bigcherry/commit/154298622e9e75e024425be6badcbdd3472350d4))


### Fixes

* **patch:** 1360 state keeps its own copy of the reduced tensors ([#150](https://github.com/audiagentic/bigcherry/issues/150)) ([f1fd14c](https://github.com/audiagentic/bigcherry/commit/f1fd14cba71124f263098ade898f7699f3bfb7a9))

## [11474.8.0](https://github.com/audiagentic/bigcherry/compare/bc-llamacpp-11474.7.0...bc-llamacpp-11474.8.0) (2026-10-10)


### New and changed

* **patch:** 1330 keeps decode-sized batches on the dense path; evaluated, ub1024 not adopted at ctx 245760 ([#77](https://github.com/audiagentic/bigcherry/issues/77)) ([456d7eb](https://github.com/audiagentic/bigcherry/commit/456d7ebfd8f225d01abca667c7599ca207bc6728))
* **patch:** the prefill pipeline defaults on; its flag is an off switch (1359) ([#140](https://github.com/audiagentic/bigcherry/issues/140)) ([a976e32](https://github.com/audiagentic/bigcherry/commit/a976e32f7acdafcf18c5ced3dc35f71fd823cf9a))

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
