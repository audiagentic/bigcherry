# Changelog

## [1170.8.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1170.7.0...bc-radiance-1170.8.0) (2026-11-07)


### New and changed

* **patch:** promote rad0013_mtp_pass_replay ([#702](https://github.com/audiagentic/bigcherry/issues/702)) ([57b87c0](https://github.com/audiagentic/bigcherry/commit/57b87c093d677b96e8faa78c057320523421e1dd))

## [1170.7.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1170.6.0...bc-radiance-1170.7.0) (2026-11-07)


### New and changed

* **serving:** manifest fields for 80 more engine options (contract amendment 13) ([#700](https://github.com/audiagentic/bigcherry/issues/700)) ([e097088](https://github.com/audiagentic/bigcherry/commit/e0970887e18d76830c92032f3916c11398b5e0f4))


### Fixes

* **radiance:** rad0013 restores MTP draft/verify pass replay lost in v1.17 (~2% decode step), with a start guard ([#699](https://github.com/audiagentic/bigcherry/issues/699)) ([0976fa8](https://github.com/audiagentic/bigcherry/commit/0976fa82d12bf9d8efd4cc1b91713b19036f4a31))

## [1170.6.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1170.5.1...bc-radiance-1170.6.0) (2026-11-07)


### New and changed

* **radiance:** campaign arms start from a serving manifest ([#692](https://github.com/audiagentic/bigcherry/issues/692)) ([7eb8664](https://github.com/audiagentic/bigcherry/commit/7eb8664105ec62f940424b94dc97b40379aa968b))
* **radiance:** RHP16 step 5, the live placement policy derives every setting from calibration (off until a layout is calibrated) ([#693](https://github.com/audiagentic/bigcherry/issues/693)) ([52c48e1](https://github.com/audiagentic/bigcherry/commit/52c48e1da291c2c47478146b054157cebbaefbe6))


### Fixes

* **moe:** replay sweep sizes its memory from the captured work ([#696](https://github.com/audiagentic/bigcherry/issues/696)) ([16f0110](https://github.com/audiagentic/bigcherry/commit/16f0110c543838d03c828b23374a0e87a75019e7))
* **radiance:** bc-serve writes engine-only manifests for dense models ([#691](https://github.com/audiagentic/bigcherry/issues/691)) ([11e5078](https://github.com/audiagentic/bigcherry/commit/11e50786e6a8defb77c896b15ab9fe562959f4d5))
* **radiance:** component tests run their assertions ([#688](https://github.com/audiagentic/bigcherry/issues/688)) ([8faeb6a](https://github.com/audiagentic/bigcherry/commit/8faeb6ab95151d30cac1ff2bb42428800a2e10f5))
* **rhp16:** arm calibration deadlines only on measurable evidence ([#695](https://github.com/audiagentic/bigcherry/issues/695)) ([8f37034](https://github.com/audiagentic/bigcherry/commit/8f37034d634300ecf528db4b8c74937986f799a1))

## [1170.5.1](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1170.5.0...bc-radiance-1170.5.1) (2026-11-06)


### Fixes

* **radiance:** calibration-ingest handles the optional handoff of amendment 12 ([#686](https://github.com/audiagentic/bigcherry/issues/686)) ([6833529](https://github.com/audiagentic/bigcherry/commit/68335292abc65b1366fab0391113c42742390c9c))

## [1170.5.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1170.4.0...bc-radiance-1170.5.0) (2026-11-06)


### New and changed

* **radiance:** RHP16 step 4, helper calibration records and the lab --calibrate run ([#682](https://github.com/audiagentic/bigcherry/issues/682)) ([a43c2cd](https://github.com/audiagentic/bigcherry/commit/a43c2cda93312bc3a079342b5938ddae442af642))
* **serving:** engine-only layouts; resolver reads layouts.toml; contract amendment 12 (RSC11 A2) ([#683](https://github.com/audiagentic/bigcherry/issues/683)) ([f115420](https://github.com/audiagentic/bigcherry/commit/f115420b7669bbe4e861dbd629e3e43f0ea0d837))

## [1170.4.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1170.3.0...bc-radiance-1170.4.0) (2026-11-06)


### New and changed

* **serving:** --set keys for tp_wire, no_prefix_cache and gpu headroom; manifest operator guide (RSC11 A1) ([#677](https://github.com/audiagentic/bigcherry/issues/677)) ([666a9fc](https://github.com/audiagentic/bigcherry/commit/666a9fc2d11861f470c71ed8babef4ad56eb143b))

## [1170.3.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1170.2.0...bc-radiance-1170.3.0) (2026-11-05)


### New and changed

* **radiance:** RHP16 step 3, bc-serve derives the live placement policy from calibration records ([#666](https://github.com/audiagentic/bigcherry/issues/666)) ([7cf2592](https://github.com/audiagentic/bigcherry/commit/7cf259297b385e4fa9cdff9504799c1254fb46e5))

## [1170.2.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1170.1.0...bc-radiance-1170.2.0) (2026-11-05)


### New and changed

* **build:** stable compile paths so ccache hits across build ids ([#661](https://github.com/audiagentic/bigcherry/issues/661)) ([54ecaa6](https://github.com/audiagentic/bigcherry/commit/54ecaa6b4c12e9a2647950c874d9c58f4b81bff5))

## [1170.1.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1170.0.0...bc-radiance-1170.1.0) (2026-11-05)


### New and changed

* **patch:** promote rad0009_serving_manifest, rad0011_plan_memory ([#663](https://github.com/audiagentic/bigcherry/issues/663)) ([d610837](https://github.com/audiagentic/bigcherry/commit/d610837b304f1193d2f36c05051def91b5e2478d))

## [1170.0.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1160.5.0...bc-radiance-1170.0.0) (2026-11-05)


### New and changed

* **serving:** bc-serve refuses to run against a ROCm SDK other than the one its build recorded (BRVP06 item 5) ([#652](https://github.com/audiagentic/bigcherry/issues/652)) ([8b8e2fe](https://github.com/audiagentic/bigcherry/commit/8b8e2fee789ae240e9de44abf6e6a7fb19995486))
* **serving:** the engine starts from its measured manifest and admits its plan; contract amendment 11 (RSC08 S3) ([#651](https://github.com/audiagentic/bigcherry/issues/651)) ([d64b655](https://github.com/audiagentic/bigcherry/commit/d64b655dc9913fcfa5b6e64b259bc3812d1ef12e))

## [1160.5.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1160.4.0...bc-radiance-1160.5.0) (2026-11-05)


### New and changed

* **patch:** a profile-evidence record carries forward across pin bumps when only the pin differs; rad0008 gets its promotion record from the evidence on hand ([#642](https://github.com/audiagentic/bigcherry/issues/642)) ([f2c2e2c](https://github.com/audiagentic/bigcherry/commit/f2c2e2c53c5339298667c92ebc775ba099a35338))
* **radiance:** helpers read their whole configuration from the serving manifest (RSC06, contract amendment 10) ([#633](https://github.com/audiagentic/bigcherry/issues/633)) ([9d5d0b8](https://github.com/audiagentic/bigcherry/commit/9d5d0b8081a1a60c2c2337d65c5b7d0d871d6788))

## [1160.4.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1160.3.0...bc-radiance-1160.4.0) (2026-11-05)


### New and changed

* **radiance:** rad0009 reads the v1 manifest through the shared reader; the interim manifest format and bc-serve's own plan measurement are deleted (RSC08 S2b) ([#588](https://github.com/audiagentic/bigcherry/issues/588)) ([ef0579f](https://github.com/audiagentic/bigcherry/commit/ef0579fd48429c6ae37200056d5cec7c3c9ab9cf))

## [1160.3.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1160.2.0...bc-radiance-1160.3.0) (2026-11-05)


### New and changed

* **radiance:** RHP16 - offline replay of the live placement policy over a recorded heat series ([#620](https://github.com/audiagentic/bigcherry/issues/620)) ([6f45c66](https://github.com/audiagentic/bigcherry/commit/6f45c668ccf738bcc68049da703b9cfa29ce207a))

## [1160.2.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1160.1.0...bc-radiance-1160.2.0) (2026-11-05)


### New and changed

* **build:** every build names a ROCm stack; SDK recorded, hashed, bound to CI image and launcher (BRVP06) ([#611](https://github.com/audiagentic/bigcherry/issues/611)) ([131c2b1](https://github.com/audiagentic/bigcherry/commit/131c2b1fe77e3cc15560e883cb6faa2ef9f8f875))
* **radiance:** live expert placement policy (RHP slices H, I, J): target map = the static placer's plan for the live heat ([#597](https://github.com/audiagentic/bigcherry/issues/597)) ([4d3c217](https://github.com/audiagentic/bigcherry/commit/4d3c2178ea78faf6070ed60b9abc20959d23d072))
* **radiance:** RHP16 step 2 - the live policy measures transaction time and bandwidth, update gaps, hot-set drift and sampling error (no setting derived yet) ([#619](https://github.com/audiagentic/bigcherry/issues/619)) ([8acb6d8](https://github.com/audiagentic/bigcherry/commit/8acb6d884848b4c09e7fb6774b2f58a34d0fcf40))


### Fixes

* **radiance:** RHP16 - measure hot-set drift on per-interval counts, not on the heat the half-life decays; record the first measured values ([#621](https://github.com/audiagentic/bigcherry/issues/621)) ([18579f7](https://github.com/audiagentic/bigcherry/commit/18579f7396fc1cf3bbbe1b873c14f0499c750971))


### Performance

* **radiance:** RAD48 - native gfx11 int8 prefill tile (27B INT8 prefill GEMM 3.1x) ([#601](https://github.com/audiagentic/bigcherry/issues/601)) ([424a6e4](https://github.com/audiagentic/bigcherry/commit/424a6e47eb8462f1b6742d4f2964b817f84842e7))

## [1160.1.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1160.0.0...bc-radiance-1160.1.0) (2026-11-04)


### New and changed

* **build:** BU01 Radiance CI cache, release bundle and kernel recompiles ([#584](https://github.com/audiagentic/bigcherry/issues/584)) ([e92fd1a](https://github.com/audiagentic/bigcherry/commit/e92fd1a20b20af491e2252455a516190d1db3891))
* **radiance:** qwen27b-2xtx layout - Qwen3.8-27B MXFP4 tensor-parallel over two RDNA3 cards ([#573](https://github.com/audiagentic/bigcherry/issues/573)) ([509d991](https://github.com/audiagentic/bigcherry/commit/509d991b83c0f42ad0d6295beb22a9feefe5da99))
* **radiance:** RHP slice H part 1 - heat source interface and policy-input builder (card-free, not wired) ([#591](https://github.com/audiagentic/bigcherry/issues/591)) ([40829e0](https://github.com/audiagentic/bigcherry/commit/40829e0dfff5ee5bbf8e03c52e528956f5fb9aee))
* **radiance:** RHP01 part 1 - rad0010 routing-evidence diagnostic patch ([#582](https://github.com/audiagentic/bigcherry/issues/582)) ([f48cc55](https://github.com/audiagentic/bigcherry/commit/f48cc5560e612faa81163627019223a557e19169))
* **radiance:** RHP11 slice D - live owner-map generations on the engine GPU, full epoch cycle ([#557](https://github.com/audiagentic/bigcherry/issues/557)) ([78704e2](https://github.com/audiagentic/bigcherry/commit/78704e2fa32676b2ac38821cc7e031b804ea396c))
* **radiance:** RHP11 slice E - live helper slot migration ([#558](https://github.com/audiagentic/bigcherry/issues/558)) ([c78a819](https://github.com/audiagentic/bigcherry/commit/c78a8191d57004b6cd67d5645a74f0f2f5b158c4))
* **radiance:** RHP11 slices F and G - combine epoch checks, v23 segment requirement, forced-move harness ([#572](https://github.com/audiagentic/bigcherry/issues/572)) ([aa38683](https://github.com/audiagentic/bigcherry/commit/aa38683d3dc457310dd88e4d15d91bfa8eddc94f))
* **serving:** bc-serve --write-manifest writes a v1 serving manifest from the resolver (RSC08 S2a) ([#578](https://github.com/audiagentic/bigcherry/issues/578)) ([a8e9554](https://github.com/audiagentic/bigcherry/commit/a8e9554d8f46948ba3a1574925cb060e66e25fcc))


### Fixes

* **radiance:** qwen27b-2xtx runs with 3 speculative draft tokens (single-stream decode 48 -&gt; 96.7 tok/s) ([#590](https://github.com/audiagentic/bigcherry/issues/590)) ([5c88f5c](https://github.com/audiagentic/bigcherry/commit/5c88f5c38ca8cdf27836c81c25ba17924b7264b6))
* **radiance:** r3_moe_worker replay runs the production bootstrap again ([#565](https://github.com/audiagentic/bigcherry/issues/565)) ([6eb31a7](https://github.com/audiagentic/bigcherry/commit/6eb31a708b92cefae2f90afed727e0638a449a50))
* **radiance:** RAD46 accurate gfx1030 rows kernels, selected per stage by request rows ([#577](https://github.com/audiagentic/bigcherry/issues/577)) ([5bcb0b0](https://github.com/audiagentic/bigcherry/commit/5bcb0b006cd0866500f733ba7691a3a810d52133))


### Performance

* **radiance:** RAD47 - gfx1100 MXFP4 GEMM rows: tiled from 17 rows up (27B 4 streams +27%, 6 streams +37%) ([#596](https://github.com/audiagentic/bigcherry/issues/596)) ([6195907](https://github.com/audiagentic/bigcherry/commit/6195907289734ba42f78a18e3962827b174e67ad))

## [1160.0.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-1140.0.0...bc-radiance-1160.0.0) (2026-11-03)


### New and changed

* **radiance:** pin radiance v1.16.0 (server hardening, the place caching layer, the RDNA3 libr3d plugin); plugins export the 1.16 ABI layout fingerprint ([#555](https://github.com/audiagentic/bigcherry/issues/555)). Release record, hardware evidence and known open items: docs/releases/notes/bc-radiance-1160.0.0.md

## [1140.0.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-180.0.0...bc-radiance-1140.0.0) (2026-11-03)


### New and changed

* **radiance:** pin radiance v1.14.0; production patch sets serving-core [rad0008, rad0002 promoted in #546], host-compat, serving-manifest. Release record, hardware evidence and known open items: docs/releases/notes/bc-radiance-1140.0.0.md

## [180.1.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-180.0.0...bc-radiance-180.1.0) (2026-10-21)


### New and changed

* **libr3:** checked runtime shared-memory layout_for(shape) beside the constexpr layout (RCS09-02) ([#321](https://github.com/audiagentic/bigcherry/issues/321)) ([bd1dd96](https://github.com/audiagentic/bigcherry/commit/bd1dd9669391760a2931268c6759599cebdb7e3d))
* **libr3:** expert width and every remaining expert-count site are runtime (RCS09-05b, 07) ([#328](https://github.com/audiagentic/bigcherry/issues/328)) ([f792f51](https://github.com/audiagentic/bigcherry/commit/f792f51e183abc63242effa3e07ac870780ee301))
* **libr3:** expert-backend contract with FP8 and W4 sizing and a capability probe (RCS05, RAD11) ([#317](https://github.com/audiagentic/bigcherry/issues/317)) ([5dc8fdc](https://github.com/audiagentic/bigcherry/commit/5dc8fdc370d47d21c93f0b42dc01c161a5b44450))
* **libr3:** FP8 decode GEMV for the small lane (RAD14) ([#331](https://github.com/audiagentic/bigcherry/issues/331)) ([09e3be7](https://github.com/audiagentic/bigcherry/commit/09e3be7deacf8c300b392f7da96263a0bbabb4ac))
* **libr3:** helper describes the container and refuses an unsupported format by name (RCS09, RAD11) ([#319](https://github.com/audiagentic/bigcherry/issues/319)) ([9ac5150](https://github.com/audiagentic/bigcherry/commit/9ac51507f7adab34e1e89d297e6ba4848fd60e28))
* **libr3:** helper small-lane classes, protocol v13, lane A/B harness (RAD12) ([#314](https://github.com/audiagentic/bigcherry/issues/314)) ([7bdd08a](https://github.com/audiagentic/bigcherry/commit/7bdd08ae5abb5006ad34aa2060d91869c21d6cbc))
* **libr3:** lane kernels and host references take hidden width, top-k and activation group from the model (RCS09-06) ([#327](https://github.com/audiagentic/bigcherry/issues/327)) ([fa17269](https://github.com/audiagentic/bigcherry/commit/fa17269a63c8768a04744390b6315e4e641da383))
* **libr3:** routing kernels take the expert count at runtime (RCS09-05a) ([#326](https://github.com/audiagentic/bigcherry/issues/326)) ([e912e49](https://github.com/audiagentic/bigcherry/commit/e912e49a8fd42612c6bfbef14fa1177402b8edac))
* **libr3:** shared-memory protocol v14 records the model shape (RCS09-03) ([#323](https://github.com/audiagentic/bigcherry/issues/323)) ([52b1c0d](https://github.com/audiagentic/bigcherry/commit/52b1c0dc286d9a929f5474fee63d306574fb699f))
* **libr3:** the worker serves FP8 block-scaled experts through radiance's fp8a8 grouped GEMM (RAD13) ([#330](https://github.com/audiagentic/bigcherry/issues/330)) ([73794d0](https://github.com/audiagentic/bigcherry/commit/73794d05bd18b287a08b9a2fca25a0f321d17b6d))
* **radiance:** bc-gfx12 hc_read prefill row stage 1 (RAD15) ([#358](https://github.com/audiagentic/bigcherry/issues/358)) ([48e2315](https://github.com/audiagentic/bigcherry/commit/48e2315bc0a95813c4651ac44fbeba4a36ec6f00))
* **radiance:** bc-gfx12 plugin - moe_gemm_q fp8a8 decode GEMV ahead of libr4d (RAD15) ([#355](https://github.com/audiagentic/bigcherry/issues/355)) ([92f76d9](https://github.com/audiagentic/bigcherry/commit/92f76d97eeb154e157cfd223d71f86e55bfcee4b))
* **radiance:** helper trace wall clock, per-round trace report and the RAD10-02 findings ([#307](https://github.com/audiagentic/bigcherry/issues/307)) ([996ca9c](https://github.com/audiagentic/bigcherry/commit/996ca9cf2f204b6215c6ca9502ce3c2b97be29f7))
* **radiance:** MTP prefix-cache bisection tooling, rad0006 and 35B model page (RAD14) ([#332](https://github.com/audiagentic/bigcherry/issues/332)) ([9f08db6](https://github.com/audiagentic/bigcherry/commit/9f08db69ab48ff427814f7c28ce1aa10d103aeca))
* **radiance:** packed-f16 two-row FP8 decode GEMV is the FP8 default (RAD14) ([#338](https://github.com/audiagentic/bigcherry/issues/338)) ([f28517c](https://github.com/audiagentic/bigcherry/commit/f28517cdcbd1f25e2785a70bb6a6ee2d34ff4c87))
* **radiance:** replay_hash cross-build bit-identity check for the worker (RCS05) ([#335](https://github.com/audiagentic/bigcherry/issues/335)) ([234c762](https://github.com/audiagentic/bigcherry/commit/234c7623f1c1cb6b976ce6588bdc509217f68b53))
* **radiance:** W4 packed-f16 decode GEMV, parked as evaluated (RAD14) ([#345](https://github.com/audiagentic/bigcherry/issues/345)) ([09b6850](https://github.com/audiagentic/bigcherry/commit/09b6850c4dddfd23e7e34e9f7b249ea9d70b5795))
* **radiance:** wide FP8 decode GEMV (default for FP8), few-rows FP8 kernel, multi-row GEMV (evaluated), kernel-time replay tooling (RAD14) ([#337](https://github.com/audiagentic/bigcherry/issues/337)) ([a12b39a](https://github.com/audiagentic/bigcherry/commit/a12b39ab7f153e9b66c51d5dfa6dfd31621b2760))


### Fixes

* **radiance:** core tests after the 1.8.0 bump and a CI job that runs them ([#305](https://github.com/audiagentic/bigcherry/issues/305)) ([df3cfe7](https://github.com/audiagentic/bigcherry/commit/df3cfe743f60ebab3b8ea03647d63ac4c6ddb128))

## [180.0.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-170.0.0...bc-radiance-180.0.0) (2026-10-18)


### New and changed

* **campaign:** engine acceptance against stored baselines; stored KLD references (MEN17) ([#269](https://github.com/audiagentic/bigcherry/issues/269)) ([8e3e5ab](https://github.com/audiagentic/bigcherry/commit/8e3e5abbe2710c2d3a1c6bcc1b14b2c4e0f412c0))
* **radiance:** 16-byte accesses for the helper's activation copy and result write (R3_MOE_HELPER_VECIO, default on) (RAD06) ([#278](https://github.com/audiagentic/bigcherry/issues/278)) ([c06dd0a](https://github.com/audiagentic/bigcherry/commit/c06dd0a9b1c26ba40866158c663ae1c116947810))
* **radiance:** bump to radiance 1.8.0 (MEN12) ([#289](https://github.com/audiagentic/bigcherry/issues/289)) ([b3eabe1](https://github.com/audiagentic/bigcherry/commit/b3eabe1f2bad747e6094ba1323fc836520eb8327))
* **radiance:** f16 WMMA MoE GEMM in the helpers (R3_MOE_FP16, default on) + replay sweep harness (RAD06) ([#276](https://github.com/audiagentic/bigcherry/issues/276)) ([5fea8be](https://github.com/audiagentic/bigcherry/commit/5fea8be8b0a5e8ebef13a3fa0a06083ad8bc0440))
* **radiance:** few-rows kernel on the decode lane (R3_MOE_HELPER_ROWS_SMALL, default off, evaluated) (RAD06) ([#283](https://github.com/audiagentic/bigcherry/issues/283)) ([52457e2](https://github.com/audiagentic/bigcherry/commit/52457e2ea12a23aa3bd1bc775ece5a4ca919b19a))
* **radiance:** fold the small lane's activation copy and table selection into the fused routing kernel (RAD06) ([#284](https://github.com/audiagentic/bigcherry/issues/284)) ([7d1d56a](https://github.com/audiagentic/bigcherry/commit/7d1d56a545b696500ca48829fd180c99f400d715))
* **radiance:** fused small-lane routing (decode +5-8%), unprofiled critical-path trace, protocol version 12 (RAD06/RHP) ([#282](https://github.com/audiagentic/bigcherry/issues/282)) ([6b485c4](https://github.com/audiagentic/bigcherry/commit/6b485c4c8f809abcfd0fa6868211d8c8187d0af6))
* **radiance:** helper request capture/replay + GEMV_LARGE_ROWS (default off) (RAD06) ([#274](https://github.com/audiagentic/bigcherry/issues/274)) ([cec1316](https://github.com/audiagentic/bigcherry/commit/cec1316175b920fe8b4dc37cc136dc7cab17232e))
* **radiance:** long-run tile split (R3_MOE_HELPER_MOE_LONG_ROWS, default off, evaluated) + counter/lab scripts (RAD06) ([#275](https://github.com/audiagentic/bigcherry/issues/275)) ([abaa59f](https://github.com/audiagentic/bigcherry/commit/abaa59fe9d2fea4d875058830ed1c6567f044876))
* **radiance:** per-layer cost-balanced helper ownership (R3_MOE_HELPER_BALANCE, default off, evaluated) (RAD06/RHP) ([#280](https://github.com/audiagentic/bigcherry/issues/280)) ([7e7caa0](https://github.com/audiagentic/bigcherry/commit/7e7caa0278f0ccef3b0517be59fcfa5ce0e2f0a5))
* **radiance:** R3_MOE_AHALF (default on), real-activation helper capture, replay sweep checksum (RAD06) ([#277](https://github.com/audiagentic/bigcherry/issues/277)) ([9fdbe49](https://github.com/audiagentic/bigcherry/commit/9fdbe495a0a14d0ad3f47da55d4c2767cff6125c))
* **radiance:** the engine's decode-sized activation staging joins the publish kernel (R3_MOE_HELPER_FUSESTAGE, default on) (RAD06) ([#285](https://github.com/audiagentic/bigcherry/issues/285)) ([2c7c170](https://github.com/audiagentic/bigcherry/commit/2c7c17080cea3ae16ec1ecaa8cb6c9a8c9e05dc9))
* **radiance:** validate the five libr3 helper settings with run evidence (RAD06) ([#288](https://github.com/audiagentic/bigcherry/issues/288)) ([8af8e58](https://github.com/audiagentic/bigcherry/commit/8af8e5826913b236d5abe96c8e9ca60894d4b0ff))
* **release:** native rewrite drift phase; pin-release confirm-then-run and release PR checks (MEN19) ([#271](https://github.com/audiagentic/bigcherry/issues/271)) ([20c2725](https://github.com/audiagentic/bigcherry/commit/20c27259c041372598adc72c7a65a01707e590f3))


### Fixes

* **radiance:** the lab source takes the host-compat patch set (MEN12) ([#291](https://github.com/audiagentic/bigcherry/issues/291)) ([66fb859](https://github.com/audiagentic/bigcherry/commit/66fb8596474c384406b20e2a36b3119c274befa4))


### Performance

* **radiance:** helper MoE GEMM tuning on gfx1100 - narrow down tile, request histogram, form/tile pins (RAD06) ([#273](https://github.com/audiagentic/bigcherry/issues/273)) ([31ee442](https://github.com/audiagentic/bigcherry/commit/31ee44280d53ff926c612586b5204f3cc7bb1335))

## [170.0.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-130.0.0...bc-radiance-170.0.0) (2026-10-17)


### New and changed

* **build:** engine-aware build context, GPU-target option and default binary (RAD01 PR 4) ([#220](https://github.com/audiagentic/bigcherry/issues/220)) ([7a88895](https://github.com/audiagentic/bigcherry/commit/7a88895c504fac8d5bebfa1f65d2b2761b4fa46e))
* **build:** lanes build the engine's runtime and components; build identity binds them (RAD01 PR 8b) ([#225](https://github.com/audiagentic/bigcherry/issues/225)) ([cfdd9cb](https://github.com/audiagentic/bigcherry/commit/cfdd9cb95d41658708ac4b0d3b637d1c56a26114))
* **campaign:** engine adapters and sidecar processes for server A/B; radiance helper layout (MEN11) ([#247](https://github.com/audiagentic/bigcherry/issues/247)) ([9b99b35](https://github.com/audiagentic/bigcherry/commit/9b99b35293dedf442bcae1f97210f3a40e73e51e))
* **campaign:** RAM-only CPU smoke test per engine, first phase of pin-bump (MEN15) ([#258](https://github.com/audiagentic/bigcherry/issues/258)) ([f8b514b](https://github.com/audiagentic/bigcherry/commit/f8b514b0fd1311748faf18e670f4d905bfec2c77))
* **config:** each engine declares its recipes; Radiance recipes (RAD01 PR 3) ([#219](https://github.com/audiagentic/bigcherry/issues/219)) ([e8ee8e5](https://github.com/audiagentic/bigcherry/commit/e8ee8e5f2c8e5fe739f87b792b1bd2dee9f2fb6f))
* engine registry for per-engine patch, overlay and vendor locations ([#87](https://github.com/audiagentic/bigcherry/issues/87)) ([c9a8c58](https://github.com/audiagentic/bigcherry/commit/c9a8c58eab5e86b5e6122e5d62c17a799949b8e4))
* engine serve specification and engine-neutral server runner (MEN03) ([#90](https://github.com/audiagentic/bigcherry/issues/90)) ([b72b150](https://github.com/audiagentic/bigcherry/commit/b72b1504e06656e86c89ca3dc730a0cad136d230))
* **experiment:** engine-neutral activation and KL quality evidence; radiance adapter (RAD01 PR 7) ([#223](https://github.com/audiagentic/bigcherry/issues/223)) ([5d7f649](https://github.com/audiagentic/bigcherry/commit/5d7f649d9217e815e9ea3d289d410ba5c173b615))
* **experiment:** KLD reports median, P99, P99.9 and top-1 agreement (RAD08) ([#249](https://github.com/audiagentic/bigcherry/issues/249)) ([9c6d9fe](https://github.com/audiagentic/bigcherry/commit/9c6d9fed0e775e1910bc9240ea0fa1acb7c354b7))
* **patch:** engine-declared patch ID prefix; radiance patches become rad0001-rad0003 ([#232](https://github.com/audiagentic/bigcherry/issues/232)) ([3527b6a](https://github.com/audiagentic/bigcherry/commit/3527b6afae7e9e60e6f0168a81e6d2eac50590c4))
* **radiance:** dev build platform - radiance lanes build through bigcherry build (RAD01 PR 8a) ([#224](https://github.com/audiagentic/bigcherry/issues/224)) ([0df06a3](https://github.com/audiagentic/bigcherry/commit/0df06a3c61f56b605951a255b3e1a9d4b467d7a7))
* **radiance:** engine patches 0001-0003 as patch packages (RAD01 PR 1) ([#210](https://github.com/audiagentic/bigcherry/issues/210)) ([dd1fbdf](https://github.com/audiagentic/bigcherry/commit/dd1fbdf290a2b0448f1cb481a38874cf416c9bc1))
* **radiance:** libr3 as a declared engine component with typed runtime settings (RAD01 PR 5) ([#221](https://github.com/audiagentic/bigcherry/issues/221)) ([678dc96](https://github.com/audiagentic/bigcherry/commit/678dc96d8088754255fceac8ea3362bee54c953e))
* **radiance:** port to radiance 1.7.0 - thin-hook rad0002, diagnostics in rad0004, plugins on ABI 17, natives ported (MEN12) ([#261](https://github.com/audiagentic/bigcherry/issues/261)) ([2d374d1](https://github.com/audiagentic/bigcherry/commit/2d374d183770cee668a240e2138a3b3cda886571))
* **radiance:** R3_MOE_HELPER_ROWS declares an activation marker; harness A/B recorded (MEN11) ([#250](https://github.com/audiagentic/bigcherry/issues/250)) ([a6abb30](https://github.com/audiagentic/bigcherry/commit/a6abb30290b6b4b6a6485cb2de238ca7610cde88))
* **radiance:** R9700 engine with two XTX expert helpers - GPU mailbox protocol, placement, decode GEMV (RAD02-RAD06, RAD09) ([#236](https://github.com/audiagentic/bigcherry/issues/236)) ([63cc952](https://github.com/audiagentic/bigcherry/commit/63cc952f5df67093a70e038c67a9d6185aac20bb))
* **radiance:** R9700 expert helper - shared-memory protocol and helper skeleton (slice 2a) ([#230](https://github.com/audiagentic/bigcherry/issues/230)) ([ba708fb](https://github.com/audiagentic/bigcherry/commit/ba708fb43e4f1295050b4ac3fddf453d056138c4))
* **release:** per-engine bump and audit gates; radiance component gate (MEN10 step f) ([#260](https://github.com/audiagentic/bigcherry/issues/260)) ([3cef62f](https://github.com/audiagentic/bigcherry/commit/3cef62f94079c9686049d942c1f064a0d063e8f6))
* **release:** radiance release line with upstream-as-major versions (MEN13) ([#252](https://github.com/audiagentic/bigcherry/issues/252)) ([1bd9b91](https://github.com/audiagentic/bigcherry/commit/1bd9b914f386ecff3100480b6ca067b689d1a87b))


### Fixes

* **release:** tag pins in pin-release, one release-record key, pull fetches tags ([#264](https://github.com/audiagentic/bigcherry/issues/264)) ([6dc1788](https://github.com/audiagentic/bigcherry/commit/6dc17887235ae9674c7e05fa39fb81b9230c2c4c))

## Changelog
