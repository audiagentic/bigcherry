# Changelog

## [1.3.0](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.2.0...bc-platform-1.3.0) (2026-10-21)


### New and changed

* **bench:** engine counters per round and the small-lane row limit A/B (RAD10-02) ([#303](https://github.com/audiagentic/bigcherry/issues/303)) ([120f850](https://github.com/audiagentic/bigcherry/commit/120f850c7070e9bb0891de69ed87262fafdc3aee))
* **bench:** stream-bench, concurrent streaming completions (RAD10-01) ([#300](https://github.com/audiagentic/bigcherry/issues/300)) ([6d450d6](https://github.com/audiagentic/bigcherry/commit/6d450d62e0d323e6bfb2aff0f4e05434baa1876c))
* **build:** engine-aware build context, GPU-target option and default binary (RAD01 PR 4) ([#220](https://github.com/audiagentic/bigcherry/issues/220)) ([7a88895](https://github.com/audiagentic/bigcherry/commit/7a88895c504fac8d5bebfa1f65d2b2761b4fa46e))
* **build:** lanes build the engine's runtime and components; build identity binds them (RAD01 PR 8b) ([#225](https://github.com/audiagentic/bigcherry/issues/225)) ([cfdd9cb](https://github.com/audiagentic/bigcherry/commit/cfdd9cb95d41658708ac4b0d3b637d1c56a26114))
* **campaign:** engine acceptance against stored baselines; stored KLD references (MEN17) ([#269](https://github.com/audiagentic/bigcherry/issues/269)) ([8e3e5ab](https://github.com/audiagentic/bigcherry/commit/8e3e5abbe2710c2d3a1c6bcc1b14b2c4e0f412c0))
* **campaign:** engine adapters and sidecar processes for server A/B; radiance helper layout (MEN11) ([#247](https://github.com/audiagentic/bigcherry/issues/247)) ([9b99b35](https://github.com/audiagentic/bigcherry/commit/9b99b35293dedf442bcae1f97210f3a40e73e51e))
* **campaign:** KLD through the server harness, component-setting evidence, radiance lab scripts retired (MEN11) ([#255](https://github.com/audiagentic/bigcherry/issues/255)) ([76279df](https://github.com/audiagentic/bigcherry/commit/76279df854850ee4faac6a858b611eb165257756))
* **campaign:** RAM-only CPU smoke test per engine, first phase of pin-bump (MEN15) ([#258](https://github.com/audiagentic/bigcherry/issues/258)) ([f8b514b](https://github.com/audiagentic/bigcherry/commit/f8b514b0fd1311748faf18e670f4d905bfec2c77))
* **campaign:** server A/B names its engine; first Radiance A/B through ab-benchmark (RAD01 PR 8c) ([#226](https://github.com/audiagentic/bigcherry/issues/226)) ([2052122](https://github.com/audiagentic/bigcherry/commit/2052122aed40f2009c971103ec192a65e84810a9))
* **check:** TR14.STRING_LITERAL_NEWLINE, a string literal containing a real line break ([#304](https://github.com/audiagentic/bigcherry/issues/304)) ([d16f402](https://github.com/audiagentic/bigcherry/commit/d16f402fdad000940d4ab1dec9fc19f3c2909066))
* **config:** each engine declares its recipes; Radiance recipes (RAD01 PR 3) ([#219](https://github.com/audiagentic/bigcherry/issues/219)) ([e8ee8e5](https://github.com/audiagentic/bigcherry/commit/e8ee8e5f2c8e5fe739f87b792b1bd2dee9f2fb6f))
* **config:** runtime profiles carry a component-checked environment (RAD01 PR 6) ([#222](https://github.com/audiagentic/bigcherry/issues/222)) ([930b3ae](https://github.com/audiagentic/bigcherry/commit/930b3ae99120219bf500387ecb12acad21d56b0c))
* **experiment:** engine-neutral activation and KL quality evidence; radiance adapter (RAD01 PR 7) ([#223](https://github.com/audiagentic/bigcherry/issues/223)) ([5d7f649](https://github.com/audiagentic/bigcherry/commit/5d7f649d9217e815e9ea3d289d410ba5c173b615))
* **experiment:** KLD reports median, P99, P99.9 and top-1 agreement (RAD08) ([#249](https://github.com/audiagentic/bigcherry/issues/249)) ([9c6d9fe](https://github.com/audiagentic/bigcherry/commit/9c6d9fed0e775e1910bc9240ea0fa1acb7c354b7))
* **libr3:** checked runtime shared-memory layout_for(shape) beside the constexpr layout (RCS09-02) ([#321](https://github.com/audiagentic/bigcherry/issues/321)) ([bd1dd96](https://github.com/audiagentic/bigcherry/commit/bd1dd9669391760a2931268c6759599cebdb7e3d))
* **libr3:** expert width and every remaining expert-count site are runtime (RCS09-05b, 07) ([#328](https://github.com/audiagentic/bigcherry/issues/328)) ([f792f51](https://github.com/audiagentic/bigcherry/commit/f792f51e183abc63242effa3e07ac870780ee301))
* **libr3:** expert-backend contract with FP8 and W4 sizing and a capability probe (RCS05, RAD11) ([#317](https://github.com/audiagentic/bigcherry/issues/317)) ([5dc8fdc](https://github.com/audiagentic/bigcherry/commit/5dc8fdc370d47d21c93f0b42dc01c161a5b44450))
* **libr3:** FP8 decode GEMV for the small lane (RAD14) ([#331](https://github.com/audiagentic/bigcherry/issues/331)) ([09e3be7](https://github.com/audiagentic/bigcherry/commit/09e3be7deacf8c300b392f7da96263a0bbabb4ac))
* **libr3:** helper small-lane classes, protocol v13, lane A/B harness (RAD12) ([#314](https://github.com/audiagentic/bigcherry/issues/314)) ([7bdd08a](https://github.com/audiagentic/bigcherry/commit/7bdd08ae5abb5006ad34aa2060d91869c21d6cbc))
* **libr3:** the worker serves FP8 block-scaled experts through radiance's fp8a8 grouped GEMM (RAD13) ([#330](https://github.com/audiagentic/bigcherry/issues/330)) ([73794d0](https://github.com/audiagentic/bigcherry/commit/73794d05bd18b287a08b9a2fca25a0f321d17b6d))
* **patch:** engine-declared patch ID prefix; radiance patches become rad0001-rad0003 ([#232](https://github.com/audiagentic/bigcherry/issues/232)) ([3527b6a](https://github.com/audiagentic/bigcherry/commit/3527b6afae7e9e60e6f0168a81e6d2eac50590c4))
* **patch:** engine-explicit patch registry and pinned source (RAD01 PR 2) ([#216](https://github.com/audiagentic/bigcherry/issues/216)) ([b79de13](https://github.com/audiagentic/bigcherry/commit/b79de13f61c3f12fc7c8c7b367c814635dc02e55))
* **patch:** engine-explicit patch registry and pinned source, every caller migrated (RAD01 PR 2) ([#213](https://github.com/audiagentic/bigcherry/issues/213)) ([a1a557d](https://github.com/audiagentic/bigcherry/commit/a1a557de536815ba9c6a4f2d66b766d30f677f02))
* **radiance:** 16-byte accesses for the helper's activation copy and result write (R3_MOE_HELPER_VECIO, default on) (RAD06) ([#278](https://github.com/audiagentic/bigcherry/issues/278)) ([c06dd0a](https://github.com/audiagentic/bigcherry/commit/c06dd0a9b1c26ba40866158c663ae1c116947810))
* **radiance:** bc-gfx12 plugin - moe_gemm_q fp8a8 decode GEMV ahead of libr4d (RAD15) ([#355](https://github.com/audiagentic/bigcherry/issues/355)) ([92f76d9](https://github.com/audiagentic/bigcherry/commit/92f76d97eeb154e157cfd223d71f86e55bfcee4b))
* **radiance:** bump to radiance 1.8.0 (MEN12) ([#289](https://github.com/audiagentic/bigcherry/issues/289)) ([b3eabe1](https://github.com/audiagentic/bigcherry/commit/b3eabe1f2bad747e6094ba1323fc836520eb8327))
* **radiance:** decode-path logprob quality check (RAD15) ([#356](https://github.com/audiagentic/bigcherry/issues/356)) ([048551a](https://github.com/audiagentic/bigcherry/commit/048551a3f9311407fa7413831349273b5a5451d0))
* **radiance:** engine patches 0001-0003 as patch packages (RAD01 PR 1) ([#210](https://github.com/audiagentic/bigcherry/issues/210)) ([dd1fbdf](https://github.com/audiagentic/bigcherry/commit/dd1fbdf290a2b0448f1cb481a38874cf416c9bc1))
* **radiance:** f16 WMMA MoE GEMM in the helpers (R3_MOE_FP16, default on) + replay sweep harness (RAD06) ([#276](https://github.com/audiagentic/bigcherry/issues/276)) ([5fea8be](https://github.com/audiagentic/bigcherry/commit/5fea8be8b0a5e8ebef13a3fa0a06083ad8bc0440))
* **radiance:** few-rows kernel on the decode lane (R3_MOE_HELPER_ROWS_SMALL, default off, evaluated) (RAD06) ([#283](https://github.com/audiagentic/bigcherry/issues/283)) ([52457e2](https://github.com/audiagentic/bigcherry/commit/52457e2ea12a23aa3bd1bc775ece5a4ca919b19a))
* **radiance:** Flash-Next A/B, trace and lane configs (RAD06) ([#347](https://github.com/audiagentic/bigcherry/issues/347)) ([5b2ebfc](https://github.com/audiagentic/bigcherry/commit/5b2ebfc238dc7f3ca108f14b1b68f3d8acf440a1))
* **radiance:** fold the small lane's activation copy and table selection into the fused routing kernel (RAD06) ([#284](https://github.com/audiagentic/bigcherry/issues/284)) ([7d1d56a](https://github.com/audiagentic/bigcherry/commit/7d1d56a545b696500ca48829fd180c99f400d715))
* **radiance:** FP8 WMMA hc_read down-product validated correct on gfx1201 (RAD15) ([#359](https://github.com/audiagentic/bigcherry/issues/359)) ([362c382](https://github.com/audiagentic/bigcherry/commit/362c382cb53d2183412a8970880b1db988786cf6))
* **radiance:** fused small-lane routing (decode +5-8%), unprofiled critical-path trace, protocol version 12 (RAD06/RHP) ([#282](https://github.com/audiagentic/bigcherry/issues/282)) ([6b485c4](https://github.com/audiagentic/bigcherry/commit/6b485c4c8f809abcfd0fa6868211d8c8187d0af6))
* **radiance:** helper request capture/replay + GEMV_LARGE_ROWS (default off) (RAD06) ([#274](https://github.com/audiagentic/bigcherry/issues/274)) ([cec1316](https://github.com/audiagentic/bigcherry/commit/cec1316175b920fe8b4dc37cc136dc7cab17232e))
* **radiance:** helper trace wall clock, per-round trace report and the RAD10-02 findings ([#307](https://github.com/audiagentic/bigcherry/issues/307)) ([996ca9c](https://github.com/audiagentic/bigcherry/commit/996ca9cf2f204b6215c6ca9502ce3c2b97be29f7))
* **radiance:** kbench_step joins each graph instance to its timed geometry, leaves out vision, counts MTP per draft pass (RAD15 step 1) ([#351](https://github.com/audiagentic/bigcherry/issues/351)) ([1106a51](https://github.com/audiagentic/bigcherry/commit/1106a5169be3bffd792248aa5aade834c2a9abdb))
* **radiance:** libr3 as a declared engine component with typed runtime settings (RAD01 PR 5) ([#221](https://github.com/audiagentic/bigcherry/issues/221)) ([678dc96](https://github.com/audiagentic/bigcherry/commit/678dc96d8088754255fceac8ea3362bee54c953e))
* **radiance:** long-run tile split (R3_MOE_HELPER_MOE_LONG_ROWS, default off, evaluated) + counter/lab scripts (RAD06) ([#275](https://github.com/audiagentic/bigcherry/issues/275)) ([abaa59f](https://github.com/audiagentic/bigcherry/commit/abaa59fe9d2fea4d875058830ed1c6567f044876))
* **radiance:** MTP prefix-cache bisection tooling, rad0006 and 35B model page (RAD14) ([#332](https://github.com/audiagentic/bigcherry/issues/332)) ([9f08db6](https://github.com/audiagentic/bigcherry/commit/9f08db69ab48ff427814f7c28ce1aa10d103aeca))
* **radiance:** packed-f16 two-row FP8 decode GEMV is the FP8 default (RAD14) ([#338](https://github.com/audiagentic/bigcherry/issues/338)) ([f28517c](https://github.com/audiagentic/bigcherry/commit/f28517cdcbd1f25e2785a70bb6a6ee2d34ff4c87))
* **radiance:** per-layer cost-balanced helper ownership (R3_MOE_HELPER_BALANCE, default off, evaluated) (RAD06/RHP) ([#280](https://github.com/audiagentic/bigcherry/issues/280)) ([7e7caa0](https://github.com/audiagentic/bigcherry/commit/7e7caa0278f0ccef3b0517be59fcfa5ce0e2f0a5))
* **radiance:** port to radiance 1.7.0 - thin-hook rad0002, diagnostics in rad0004, plugins on ABI 17, natives ported (MEN12) ([#261](https://github.com/audiagentic/bigcherry/issues/261)) ([2d374d1](https://github.com/audiagentic/bigcherry/commit/2d374d183770cee668a240e2138a3b3cda886571))
* **radiance:** Qwen3.6-35B-A3B engine-alone stream baseline (RCS09 genericity check) ([#316](https://github.com/audiagentic/bigcherry/issues/316)) ([3553e63](https://github.com/audiagentic/bigcherry/commit/3553e63269c4faec7106410598478084f1210b35))
* **radiance:** R3_MOE_AHALF (default on), real-activation helper capture, replay sweep checksum (RAD06) ([#277](https://github.com/audiagentic/bigcherry/issues/277)) ([9fdbe49](https://github.com/audiagentic/bigcherry/commit/9fdbe495a0a14d0ad3f47da55d4c2767cff6125c))
* **radiance:** R9700 engine with two XTX expert helpers - GPU mailbox protocol, placement, decode GEMV (RAD02-RAD06, RAD09) ([#236](https://github.com/audiagentic/bigcherry/issues/236)) ([63cc952](https://github.com/audiagentic/bigcherry/commit/63cc952f5df67093a70e038c67a9d6185aac20bb))
* **radiance:** replay_hash cross-build bit-identity check for the worker (RCS05) ([#335](https://github.com/audiagentic/bigcherry/issues/335)) ([234c762](https://github.com/audiagentic/bigcherry/commit/234c7623f1c1cb6b976ce6588bdc509217f68b53))
* **radiance:** RHP07-01 host-only placement simulator and trace format ([#293](https://github.com/audiagentic/bigcherry/issues/293)) ([77ae21a](https://github.com/audiagentic/bigcherry/commit/77ae21af245e8b1da7ec7f4e62ea58a93cc41c29))
* **radiance:** shape harness - kbench_step.py and selection-table config ([#346](https://github.com/audiagentic/bigcherry/issues/346)) ([6485132](https://github.com/audiagentic/bigcherry/commit/64851326aa3691b3e47368927f675a7766c0270f))
* **radiance:** the engine's decode-sized activation staging joins the publish kernel (R3_MOE_HELPER_FUSESTAGE, default on) (RAD06) ([#285](https://github.com/audiagentic/bigcherry/issues/285)) ([2c7c170](https://github.com/audiagentic/bigcherry/commit/2c7c17080cea3ae16ec1ecaa8cb6c9a8c9e05dc9))
* **radiance:** validate the five libr3 helper settings with run evidence (RAD06) ([#288](https://github.com/audiagentic/bigcherry/issues/288)) ([8af8e58](https://github.com/audiagentic/bigcherry/commit/8af8e5826913b236d5abe96c8e9ca60894d4b0ff))
* **radiance:** W4 packed-f16 decode GEMV, parked as evaluated (RAD14) ([#345](https://github.com/audiagentic/bigcherry/issues/345)) ([09b6850](https://github.com/audiagentic/bigcherry/commit/09b6850c4dddfd23e7e34e9f7b249ea9d70b5795))
* **radiance:** wide FP8 decode GEMV (default for FP8), few-rows FP8 kernel, multi-row GEMV (evaluated), kernel-time replay tooling (RAD14) ([#337](https://github.com/audiagentic/bigcherry/issues/337)) ([a12b39a](https://github.com/audiagentic/bigcherry/commit/a12b39ab7f153e9b66c51d5dfa6dfd31621b2760))
* **release:** add pin-bump source compile phase ([dfafa45](https://github.com/audiagentic/bigcherry/commit/dfafa45b78d635285b775c0fc6eeddfd6579161d))
* **release:** native rewrite drift phase; pin-release confirm-then-run and release PR checks (MEN19) ([#271](https://github.com/audiagentic/bigcherry/issues/271)) ([20c2725](https://github.com/audiagentic/bigcherry/commit/20c27259c041372598adc72c7a65a01707e590f3))
* **release:** per-engine bump and audit gates; radiance component gate (MEN10 step f) ([#260](https://github.com/audiagentic/bigcherry/issues/260)) ([3cef62f](https://github.com/audiagentic/bigcherry/commit/3cef62f94079c9686049d942c1f064a0d063e8f6))
* **release:** pin-bump builds the bumped engine through its lane and smokes that build (MEN16) ([#268](https://github.com/audiagentic/bigcherry/issues/268)) ([91c4782](https://github.com/audiagentic/bigcherry/commit/91c47829e7a4ef481080bda02174d389ee48fc95))
* **release:** pin-bump runs only the bumped engine's patch tests (MEN24) ([#349](https://github.com/audiagentic/bigcherry/issues/349)) ([7647042](https://github.com/audiagentic/bigcherry/commit/76470429e25de15e607968f3b253e6090df336d6))
* **release:** pin-bump runs the source compile check as its own phase ([8ecb3c3](https://github.com/audiagentic/bigcherry/commit/8ecb3c395d7ec23836059c105b250ccbfcfdaaf5))
* **release:** radiance release line with upstream-as-major versions (MEN13) ([#252](https://github.com/audiagentic/bigcherry/issues/252)) ([1bd9b91](https://github.com/audiagentic/bigcherry/commit/1bd9b914f386ecff3100480b6ca067b689d1a87b))
* **tooling:** bigcherry remote run/status/wait for GPU-host lab runs (MEN18) ([#267](https://github.com/audiagentic/bigcherry/issues/267)) ([8a88eb3](https://github.com/audiagentic/bigcherry/commit/8a88eb345bded7edff218781cfaaed34917722c2))
* **workspace:** journalled quarantine, restore and purge of a harvested run's bulk (WSL03) ([#315](https://github.com/audiagentic/bigcherry/issues/315)) ([197a114](https://github.com/audiagentic/bigcherry/commit/197a1144b84661399aaf0b7e4e7dfdd06f6b16ae))
* **workspace:** managed work root - run manifest, run creation, read-only inventory, bigcherry run and clean (WSL01, WSL03 skeleton) ([#311](https://github.com/audiagentic/bigcherry/issues/311)) ([afe993d](https://github.com/audiagentic/bigcherry/commit/afe993da6026a959879c5b1f9d3aee02b1da9328))
* **workspace:** run harvest writes a hash-verified evidence bundle (WSL02) ([#312](https://github.com/audiagentic/bigcherry/issues/312)) ([cd2fea0](https://github.com/audiagentic/bigcherry/commit/cd2fea0db2ae2687dacfe46a91bd6a80718c7ff6))


### Fixes

* **bench:** stream-bench accounting - token counts from usage, text events apart, finite-batch definition (RAD10) ([#302](https://github.com/audiagentic/bigcherry/issues/302)) ([9187b97](https://github.com/audiagentic/bigcherry/commit/9187b97a176fec6c6fddf7c16da68c7f420c4726))
* **campaign:** create the KL reference directory before recording (MEN11) ([#256](https://github.com/audiagentic/bigcherry/issues/256)) ([e566184](https://github.com/audiagentic/bigcherry/commit/e566184084443bcda04f3ca73b4fbb98949fdf2a))
* **campaign:** radiance helpers select their card with ROCR and HIP index 0 (MEN11) ([#248](https://github.com/audiagentic/bigcherry/issues/248)) ([c95d49e](https://github.com/audiagentic/bigcherry/commit/c95d49e28013275ef7f7f4cb2da2dd947b336b85))
* **lab:** helper-kld control arm gets the 44 GiB host pool the R9700-alone layout needs (MEN11) ([#257](https://github.com/audiagentic/bigcherry/issues/257)) ([ffb3b83](https://github.com/audiagentic/bigcherry/commit/ffb3b835b112354835a47be8eefadfb87e3103d9))
* **lab:** helper-kld thresholds from the first two Brutus runs (MEN11) ([#259](https://github.com/audiagentic/bigcherry/issues/259)) ([5aedfc1](https://github.com/audiagentic/bigcherry/commit/5aedfc1b7f42c5bff4dc9510e5084f8efa43c53c))
* **radiance:** core tests after the 1.8.0 bump and a CI job that runs them ([#305](https://github.com/audiagentic/bigcherry/issues/305)) ([df3cfe7](https://github.com/audiagentic/bigcherry/commit/df3cfe743f60ebab3b8ea03647d63ac4c6ddb128))
* **radiance:** placement simulator review fixes (RHP07-01) ([#294](https://github.com/audiagentic/bigcherry/issues/294)) ([62a0348](https://github.com/audiagentic/bigcherry/commit/62a0348ed8af1acc807b115aa1140ce67a22057a))
* **release:** pin-release names a release version from a squash-merged notes commit ([#297](https://github.com/audiagentic/bigcherry/issues/297)) ([74922a4](https://github.com/audiagentic/bigcherry/commit/74922a4ec4548b026d0b7577c80ffbc841742477))
* **release:** radiance compile gate configures with the build platform's compilers and options ([#262](https://github.com/audiagentic/bigcherry/issues/262)) ([ae92195](https://github.com/audiagentic/bigcherry/commit/ae92195a6566da456dee31d29f6f87d592359448))
* **release:** tag pins in pin-release, one release-record key, pull fetches tags ([#264](https://github.com/audiagentic/bigcherry/issues/264)) ([6dc1788](https://github.com/audiagentic/bigcherry/commit/6dc17887235ae9674c7e05fa39fb81b9230c2c4c))
* **workspace:** harden harvest per GPT review; RCS10 stream-derived settings note ([#313](https://github.com/audiagentic/bigcherry/issues/313)) ([3a4de33](https://github.com/audiagentic/bigcherry/commit/3a4de33d7c92ffe22b8605f4950a1a15c81d4312))


### Performance

* **radiance:** Flash-Next helpers own the hottest experts (RAD06) ([#272](https://github.com/audiagentic/bigcherry/issues/272)) ([7284db3](https://github.com/audiagentic/bigcherry/commit/7284db35920279b7fde4ecc96295090e63ba4e20))
* **radiance:** helper MoE GEMM tuning on gfx1100 - narrow down tile, request histogram, form/tile pins (RAD06) ([#273](https://github.com/audiagentic/bigcherry/issues/273)) ([31ee442](https://github.com/audiagentic/bigcherry/commit/31ee44280d53ff926c612586b5204f3cc7bb1335))


### Reverts

* RAD01 PR 2 engine-explicit registry ([#213](https://github.com/audiagentic/bigcherry/issues/213)) - offline-patch CI failed; merged in error ([#214](https://github.com/audiagentic/bigcherry/issues/214)) ([3eaa178](https://github.com/audiagentic/bigcherry/commit/3eaa1781e21dd0fd2a5ba2f27f7d2ebd4f7b2670))

## [1.2.0](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.1.1...bc-platform-1.2.0) (2026-10-12)


### New and changed

* **patch:** keep the shelved 1361 and 1356 code on main (evaluated, default off) ([71d6f49](https://github.com/audiagentic/bigcherry/commit/71d6f498ef45c0423ca24664d677b028f6643784))
* **patch:** keep the shelved code on main - 1361 as an evaluated patch, 1356 with its lock attempt, harness draft-model mode ([8db3f9a](https://github.com/audiagentic/bigcherry/commit/8db3f9ab325de4637e0aafe52b09bcb354c89eca))


### Fixes

* **patch:** 1361 puts its second-arena helpers in 1340's own header (the anchor moved there with the 1340 refactor) ([bf01c91](https://github.com/audiagentic/bigcherry/commit/bf01c9116adaeabef8d297787bb33224d8592d38))

## [1.1.1](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.1.0...bc-platform-1.1.1) (2026-10-11)


### Fixes

* **patch:** shelve 1356 and 1360 as evaluated (owner decision); QFP41 and QFP49 closed; main ruleset prepared for the owner (PA46) ([ff10025](https://github.com/audiagentic/bigcherry/commit/ff10025d48e79e516c85e97c75a0bacc374e1fb9))
* **patch:** shelve 1356 and 1360 as evaluated; close QFP41 and QFP49 ([5d9c163](https://github.com/audiagentic/bigcherry/commit/5d9c1631c6989e6a85202a9fe17e6c64d2b8e3fa))
* **release:** add resumable release phases ([433b4f6](https://github.com/audiagentic/bigcherry/commit/433b4f6b77e9534d1a3b7bd6d012687feb23d015))
* **release:** expose unpin and PR release flow ([455fc46](https://github.com/audiagentic/bigcherry/commit/455fc46e14fa5a7ec6a75edf0ee4616ea6583b3a))
* **release:** make publish dry-run concrete ([8327953](https://github.com/audiagentic/bigcherry/commit/8327953d5b991c99a5bc6a66e4d116f7f845eb1e))
* **release:** merge release record through a pull request ([c719844](https://github.com/audiagentic/bigcherry/commit/c7198447497ac677c436a0d17d87889f7f3af611))
* **release:** pin version in release-please config ([b374c1b](https://github.com/audiagentic/bigcherry/commit/b374c1b8aa17d72f6e4cfad3f1dcc816c20b6c32))
* **release:** pin-release goes through pull requests and sets the package version in the release-please config ([af9f6a6](https://github.com/audiagentic/bigcherry/commit/af9f6a618b6bbfb28243d1c51503eed22231f61a))
* **release:** reject conflicting release-as before writing notes ([2b2ef35](https://github.com/audiagentic/bigcherry/commit/2b2ef35d7e5a39f8e0e4e6ac3b64e788317a03c2))
* **release:** report post-tag notes as complete in dry run ([2e44aef](https://github.com/audiagentic/bigcherry/commit/2e44aef984a0950c0fb1b08c0d8a44f5f5c18b45))
* **release:** resume cleanly after release tag exists ([59d7420](https://github.com/audiagentic/bigcherry/commit/59d74207291ddb9f9130b367c3e9f6cb1341ed51))
* **release:** select previous tag across release families ([23bf18c](https://github.com/audiagentic/bigcherry/commit/23bf18cdd0ec5c0063974bd7fe79797548b8b90e))
* **release:** show current manifest on completed dry runs ([3c8df01](https://github.com/audiagentic/bigcherry/commit/3c8df01dc4690f769d76f5e61bf2b9b7f7af7622))
* **release:** validate release PR and clear release-as ([5f2a1e4](https://github.com/audiagentic/bigcherry/commit/5f2a1e44c77c5771cca043fab62a1306b3c76175))

## [1.1.0](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.0.1...bc-platform-1.1.0) (2026-10-11)


### New and changed

* **patch:** 1330 keeps decode-sized batches on the dense path; evaluated, ub1024 not adopted at ctx 245760 ([#77](https://github.com/audiagentic/bigcherry/issues/77)) ([456d7eb](https://github.com/audiagentic/bigcherry/commit/456d7ebfd8f225d01abca667c7599ca207bc6728))
* **patch:** add opt-in prompt KV padding (1362) ([4d690b5](https://github.com/audiagentic/bigcherry/commit/4d690b5013183de60b3d93eac899388e39ad1a85))
* **patch:** cross-card sum on a transfer stream, default off (1360) ([#145](https://github.com/audiagentic/bigcherry/issues/145)) ([1542986](https://github.com/audiagentic/bigcherry/commit/154298622e9e75e024425be6badcbdd3472350d4))
* **patch:** evict only the stale Meta split-state cache entry (1358); reject 1328 expert offload ([#93](https://github.com/audiagentic/bigcherry/issues/93)) ([aadc5a3](https://github.com/audiagentic/bigcherry/commit/aadc5a3338eb622dd56ed1033ea87930cc443f7a))
* **patch:** keep one prompt batch queued ahead of the cards with an MTP drafter, default off (1359) ([#135](https://github.com/audiagentic/bigcherry/issues/135)) ([94736f1](https://github.com/audiagentic/bigcherry/commit/94736f1b6b1ebdbb799c4b5f04eee5794c7cb46f))
* **patch:** lock counters for the dispatch workers (1356), standalone HIP probes and prefill trace tools ([#129](https://github.com/audiagentic/bigcherry/issues/129)) ([82bdc03](https://github.com/audiagentic/bigcherry/commit/82bdc0300f765b0ba2098675d8219f028fad1609))
* **patch:** opt-in KV padding so prompt batches reuse the graph (1362) ([215424f](https://github.com/audiagentic/bigcherry/commit/215424f62e0dd19dc1e3487bd114e09dcb097305))
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
* **lab:** quick-ab compares a new binary's second run, not its first ([a6a5eac](https://github.com/audiagentic/bigcherry/commit/a6a5eac8cffb8940ebe028168a8c321983c884f7))
* **lab:** quick-ab runs four arms so a new binary's first decode is not what gets compared; PA44 - 1281 slice record ([fad077c](https://github.com/audiagentic/bigcherry/commit/fad077c20431779ea19a4d69e9e216363dd25667))
* **patch:** 1281 - the MoE kernel edit's guard keeps its indent (without it the guard matches upstream's own line and the edit is skipped); test checks the barrier after the clear ([9e6be0a](https://github.com/audiagentic/bigcherry/commit/9e6be0a7168bc35dc14004fd264c0dd205371b49))
* **patch:** 1362 pads the pooled-block count too, so Qwen4Exp prompt batches reuse the graph ([30d3f1b](https://github.com/audiagentic/bigcherry/commit/30d3f1b2d6ef2b0ad5ed7e85109ed3125d07ac21))
* **patch:** b11552 - 1294 follows top-k's chunked row loop; 1333 anchors on the has_mixed decision alone ([9477827](https://github.com/audiagentic/bigcherry/commit/94778276bd17485a481f04f1ed4328f73309b0d1))
* **patch:** b11552 - 1321 and 1348 use set_embd_state for hidden-state rows; 1307's unary edit follows the contiguous branch; legacy-part tool ([142a0b8](https://github.com/audiagentic/bigcherry/commit/142a0b80866bb41c770937be43ecb18cf998ad11))
* **patch:** b11552 - 1337 state superseded, MoE cache experiments removed, 1338 test skipped until its rebuild (MET07) ([9fa5026](https://github.com/audiagentic/bigcherry/commit/9fa502648e4f00ea334d4f783750417edd8f275d))
* **patch:** b11552 - 1337 superseded by upstream [#29887](https://github.com/audiagentic/bigcherry/issues/29887) in the pin; 1338 parked until rebuilt on upstream's cache (MET07) ([6f1fe4f](https://github.com/audiagentic/bigcherry/commit/6f1fe4f6a077d47cc8d8b904a0d4972f9a22b141))
* **patch:** demote 1356_meta_dispatch_workers to evaluated (output race at 98K) ([#130](https://github.com/audiagentic/bigcherry/issues/130)) ([1ce33a5](https://github.com/audiagentic/bigcherry/commit/1ce33a5dbd43dee43d847129b524d876ac5c4b4a))
* **patch:** stabilize Qwen4Exp k-pool padding in 1362 ([0c18af2](https://github.com/audiagentic/bigcherry/commit/0c18af263cb1cc3a55b842d8f6e501283b656bdd))
* **release:** bc-platform back to 1.0.1; its next release is 1.1.0 ([1519efd](https://github.com/audiagentic/bigcherry/commit/1519efdbe63b235287a1c0d3d8d76f45eaf10e45))
* **release:** pin-release takes the release PR of its own component only ([c5107c7](https://github.com/audiagentic/bigcherry/commit/c5107c776d5dc5fcaf822eb0e2035e4f88cb6440))
* **release:** release notes list patches under engines/llamacpp/patches, not the pre-move patches/ directory ([ebe3527](https://github.com/audiagentic/bigcherry/commit/ebe35277bc7e91158057e5def23c26738b623427))
* **release:** restore bc-platform to 1.0.1 after the mistaken 11552.0.0 release ([d4bc138](https://github.com/audiagentic/bigcherry/commit/d4bc138d59903f47da3cf245eed3f4553857070c))


### Performance

* **patch:** registry and contracts read once per unchanged tree; shared production-source helper for mechanics tests ([e8224c7](https://github.com/audiagentic/bigcherry/commit/e8224c777532e2132a3dd84a75b1f4dda69bf366))


### Maintenance

* release notes bc-llamacpp-11552.0.0 ([d4912ed](https://github.com/audiagentic/bigcherry/commit/d4912ed917b6b37319cf0e8104cc805dd555ca99))

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
