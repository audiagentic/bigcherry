# Changelog

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
* **radiance:** Brutus build platform - radiance lanes build through bigcherry build (RAD01 PR 8a) ([#224](https://github.com/audiagentic/bigcherry/issues/224)) ([0df06a3](https://github.com/audiagentic/bigcherry/commit/0df06a3c61f56b605951a255b3e1a9d4b467d7a7))
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
