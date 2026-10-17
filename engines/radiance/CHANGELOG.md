# Changelog

## [170.1.0](https://github.com/audiagentic/bigcherry/compare/bc-radiance-170.0.0...bc-radiance-170.1.0) (2026-10-17)


### New and changed

* **campaign:** engine acceptance against stored baselines; stored KLD references (MEN17) ([#269](https://github.com/audiagentic/bigcherry/issues/269)) ([8e3e5ab](https://github.com/audiagentic/bigcherry/commit/8e3e5abbe2710c2d3a1c6bcc1b14b2c4e0f412c0))
* **release:** native rewrite drift phase; pin-release confirm-then-run and release PR checks (MEN19) ([#271](https://github.com/audiagentic/bigcherry/issues/271)) ([20c2725](https://github.com/audiagentic/bigcherry/commit/20c27259c041372598adc72c7a65a01707e590f3))


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
