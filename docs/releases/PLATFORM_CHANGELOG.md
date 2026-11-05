# Changelog

## [1.9.0](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.8.0...bc-platform-1.9.0) (2026-11-05)


### New and changed

* **tools:** pin-triage and a simpler pin-bump orchestrator ([#629](https://github.com/audiagentic/bigcherry/issues/629)) ([d0521e3](https://github.com/audiagentic/bigcherry/commit/d0521e33c5514c583cc9a8eca786c46ed7e3c2f5))

## [1.8.0](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.7.0...bc-platform-1.8.0) (2026-11-05)


### New and changed

* **bench:** accept-bench, draft acceptance by prompt kind with RAD23's five prompts at temperature 0, prompt and text digests recorded ([#638](https://github.com/audiagentic/bigcherry/issues/638)) ([5daded0](https://github.com/audiagentic/bigcherry/commit/5daded02860b3b80c06b37157dc4565b9411ce15))
* **patch:** a profile-evidence record carries forward across pin bumps when only the pin differs; rad0008 gets its promotion record from the evidence on hand ([#642](https://github.com/audiagentic/bigcherry/issues/642)) ([f2c2e2c](https://github.com/audiagentic/bigcherry/commit/f2c2e2c53c5339298667c92ebc775ba099a35338))

## [1.7.0](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.6.0...bc-platform-1.7.0) (2026-11-05)


### New and changed

* **radiance:** rad0009 reads the v1 manifest through the shared reader; the interim manifest format and bc-serve's own plan measurement are deleted (RSC08 S2b) ([#588](https://github.com/audiagentic/bigcherry/issues/588)) ([ef0579f](https://github.com/audiagentic/bigcherry/commit/ef0579fd48429c6ae37200056d5cec7c3c9ab9cf))
* **tools:** build system writes persistent per-run and per-stage logs with a summary; failures name the exact log (BRBC04) ([#574](https://github.com/audiagentic/bigcherry/issues/574)) ([dfb5e51](https://github.com/audiagentic/bigcherry/commit/dfb5e51397eabb72b33f245a9a9f3a5aa989d2e1))

## [1.6.0](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.5.0...bc-platform-1.6.0) (2026-11-05)


### New and changed

* **radiance:** RHP16 - offline replay of the live placement policy over a recorded heat series ([#620](https://github.com/audiagentic/bigcherry/issues/620)) ([6f45c66](https://github.com/audiagentic/bigcherry/commit/6f45c668ccf738bcc68049da703b9cfa29ce207a))
* **tools:** pin-bump resumes without arguments, commits its own records before the lane build, keeps the vendored tree on target ([#625](https://github.com/audiagentic/bigcherry/issues/625)) ([75a4f0f](https://github.com/audiagentic/bigcherry/commit/75a4f0f61510ec5944d4f6891a715baef8a1e019))


### Fixes

* **ci:** key the engine layer on every component source so a layer set is restored whole or not at all ([#622](https://github.com/audiagentic/bigcherry/issues/622)) ([7827c1a](https://github.com/audiagentic/bigcherry/commit/7827c1a691326c9320d2f3db71e7b992ac4d819b))

## [1.5.0](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.4.0...bc-platform-1.5.0) (2026-11-05)


### New and changed

* **build:** every build names a ROCm stack; SDK recorded, hashed, bound to CI image and launcher (BRVP06) ([#611](https://github.com/audiagentic/bigcherry/issues/611)) ([131c2b1](https://github.com/audiagentic/bigcherry/commit/131c2b1fe77e3cc15560e883cb6faa2ef9f8f875))
* **radiance:** live expert placement policy (RHP slices H, I, J): target map = the static placer's plan for the live heat ([#597](https://github.com/audiagentic/bigcherry/issues/597)) ([4d3c217](https://github.com/audiagentic/bigcherry/commit/4d3c2178ea78faf6070ed60b9abc20959d23d072))
* **radiance:** RHP16 step 2 - the live policy measures transaction time and bandwidth, update gaps, hot-set drift and sampling error (no setting derived yet) ([#619](https://github.com/audiagentic/bigcherry/issues/619)) ([8acb6d8](https://github.com/audiagentic/bigcherry/commit/8acb6d884848b4c09e7fb6774b2f58a34d0fcf40))


### Fixes

* **ci:** pass the compiler image digest into the prebuilt layer export ([#602](https://github.com/audiagentic/bigcherry/issues/602)) ([576d854](https://github.com/audiagentic/bigcherry/commit/576d8542ab8a4c0d4f8f0f8fa4a95ed9f9b19abd))
* **ci:** store the engine layer's source-tree link relative so prebuilt restore works ([#604](https://github.com/audiagentic/bigcherry/issues/604)) ([53a8969](https://github.com/audiagentic/bigcherry/commit/53a8969c53131853e131cb0b083f7fd496febf23))
* **radiance:** RHP16 - measure hot-set drift on per-interval counts, not on the heat the half-life decays; record the first measured values ([#621](https://github.com/audiagentic/bigcherry/issues/621)) ([18579f7](https://github.com/audiagentic/bigcherry/commit/18579f7396fc1cf3bbbe1b873c14f0499c750971))


### Performance

* **radiance:** RAD48 - native gfx11 int8 prefill tile (27B INT8 prefill GEMM 3.1x) ([#601](https://github.com/audiagentic/bigcherry/issues/601)) ([424a6e4](https://github.com/audiagentic/bigcherry/commit/424a6e47eb8462f1b6742d4f2964b817f84842e7))

## [1.4.0](https://github.com/audiagentic/bigcherry/compare/bc-platform-1.3.0...bc-platform-1.4.0) (2026-11-04)


### New and changed

* **bench:** ab-benchmark cells record speculative draft totals ([#463](https://github.com/audiagentic/bigcherry/issues/463)) ([c8758b5](https://github.com/audiagentic/bigcherry/commit/c8758b5c7bf2903fafbde03c3795b321e24bf8cb))
* **bench:** engine counters per round and the small-lane row limit A/B (RAD10-02) ([#303](https://github.com/audiagentic/bigcherry/issues/303)) ([436f0e5](https://github.com/audiagentic/bigcherry/commit/436f0e5ccbd1dc4f8a9cd4191cd5af2b7faee52f))
* **bench:** stream-bench, concurrent streaming completions (RAD10-01) ([#300](https://github.com/audiagentic/bigcherry/issues/300)) ([bdb285c](https://github.com/audiagentic/bigcherry/commit/bdb285c2e4a08b25b98f32757834880953c8bfda))
* **build:** BU01 Radiance CI cache, release bundle and kernel recompiles ([#584](https://github.com/audiagentic/bigcherry/issues/584)) ([e92fd1a](https://github.com/audiagentic/bigcherry/commit/e92fd1a20b20af491e2252455a516190d1db3891))
* **build:** engine-aware build context, GPU-target option and default binary (RAD01 PR 4) ([#220](https://github.com/audiagentic/bigcherry/issues/220)) ([6d92679](https://github.com/audiagentic/bigcherry/commit/6d92679d6538ca68437e891ef31fc7b58318b53b))
* **build:** lanes build the engine's runtime and components; build identity binds them (RAD01 PR 8b) ([#225](https://github.com/audiagentic/bigcherry/issues/225)) ([4e4341f](https://github.com/audiagentic/bigcherry/commit/4e4341f50846cde2231710afbde7897575354a44))
* **campaign:** engine acceptance against stored baselines; stored KLD references (MEN17) ([#269](https://github.com/audiagentic/bigcherry/issues/269)) ([dd0e43a](https://github.com/audiagentic/bigcherry/commit/dd0e43af6754f748979cae09251226f4af22576c))
* **campaign:** engine adapters and sidecar processes for server A/B; radiance helper layout (MEN11) ([#247](https://github.com/audiagentic/bigcherry/issues/247)) ([18351e1](https://github.com/audiagentic/bigcherry/commit/18351e138aa6892c6283fd2ce3303503b3fc55d5))
* **campaign:** KLD through the server harness, component-setting evidence, radiance lab scripts retired (MEN11) ([#255](https://github.com/audiagentic/bigcherry/issues/255)) ([3466c93](https://github.com/audiagentic/bigcherry/commit/3466c9301c2fbf6f0944a64be3b8b1a4a77edf06))
* **campaign:** RAM-only CPU smoke test per engine, first phase of pin-bump (MEN15) ([#258](https://github.com/audiagentic/bigcherry/issues/258)) ([cdf9ce8](https://github.com/audiagentic/bigcherry/commit/cdf9ce82d425a7f5fcbd67032e1a32018c5e3868))
* **campaign:** server A/B names its engine; first Radiance A/B through ab-benchmark (RAD01 PR 8c) ([#226](https://github.com/audiagentic/bigcherry/issues/226)) ([20d4841](https://github.com/audiagentic/bigcherry/commit/20d48411e178406754ccd62cd6aab020d617ba33))
* **check:** TR14.STRING_LITERAL_NEWLINE, a string literal containing a real line break ([#304](https://github.com/audiagentic/bigcherry/issues/304)) ([01edebe](https://github.com/audiagentic/bigcherry/commit/01edebecb9926a3b6f732c2c33217099a81e149e))
* **ci:** reuse prebuilt Radiance engine and component layers ([#593](https://github.com/audiagentic/bigcherry/issues/593)) ([19b3a96](https://github.com/audiagentic/bigcherry/commit/19b3a962c203c488a5b9cf7a82c8ed306390cf19))
* **config:** each engine declares its recipes; Radiance recipes (RAD01 PR 3) ([#219](https://github.com/audiagentic/bigcherry/issues/219)) ([2d54ae9](https://github.com/audiagentic/bigcherry/commit/2d54ae9d843288522ebc029eaecd5af390eb8405))
* **config:** runtime profiles carry a component-checked environment (RAD01 PR 6) ([#222](https://github.com/audiagentic/bigcherry/issues/222)) ([645158b](https://github.com/audiagentic/bigcherry/commit/645158b4c4155dc6c920bb82c075c690da138b96))
* **experiment:** engine-neutral activation and KL quality evidence; radiance adapter (RAD01 PR 7) ([#223](https://github.com/audiagentic/bigcherry/issues/223)) ([c817656](https://github.com/audiagentic/bigcherry/commit/c8176562a3ce734f084f6ac60fc6b89b17d37dfb))
* **experiment:** KLD reports median, P99, P99.9 and top-1 agreement (RAD08) ([#249](https://github.com/audiagentic/bigcherry/issues/249)) ([9d98fcb](https://github.com/audiagentic/bigcherry/commit/9d98fcb4373b7ee451a5b0fcfad69f6dd2f6d873))
* **lab:** kbench_ab - per-op kernel A/B runs with progress, resume, probe and strict comparison ([#587](https://github.com/audiagentic/bigcherry/issues/587)) ([2a5d5cf](https://github.com/audiagentic/bigcherry/commit/2a5d5cf88552cc446606c5fe44bac64967dd1804))
* **lab:** radiance lab scripts take ROCM_PATH and device ordinals from the environment settings (BRVP05) ([#450](https://github.com/audiagentic/bigcherry/issues/450)) ([6ee6022](https://github.com/audiagentic/bigcherry/commit/6ee60222f79648f2356ef91832af7d2b4a10c504))
* **libr3:** checked runtime shared-memory layout_for(shape) beside the constexpr layout (RCS09-02) ([#321](https://github.com/audiagentic/bigcherry/issues/321)) ([9f954a1](https://github.com/audiagentic/bigcherry/commit/9f954a1e253ab15fbd9c2d0523fc2508b3dd2114))
* **libr3:** expert width and every remaining expert-count site are runtime (RCS09-05b, 07) ([#328](https://github.com/audiagentic/bigcherry/issues/328)) ([98c0545](https://github.com/audiagentic/bigcherry/commit/98c054592e5bc33f07027dce3b644c3231604a14))
* **libr3:** expert-backend contract with FP8 and W4 sizing and a capability probe (RCS05, RAD11) ([#317](https://github.com/audiagentic/bigcherry/issues/317)) ([54b9f6e](https://github.com/audiagentic/bigcherry/commit/54b9f6eba5bc81d13ed1a5be5a26636c3844f1c7))
* **libr3:** FP8 decode GEMV for the small lane (RAD14) ([#331](https://github.com/audiagentic/bigcherry/issues/331)) ([8d01d69](https://github.com/audiagentic/bigcherry/commit/8d01d6952c38840e0a8aca08a537fcd5b88fb522))
* **libr3:** helper small-lane classes, protocol v13, lane A/B harness (RAD12) ([#314](https://github.com/audiagentic/bigcherry/issues/314)) ([a4b66af](https://github.com/audiagentic/bigcherry/commit/a4b66af38b614cf805f895e10130d0f7f2afe5bd))
* **libr3:** the worker serves FP8 block-scaled experts through radiance's fp8a8 grouped GEMM (RAD13) ([#330](https://github.com/audiagentic/bigcherry/issues/330)) ([44f7637](https://github.com/audiagentic/bigcherry/commit/44f7637b7d1a1a6f201afbacaa4f14c6138fe5c4))
* **patch:** add opt-in prompt KV padding (1362) ([6b9dd85](https://github.com/audiagentic/bigcherry/commit/6b9dd85324df8233f98162af04baf125f404da14))
* **patch:** engine-declared patch ID prefix; radiance patches become rad0001-rad0003 ([#232](https://github.com/audiagentic/bigcherry/issues/232)) ([34cf484](https://github.com/audiagentic/bigcherry/commit/34cf4840aaac52a17f8f0f37cc79bdc1849f3a3f))
* **patch:** engine-explicit patch registry and pinned source (RAD01 PR 2) ([#216](https://github.com/audiagentic/bigcherry/issues/216)) ([1328f1e](https://github.com/audiagentic/bigcherry/commit/1328f1ec993598765c609dea90b06e9fd8f6d1e3))
* **patch:** engine-explicit patch registry and pinned source, every caller migrated (RAD01 PR 2) ([#213](https://github.com/audiagentic/bigcherry/issues/213)) ([7d0de68](https://github.com/audiagentic/bigcherry/commit/7d0de68c5fabacb3b972ac098c8ce16db751acb7))
* **patch:** keep the shelved 1361 and 1356 code on main (evaluated, default off) ([d894ac1](https://github.com/audiagentic/bigcherry/commit/d894ac1c37a6a961f09f0eb667f1b65dadf2e551))
* **patch:** keep the shelved code on main - 1361 as an evaluated patch, 1356 with its lock attempt, harness draft-model mode ([1b6aca6](https://github.com/audiagentic/bigcherry/commit/1b6aca6da7ecb3fbc9ddd7622ffe4595a00b29fe))
* **patch:** opt-in KV padding so prompt batches reuse the graph (1362) ([1c5c4f2](https://github.com/audiagentic/bigcherry/commit/1c5c4f20476296f6b7aafdbfb09d049c5ccd44bf))
* **patch:** promote rad0002_moe_remote_handoff ([#546](https://github.com/audiagentic/bigcherry/issues/546)) ([f1d49c7](https://github.com/audiagentic/bigcherry/commit/f1d49c78c747f335371e5976049e06c034bbab2b))
* **patch:** promote rad0008_hc_upmix_rt_sweep to validated in serving-core (RAD18) ([#389](https://github.com/audiagentic/bigcherry/issues/389)) ([1787a7a](https://github.com/audiagentic/bigcherry/commit/1787a7ad066769323c3ac9c064fef33c7cd21f71))
* **radiance:** 16-byte accesses for the helper's activation copy and result write (R3_MOE_HELPER_VECIO, default on) (RAD06) ([#278](https://github.com/audiagentic/bigcherry/issues/278)) ([e5f778c](https://github.com/audiagentic/bigcherry/commit/e5f778c263d6c8d2746c5b7ca1c98a21dbaafdcf))
* **radiance:** arm helper small-lane waits only inside an inference burst (RAD20 Option A) ([#422](https://github.com/audiagentic/bigcherry/issues/422)) ([cb80f77](https://github.com/audiagentic/bigcherry/commit/cb80f770a8bec0050dffd20422ffc27095d2c02a))
* **radiance:** bc-gfx12 moe GEMV default on - owner-accepted single-model evidence (RAD15) ([#361](https://github.com/audiagentic/bigcherry/issues/361)) ([897fc8d](https://github.com/audiagentic/bigcherry/commit/897fc8d8131ebf271efbe719bf4915b64ba84dd4))
* **radiance:** bc-gfx12 plugin - moe_gemm_q fp8a8 decode GEMV ahead of libr4d (RAD15) ([#355](https://github.com/audiagentic/bigcherry/issues/355)) ([87a7d6e](https://github.com/audiagentic/bigcherry/commit/87a7d6ec04b29d264663105ea109d803a3cc7133))
* **radiance:** bump to radiance 1.8.0 (MEN12) ([#289](https://github.com/audiagentic/bigcherry/issues/289)) ([d912921](https://github.com/audiagentic/bigcherry/commit/d912921b37eab73d8ad28d2cf2fe74c2e8e7dd18))
* **radiance:** card types and layouts are configuration, not code (RAD23) ([#472](https://github.com/audiagentic/bigcherry/issues/472)) ([2439b90](https://github.com/audiagentic/bigcherry/commit/2439b90de238836eb7813e5b348692de4fa22c4b))
* **radiance:** complete rad0007's FP8 down-product epilogue, validated at real scale (RAD17) ([#371](https://github.com/audiagentic/bigcherry/issues/371)) ([06225d4](https://github.com/audiagentic/bigcherry/commit/06225d455a94e2deece64e52d92750e92a2befe1))
* **radiance:** confirm hc_read's real W_down row stride against the live container (RAD17) ([#366](https://github.com/audiagentic/bigcherry/issues/366)) ([200f4f6](https://github.com/audiagentic/bigcherry/commit/200f4f6cc87cbf7ab185350641a57cca6c41a9ef))
* **radiance:** decode-path logprob quality check (RAD15) ([#356](https://github.com/audiagentic/bigcherry/issues/356)) ([c7075dd](https://github.com/audiagentic/bigcherry/commit/c7075dd3de8d16b1f4fa5d9a7a51d34f444884bd))
* **radiance:** engine fails fast when a helper dies (RAD20 helper liveness) ([#427](https://github.com/audiagentic/bigcherry/issues/427)) ([83834d9](https://github.com/audiagentic/bigcherry/commit/83834d9a53925d3e695abac8f10facc6642c6fdd))
* **radiance:** engine patches 0001-0003 as patch packages (RAD01 PR 1) ([#210](https://github.com/audiagentic/bigcherry/issues/210)) ([d0ae82e](https://github.com/audiagentic/bigcherry/commit/d0ae82ea26cb56214f18b5d0c44bf5912c6414a1))
* **radiance:** f16 WMMA MoE GEMM in the helpers (R3_MOE_FP16, default on) + replay sweep harness (RAD06) ([#276](https://github.com/audiagentic/bigcherry/issues/276)) ([104a296](https://github.com/audiagentic/bigcherry/commit/104a2961042b51583845cc357486f7fe87dd22a9))
* **radiance:** few-rows kernel on the decode lane (R3_MOE_HELPER_ROWS_SMALL, default off, evaluated) (RAD06) ([#283](https://github.com/audiagentic/bigcherry/issues/283)) ([a43814a](https://github.com/audiagentic/bigcherry/commit/a43814a47fed82e54c5eedabd501463d11d75505))
* **radiance:** Flash-Next A/B, trace and lane configs (RAD06) ([#347](https://github.com/audiagentic/bigcherry/issues/347)) ([4d29289](https://github.com/audiagentic/bigcherry/commit/4d2928984360d494cd598a1ecc2b24b379e3afe1))
* **radiance:** fold the small lane's activation copy and table selection into the fused routing kernel (RAD06) ([#284](https://github.com/audiagentic/bigcherry/issues/284)) ([8ee6fe9](https://github.com/audiagentic/bigcherry/commit/8ee6fe9a24f3f3e41be78cce8291d2eacd1e96ba))
* **radiance:** fp16 helper-row combine with an F16C CPU path, off by default (RAD18, RAD23) ([#404](https://github.com/audiagentic/bigcherry/issues/404)) ([a9f0faa](https://github.com/audiagentic/bigcherry/commit/a9f0faac4f11d618490f285ae574f70982bd3609))
* **radiance:** FP8 down-product cross-thread quantiser proven end-to-end on real data (RAD15) ([#360](https://github.com/audiagentic/bigcherry/issues/360)) ([9c85765](https://github.com/audiagentic/bigcherry/commit/9c85765eb811f07d6251e00d1111ae99db0ae647))
* **radiance:** FP8 WMMA hc_read down-product validated correct on gfx1201 (RAD15) ([#359](https://github.com/audiagentic/bigcherry/issues/359)) ([bdac8a3](https://github.com/audiagentic/bigcherry/commit/bdac8a3a9176371d888b9d65afd3d87cbba63253))
* **radiance:** fused small-lane routing (decode +5-8%), unprofiled critical-path trace, protocol version 12 (RAD06/RHP) ([#282](https://github.com/audiagentic/bigcherry/issues/282)) ([18b9c68](https://github.com/audiagentic/bigcherry/commit/18b9c680989d03f1626d77fb31427674208b51dd))
* **radiance:** gfx1030 backend for the remote-MoE helper (RCS07) ([#464](https://github.com/audiagentic/bigcherry/issues/464)) ([e19caab](https://github.com/audiagentic/bigcherry/commit/e19caab087a0c3b9203edaf49469017d62c97e5e))
* **radiance:** hc_up_mix_wide_kernel PF2 RT sweep, default off (RAD17/RAD18) ([#381](https://github.com/audiagentic/bigcherry/issues/381)) ([7f81bb1](https://github.com/audiagentic/bigcherry/commit/7f81bb1a1d220f697b18572640a30938b5178b6f))
* **radiance:** heat-placement plans reviewed against 1.14.0; seed planner worker_heat.h (RHP) ([#471](https://github.com/audiagentic/bigcherry/issues/471)) ([c8b30ca](https://github.com/audiagentic/bigcherry/commit/c8b30cad4fef0ec5b7857633da2f418571991320))
* **radiance:** helper request capture/replay + GEMV_LARGE_ROWS (default off) (RAD06) ([#274](https://github.com/audiagentic/bigcherry/issues/274)) ([24b4563](https://github.com/audiagentic/bigcherry/commit/24b45633adcc86848325f3fcffeb42ea385a791f))
* **radiance:** helper small lanes run on efficiency cores when the CPU has them (PWR03) ([#448](https://github.com/audiagentic/bigcherry/issues/448)) ([89d98c7](https://github.com/audiagentic/bigcherry/commit/89d98c7d7060b5b6a295ed4ca0fc4559816b4b65))
* **radiance:** helper trace wall clock, per-round trace report and the RAD10-02 findings ([#307](https://github.com/audiagentic/bigcherry/issues/307)) ([fa64110](https://github.com/audiagentic/bigcherry/commit/fa6411044015cf4ddd5c2fd922ad473e7a41afee))
* **radiance:** helper-row combine (off by default), blocking waits, and the internal-refs redaction ([#394](https://github.com/audiagentic/bigcherry/issues/394)) ([892f74d](https://github.com/audiagentic/bigcherry/commit/892f74d9e736e7889e039141e46b31ceec3a5d25))
* **radiance:** helper-row combine for two to four helpers (protocol v20, RAD23) ([#485](https://github.com/audiagentic/bigcherry/issues/485)) ([40b4802](https://github.com/audiagentic/bigcherry/commit/40b48023f7b5e91d9d593a34219a703e25b9cc03))
* **radiance:** inference-burst lifecycle callback from the engine loop (RAD20 phase 1) ([#420](https://github.com/audiagentic/bigcherry/issues/420)) ([4df1d51](https://github.com/audiagentic/bigcherry/commit/4df1d51d10c1208f18715810a5663ee9c7c960dc))
* **radiance:** kbench_step joins each graph instance to its timed geometry, leaves out vision, counts MTP per draft pass (RAD15 step 1) ([#351](https://github.com/audiagentic/bigcherry/issues/351)) ([8005353](https://github.com/audiagentic/bigcherry/commit/800535375fb5d73c2b2dc43fd8a2bd6ff58b4007))
* **radiance:** libr2 f16 tile without LDS exchange; RAD23 findings on the 6900 XT (RAD23) ([#473](https://github.com/audiagentic/bigcherry/issues/473)) ([b22e0ae](https://github.com/audiagentic/bigcherry/commit/b22e0aeab74c6851a99317fc94644e6ca48da94f))
* **radiance:** libr2, the RDNA2 kernel plugin, as its own component (RCS07, RAD23) ([#469](https://github.com/audiagentic/bigcherry/issues/469)) ([f7e131b](https://github.com/audiagentic/bigcherry/commit/f7e131b1ec837a7f856e4c06d7b9b9c14530bfbb))
* **radiance:** libr3 as a declared engine component with typed runtime settings (RAD01 PR 5) ([#221](https://github.com/audiagentic/bigcherry/issues/221)) ([69220d4](https://github.com/audiagentic/bigcherry/commit/69220d41415c3454ebb3a8bf247881c2e70ab290))
* **radiance:** live route-histogram snapshots while the engine runs (RHP02, RHP08) ([#474](https://github.com/audiagentic/bigcherry/issues/474)) ([76a4288](https://github.com/audiagentic/bigcherry/commit/76a42887c6269a5c4ba709189ba9c67383734289))
* **radiance:** long-run tile split (R3_MOE_HELPER_MOE_LONG_ROWS, default off, evaluated) + counter/lab scripts (RAD06) ([#275](https://github.com/audiagentic/bigcherry/issues/275)) ([7588efe](https://github.com/audiagentic/bigcherry/commit/7588efeebd5a4c06cb74d5198a9ad20d35ddc96b))
* **radiance:** moe-remote mailbox protocol v23 layout - canonical owner generations and placement epochs (RHP11 slice A) ([#513](https://github.com/audiagentic/bigcherry/issues/513)) ([99eb787](https://github.com/audiagentic/bigcherry/commit/99eb7870250cbc5f1635c4b8b49269843463a75b))
* **radiance:** MTP prefix-cache bisection tooling, rad0006 and 35B model page (RAD14) ([#332](https://github.com/audiagentic/bigcherry/issues/332)) ([aafa7e9](https://github.com/audiagentic/bigcherry/commit/aafa7e91f6e8b55c7f06d235e2d180c33cee0ba9))
* **radiance:** packed-f16 two-row FP8 decode GEMV is the FP8 default (RAD14) ([#338](https://github.com/audiagentic/bigcherry/issues/338)) ([daef528](https://github.com/audiagentic/bigcherry/commit/daef52809e789cc6023374adf2775810afe2682b))
* **radiance:** packed-f16 W4 decode GEMV is the helper default (validated, RAD23) ([#509](https://github.com/audiagentic/bigcherry/issues/509)) ([9ec7721](https://github.com/audiagentic/bigcherry/commit/9ec7721a256256c19adceec98d5bbafd3511a4e2))
* **radiance:** per-iteration dump for the RAD20 barrier-wake harness ([#425](https://github.com/audiagentic/bigcherry/issues/425)) ([792eb59](https://github.com/audiagentic/bigcherry/commit/792eb59490bcaf888178d850b9925515d8e2f079))
* **radiance:** per-layer cost-balanced helper ownership (R3_MOE_HELPER_BALANCE, default off, evaluated) (RAD06/RHP) ([#280](https://github.com/audiagentic/bigcherry/issues/280)) ([afc8ca5](https://github.com/audiagentic/bigcherry/commit/afc8ca59218f15b31105a1860940f160b03981ea))
* **radiance:** per-pass placement ticket via an ordered pre-pass kernel (RHP11 slice C step 1) ([#522](https://github.com/audiagentic/bigcherry/issues/522)) ([70b2114](https://github.com/audiagentic/bigcherry/commit/70b2114eb38a844abe12a482bd96b393c159f47c))
* **radiance:** per-request helper stage stamps and the combine's per-step timing in TRACE (protocol v21, RAD40) ([#496](https://github.com/audiagentic/bigcherry/issues/496)) ([3948d14](https://github.com/audiagentic/bigcherry/commit/3948d14ebfad5906e6c61fead14f2353939b83c3))
* **radiance:** placement-epoch controller, helper slot ledger and peak-load heat policy (RHP11 slice B) ([#515](https://github.com/audiagentic/bigcherry/issues/515)) ([bbe298d](https://github.com/audiagentic/bigcherry/commit/bbe298d6d024a2526c5665ba0801e59dd15e5b1d))
* **radiance:** port to radiance 1.7.0 - thin-hook rad0002, diagnostics in rad0004, plugins on ABI 17, natives ported (MEN12) ([#261](https://github.com/audiagentic/bigcherry/issues/261)) ([16975a1](https://github.com/audiagentic/bigcherry/commit/16975a1cca169cb01e18a8bf9aadf26ba65414f7))
* **radiance:** PWR01 whole-system power baseline harness ([#435](https://github.com/audiagentic/bigcherry/issues/435)) ([9b955fe](https://github.com/audiagentic/bigcherry/commit/9b955fea8f1373e183b799293e2bc75b16e517cf))
* **radiance:** PWR02 sweep and soak tooling (clock, voltage offset, power cap; decode text identity) ([#446](https://github.com/audiagentic/bigcherry/issues/446)) ([34d9de3](https://github.com/audiagentic/bigcherry/commit/34d9de39ec1425c53c588206f629202b2cb1bcfb))
* **radiance:** qwen27b-2xtx layout - Qwen3.8-27B MXFP4 tensor-parallel over two RDNA3 cards ([#573](https://github.com/audiagentic/bigcherry/issues/573)) ([509d991](https://github.com/audiagentic/bigcherry/commit/509d991b83c0f42ad0d6295beb22a9feefe5da99))
* **radiance:** Qwen3.6-35B-A3B engine-alone stream baseline (RCS09 genericity check) ([#316](https://github.com/audiagentic/bigcherry/issues/316)) ([8d899b0](https://github.com/audiagentic/bigcherry/commit/8d899b063fbebe48b9958265b46f7b129f2d3ef0))
* **radiance:** R3_MOE_AHALF (default on), real-activation helper capture, replay sweep checksum (RAD06) ([#277](https://github.com/audiagentic/bigcherry/issues/277)) ([43ef74f](https://github.com/audiagentic/bigcherry/commit/43ef74fc81aedcdb06b5e6df7a2c7d302c049ee4))
* **radiance:** R9700 engine with two XTX expert helpers - GPU mailbox protocol, placement, decode GEMV (RAD02-RAD06, RAD09) ([#236](https://github.com/audiagentic/bigcherry/issues/236)) ([88e9abc](https://github.com/audiagentic/bigcherry/commit/88e9abcb795f87fa31f753a3dbbcfb42f2ed051c))
* **radiance:** rad0007 FP8 WMMA down-product scaffold, inert (RAD15/RAD17) ([#370](https://github.com/audiagentic/bigcherry/issues/370)) ([dec8f43](https://github.com/audiagentic/bigcherry/commit/dec8f43fe4cbd663ed8a444c3b0b6e46e14b5510))
* **radiance:** rad0009 serving manifest - engine starts from a strict bc-serve manifest (RSC07) ([#523](https://github.com/audiagentic/bigcherry/issues/523)) ([54c8b3c](https://github.com/audiagentic/bigcherry/commit/54c8b3c6bd5864e018bbf2d5aaff60f5e09111d5))
* **radiance:** RAD20 lab harnesses from the blocking-lanes branch ([#433](https://github.com/audiagentic/bigcherry/issues/433)) ([e114f91](https://github.com/audiagentic/bigcherry/commit/e114f9130d8dd3cf332b53122b90639656db5d6b))
* **radiance:** RAD20 option B barrier wake-latency lab harness ([#423](https://github.com/audiagentic/bigcherry/issues/423)) ([34f39ec](https://github.com/audiagentic/bigcherry/commit/34f39ecce188f6fc8211b467cca135e9a021df71))
* **radiance:** RAD25 harness - blocking completion events vs yield (not adopted) ([#432](https://github.com/audiagentic/bigcherry/issues/432)) ([eea6dfb](https://github.com/audiagentic/bigcherry/commit/eea6dfb0200fa16cd185e29779eb4d0c7fbf277c))
* **radiance:** remote-MoE segment laid out for its slot count, up to four helpers (RAD23 stage 2) ([#467](https://github.com/audiagentic/bigcherry/issues/467)) ([b73ed9a](https://github.com/audiagentic/bigcherry/commit/b73ed9abc12a95c589ada3a2039daeee327ebba9))
* **radiance:** replay_hash cross-build bit-identity check for the worker (RCS05) ([#335](https://github.com/audiagentic/bigcherry/issues/335)) ([88a7ccf](https://github.com/audiagentic/bigcherry/commit/88a7ccfa4ae417fa4a2e7f6251146615cebe8a35))
* **radiance:** RHP slice H part 1 - heat source interface and policy-input builder (card-free, not wired) ([#591](https://github.com/audiagentic/bigcherry/issues/591)) ([40829e0](https://github.com/audiagentic/bigcherry/commit/40829e0dfff5ee5bbf8e03c52e528956f5fb9aee))
* **radiance:** RHP01 part 1 - rad0010 routing-evidence diagnostic patch ([#582](https://github.com/audiagentic/bigcherry/issues/582)) ([f48cc55](https://github.com/audiagentic/bigcherry/commit/f48cc5560e612faa81163627019223a557e19169))
* **radiance:** RHP07-01 host-only placement simulator and trace format ([#293](https://github.com/audiagentic/bigcherry/issues/293)) ([af8de5d](https://github.com/audiagentic/bigcherry/commit/af8de5df9d0f8991a32ff93c7b37fa04a3cf2a97))
* **radiance:** RHP11 slice C - per-step admission, consumption proof and safe placement drain ([#542](https://github.com/audiagentic/bigcherry/issues/542)) ([2f5fc39](https://github.com/audiagentic/bigcherry/commit/2f5fc39df5e11e30bef7d7109998d4d2506c8663))
* **radiance:** RHP11 slice D - live owner-map generations on the engine GPU, full epoch cycle ([#557](https://github.com/audiagentic/bigcherry/issues/557)) ([78704e2](https://github.com/audiagentic/bigcherry/commit/78704e2fa32676b2ac38821cc7e031b804ea396c))
* **radiance:** RHP11 slice E - live helper slot migration ([#558](https://github.com/audiagentic/bigcherry/issues/558)) ([c78a819](https://github.com/audiagentic/bigcherry/commit/c78a8191d57004b6cd67d5645a74f0f2f5b158c4))
* **radiance:** RHP11 slices F and G - combine epoch checks, v23 segment requirement, forced-move harness ([#572](https://github.com/audiagentic/bigcherry/issues/572)) ([aa38683](https://github.com/audiagentic/bigcherry/commit/aa38683d3dc457310dd88e4d15d91bfa8eddc94f))
* **radiance:** serving component - strict C++ schema, loader, detection and bc-serve explain (RSC01 slice 1) ([#521](https://github.com/audiagentic/bigcherry/issues/521)) ([68d6a62](https://github.com/audiagentic/bigcherry/commit/68d6a62b6c712b0f0c882efd16c22d05f971b3a9))
* **radiance:** shape harness - kbench_step.py and selection-table config ([#346](https://github.com/audiagentic/bigcherry/issues/346)) ([db0724a](https://github.com/audiagentic/bigcherry/commit/db0724ae93b0525a9cafe608e292d8b5b0bbc455))
* **radiance:** sparse large-lane activation copy; 6900 XT floor, tail and bump findings (RAD23) ([#478](https://github.com/audiagentic/bigcherry/issues/478)) ([c6fcb49](https://github.com/audiagentic/bigcherry/commit/c6fcb497e311d2c283b5c1ab8a6e2b64c65c7434))
* **radiance:** standalone reproducer for the gfx1100 MES idle-power defect ([#421](https://github.com/audiagentic/bigcherry/issues/421)) ([6fa6186](https://github.com/audiagentic/bigcherry/commit/6fa6186aceeed85b5451211c200feadd7a753ae1))
* **radiance:** the engine checks its live memory plan at every start through the shared v1 library (RSC03 on v1) ([#538](https://github.com/audiagentic/bigcherry/issues/538)) ([7f26008](https://github.com/audiagentic/bigcherry/commit/7f26008b9a2f6432f9d80304c3b5f6319d39a315))
* **radiance:** the engine's decode-sized activation staging joins the publish kernel (R3_MOE_HELPER_FUSESTAGE, default on) (RAD06) ([#285](https://github.com/audiagentic/bigcherry/issues/285)) ([8e6605d](https://github.com/audiagentic/bigcherry/commit/8e6605daccc4b4acbc54f4475a87a6c4d16cc718))
* **radiance:** validate FP8 down-product against hc_read's real packed weight layout (RAD17) ([#368](https://github.com/audiagentic/bigcherry/issues/368)) ([ca85896](https://github.com/audiagentic/bigcherry/commit/ca85896bcb07d46068a9bb5f39d22b7f71daf3b8))
* **radiance:** validate GPT's wave-per-128-group FP8 quantisation pre-pass (RAD17) ([#367](https://github.com/audiagentic/bigcherry/issues/367)) ([aa49543](https://github.com/audiagentic/bigcherry/commit/aa4954342f3f9def319ecb5af284c21f03c761ac))
* **radiance:** validate the five libr3 helper settings with run evidence (RAD06) ([#288](https://github.com/audiagentic/bigcherry/issues/288)) ([6e27103](https://github.com/audiagentic/bigcherry/commit/6e27103f88b7c5a4787805c474d04b357793f37b))
* **radiance:** W4 packed-f16 decode GEMV, parked as evaluated (RAD14) ([#345](https://github.com/audiagentic/bigcherry/issues/345)) ([805062f](https://github.com/audiagentic/bigcherry/commit/805062f20778089a34a7d32a5a449f78da61de52))
* **radiance:** wide FP8 decode GEMV (default for FP8), few-rows FP8 kernel, multi-row GEMV (evaluated), kernel-time replay tooling (RAD14) ([#337](https://github.com/audiagentic/bigcherry/issues/337)) ([452fd35](https://github.com/audiagentic/bigcherry/commit/452fd35766cfa0278e8fc415f3f776c42e3bf5d2))
* **release:** add pin-bump source compile phase ([b44b6f8](https://github.com/audiagentic/bigcherry/commit/b44b6f8a6bc272eb862fb67dafbfb88da48c2814))
* **release:** engine-agnostic patch-promote (MEN25) ([#390](https://github.com/audiagentic/bigcherry/issues/390)) ([2540b55](https://github.com/audiagentic/bigcherry/commit/2540b55d0e7eab4207fc2674eb2e54b719252bf9))
* **release:** native rewrite drift phase; pin-release confirm-then-run and release PR checks (MEN19) ([#271](https://github.com/audiagentic/bigcherry/issues/271)) ([12dc261](https://github.com/audiagentic/bigcherry/commit/12dc261ad6f73cba90790ec5cd34c96aad7e1e16))
* **release:** per-engine bump and audit gates; radiance component gate (MEN10 step f) ([#260](https://github.com/audiagentic/bigcherry/issues/260)) ([1ff34bb](https://github.com/audiagentic/bigcherry/commit/1ff34bbcc963151dba8e341468a5f22eb0f9a92e))
* **release:** pin-bump builds the bumped engine through its lane and smokes that build (MEN16) ([#268](https://github.com/audiagentic/bigcherry/issues/268)) ([e72e42f](https://github.com/audiagentic/bigcherry/commit/e72e42fbb8bebbc139194a1cf6a4994cf9ea7e50))
* **release:** pin-bump runs only the bumped engine's patch tests (MEN24) ([#349](https://github.com/audiagentic/bigcherry/issues/349)) ([e3de454](https://github.com/audiagentic/bigcherry/commit/e3de45416780ae76d03604c0c0b1bef498eeb06c))
* **release:** pin-bump runs the source compile check as its own phase ([5f9d880](https://github.com/audiagentic/bigcherry/commit/5f9d8804765d0a226783267128931ebd35a95993))
* **release:** radiance release line with upstream-as-major versions (MEN13) ([#252](https://github.com/audiagentic/bigcherry/issues/252)) ([51798f7](https://github.com/audiagentic/bigcherry/commit/51798f7afbcd8c33c5081ca076c748017a27438f))
* **serving:** bc-serve --write-manifest writes a v1 serving manifest from the resolver (RSC08 S2a) ([#578](https://github.com/audiagentic/bigcherry/issues/578)) ([a8e9554](https://github.com/audiagentic/bigcherry/commit/a8e9554d8f46948ba3a1574925cb060e66e25fcc))
* **serving:** RFC 8785 JCS for the v1 manifest subset; plan item RSC08 ([#530](https://github.com/audiagentic/bigcherry/issues/530)) ([d588513](https://github.com/audiagentic/bigcherry/commit/d58851307ef8378b716414dffcd60de604af1023))
* **serving:** schema defaults with verified sources and the section 7 provenance coverage rule (RSC08 S1c) ([#548](https://github.com/audiagentic/bigcherry/issues/548)) ([6aeeeed](https://github.com/audiagentic/bigcherry/commit/6aeeeed8b13342b00d3ed7c1eb4f0d57a2b0fa76))
* **serving:** the shared strict v1 reader: structured errors, generated codec, semantics, admission check ([#536](https://github.com/audiagentic/bigcherry/issues/536)) ([925950d](https://github.com/audiagentic/bigcherry/commit/925950d7083320cb75fa40212323a6492709dd57))
* **serving:** v1 contract field registry and generator (schema, typed model, field reference, fingerprint) ([#535](https://github.com/audiagentic/bigcherry/issues/535)) ([6a3d114](https://github.com/audiagentic/bigcherry/commit/6a3d11433005169b4064f62dc239e4228f5b2eb0))
* **serving:** verify_owner_map checks owned sets and exact device bytes against the planner (RSC08 S1a) ([#547](https://github.com/audiagentic/bigcherry/issues/547)) ([1a3785d](https://github.com/audiagentic/bigcherry/commit/1a3785dfd954e7f7eccaa419a1324620c21a2d79))
* **tooling:** bigcherry gpu-power status|apply|restore (PWR04 slice 2) ([#440](https://github.com/audiagentic/bigcherry/issues/440)) ([43887ec](https://github.com/audiagentic/bigcherry/commit/43887ecbf4933c5c68792c40b76705d3381f58a9))
* **tooling:** bigcherry host-check - warn when RAPL is unreadable or a card's configured power settings are not applied (PWR04) ([#441](https://github.com/audiagentic/bigcherry/issues/441)) ([08db0b4](https://github.com/audiagentic/bigcherry/commit/08db0b40735fdff86e29012c0b34c4a23354f6fb))
* **tooling:** bigcherry remote run/status/wait for GPU-host lab runs (MEN18) ([#267](https://github.com/audiagentic/bigcherry/issues/267)) ([3d9ade7](https://github.com/audiagentic/bigcherry/commit/3d9ade724342397f0af4155ef7c60d8d784ec98b))
* **tooling:** boot unit for per-card GPU power settings (PWR04) ([#447](https://github.com/audiagentic/bigcherry/issues/447)) ([5986d81](https://github.com/audiagentic/bigcherry/commit/5986d8172758c5d149a03007e6f7f457b2d4e3dc))
* **tooling:** bump keeps release pins in step automatically (BPB02) ([#452](https://github.com/audiagentic/bigcherry/issues/452)) ([e68a1c8](https://github.com/audiagentic/bigcherry/commit/e68a1c85f4d81ab8150fa80ab569e33d8c3339e3))
* **tooling:** GPU power settings pick a backend by the card's kernel driver; cards without one are refused ([#442](https://github.com/audiagentic/bigcherry/issues/442)) ([8ea722a](https://github.com/audiagentic/bigcherry/commit/8ea722a7b37c03accfb3c048e8d2cc890f14d6b0))
* **tooling:** gpu-power applied by the launch, granted once with sudo (PWR04) ([#453](https://github.com/audiagentic/bigcherry/issues/453)) ([bc6ecb8](https://github.com/audiagentic/bigcherry/commit/bc6ecb87a998d52044d93a9f83c65a3920667dff))
* **tooling:** gpu-power blocked messages as numbered steps per case ([#455](https://github.com/audiagentic/bigcherry/issues/455)) ([a67ddbf](https://github.com/audiagentic/bigcherry/commit/a67ddbf5f77ab788290cd229934ca2a22c280977))
* **tooling:** gpu-power container setup without bigcherry on the host ([#457](https://github.com/audiagentic/bigcherry/issues/457)) ([c5c9fdc](https://github.com/audiagentic/bigcherry/commit/c5c9fdcdb129f51877c788d7c60851de853b3a00))
* **tooling:** gpu-power restores what ran before, sets up from scratch, self-contained (PWR04) ([#456](https://github.com/audiagentic/bigcherry/issues/456)) ([de4b7e7](https://github.com/audiagentic/bigcherry/commit/de4b7e75efd7029f0edb731e17c8eee79d0cb45f))
* **tooling:** gpu-power: own group bigcherry-gpu, and container read-only /sys advice (PWR04) ([#454](https://github.com/audiagentic/bigcherry/issues/454)) ([87117af](https://github.com/audiagentic/bigcherry/commit/87117af42e628ed2d510fe914b600a1c7b947bdf))
* **tooling:** lab worktree lint and slice status/prune --lab (WSL07) ([#429](https://github.com/audiagentic/bigcherry/issues/429)) ([8c4460c](https://github.com/audiagentic/bigcherry/commit/8c4460c9ebc643fbb09746354b3ea93dc236529e))
* **tooling:** per-device power settings in the environment config (PWR04 slice 1) ([#439](https://github.com/audiagentic/bigcherry/issues/439)) ([1d941a9](https://github.com/audiagentic/bigcherry/commit/1d941a9dd0d65e8745c87137f127a3c3748e6935))
* **tools:** Flash-Next ENGINE_WRAP, short probe and host-link test (RAD23) ([#487](https://github.com/audiagentic/bigcherry/issues/487)) ([d248d8e](https://github.com/audiagentic/bigcherry/commit/d248d8e42004761c97ee1d8a6b29696eb7f26293))
* **tools:** Flash-Next launcher can add the RX 6900 XT as a third helper (--tail-mib) (RAD23) ([#470](https://github.com/audiagentic/bigcherry/issues/470)) ([b762695](https://github.com/audiagentic/bigcherry/commit/b76269595304ba5ddfbdddc51eb8b0321cb63e6a))
* **tools:** Flash-Next launcher defaults to 8 sessions at the trained 256K context (RAD23) ([#483](https://github.com/audiagentic/bigcherry/issues/483)) ([73a02f9](https://github.com/audiagentic/bigcherry/commit/73a02f9eddccf96e8a0c5647fd4642abed8e7184))
* **tools:** Flash-Next launcher funds a 12 GiB host KV tier from a right-sized 9 GiB host pool ([#512](https://github.com/audiagentic/bigcherry/issues/512)) ([3933aad](https://github.com/audiagentic/bigcherry/commit/3933aad618871abbd96598df37c3a07e7ae3e304))
* **tools:** Flash-Next launcher runs multi-stream decode on the packed-f16 W4 GEMV ([#505](https://github.com/audiagentic/bigcherry/issues/505)) ([95ff28f](https://github.com/audiagentic/bigcherry/commit/95ff28fa8afd60fd601671e7921f09e7f95620f5))
* **tools:** flashnext-4gpu with the combine is the launcher default (RAD23) ([#488](https://github.com/audiagentic/bigcherry/issues/488)) ([5076e78](https://github.com/audiagentic/bigcherry/commit/5076e78d2393c16c5d1271aa5cbfd14ef7c9dd64))
* **tools:** heat-drift.py, the RHP05 stage A check of the active heat file (RAD23) ([#489](https://github.com/audiagentic/bigcherry/issues/489)) ([a42a296](https://github.com/audiagentic/bigcherry/commit/a42a29629411a214581ac52aa92a9cb06a312807))
* **tools:** heat-watch.py, RHP05 stage A recommend-only drift watch ([#501](https://github.com/audiagentic/bigcherry/issues/501)) ([c13c693](https://github.com/audiagentic/bigcherry/commit/c13c693c01b0ef416c4defeb36110b2059b53cb3))
* **tools:** launcher --record-heat and heat-invert.py (RHP08, RHP10) ([#477](https://github.com/audiagentic/bigcherry/issues/477)) ([1eb5a6e](https://github.com/audiagentic/bigcherry/commit/1eb5a6e64abb8e3d69a01a0431f6786ed5e23140))
* **tools:** the Flash-Next launcher records heat by default and prints the drift-watch command (RHP02, RHP05) ([#502](https://github.com/audiagentic/bigcherry/issues/502)) ([9894d93](https://github.com/audiagentic/bigcherry/commit/9894d93afb97819a72e7ce9cb6336410340f92bf))
* **workspace:** inspect build and source caches without GC (WSL04) ([#409](https://github.com/audiagentic/bigcherry/issues/409)) ([42c0442](https://github.com/audiagentic/bigcherry/commit/42c0442eb9519313ea4bb6b1892235cce791718e))
* **workspace:** inventory legacy work-root objects safely (WSL06) ([#410](https://github.com/audiagentic/bigcherry/issues/410)) ([3912060](https://github.com/audiagentic/bigcherry/commit/3912060c75650a84f9aebb2ce4fd3319176e627c))
* **workspace:** journalled quarantine, restore and purge of a harvested run's bulk (WSL03) ([#315](https://github.com/audiagentic/bigcherry/issues/315)) ([e3c06c4](https://github.com/audiagentic/bigcherry/commit/e3c06c45782f347d62e331ae5c7f4026572c87ba))
* **workspace:** managed work root - run manifest, run creation, read-only inventory, bigcherry run and clean (WSL01, WSL03 skeleton) ([#311](https://github.com/audiagentic/bigcherry/issues/311)) ([96ad316](https://github.com/audiagentic/bigcherry/commit/96ad3167d28aefbed300e1a3c1521c03716579d5))
* **workspace:** run harvest writes a hash-verified evidence bundle (WSL02) ([#312](https://github.com/audiagentic/bigcherry/issues/312)) ([6bc7865](https://github.com/audiagentic/bigcherry/commit/6bc7865680a81423ccd50a3228989b9bde5f89dd))


### Fixes

* **bench:** stream-bench accounting - token counts from usage, text events apart, finite-batch definition (RAD10) ([#302](https://github.com/audiagentic/bigcherry/issues/302)) ([ebc9d2d](https://github.com/audiagentic/bigcherry/commit/ebc9d2d8e89a9ed6941c3c034c910f5dde7e00f2))
* **build:** give each component its own ccache launcher, not just the engine ([#380](https://github.com/audiagentic/bigcherry/issues/380)) ([86163f8](https://github.com/audiagentic/bigcherry/commit/86163f8930c90991db8a40a6e587131de7888d80))
* **build:** radiance builds pin hipcc, HIP compiler and ROCm runtime to the selected SDK (RCS11) ([#524](https://github.com/audiagentic/bigcherry/issues/524)) ([13ab746](https://github.com/audiagentic/bigcherry/commit/13ab746cae6b5d4c4957e2d05c240587400379f2))
* **campaign:** a sidecar that exits on its own after a clean run is not a failure ([#568](https://github.com/audiagentic/bigcherry/issues/568)) ([b0e1057](https://github.com/audiagentic/bigcherry/commit/b0e1057d288254ffe25fcbcd7883f2783b479bc6))
* **campaign:** create the KL reference directory before recording (MEN11) ([#256](https://github.com/audiagentic/bigcherry/issues/256)) ([49adf5d](https://github.com/audiagentic/bigcherry/commit/49adf5d5906d2b558ddd88e0415bc2f9be4d20f9))
* **campaign:** radiance helpers select their card with ROCR and HIP index 0 (MEN11) ([#248](https://github.com/audiagentic/bigcherry/issues/248)) ([6cadf9f](https://github.com/audiagentic/bigcherry/commit/6cadf9f2e701d85cee1db37980eac74c359179f1))
* **ci:** run radiance-build on push to main so PR caches actually share ([#377](https://github.com/audiagentic/bigcherry/issues/377)) ([fb2a871](https://github.com/audiagentic/bigcherry/commit/fb2a871ed0882f879f5f558f1550a351cb243e22))
* **ci:** wire ccache into the radiance build to stop full from-scratch recompiles ([#374](https://github.com/audiagentic/bigcherry/issues/374)) ([e7f579c](https://github.com/audiagentic/bigcherry/commit/e7f579cff31fb2775836fd1a50065a26b01d9bd2))
* **core:** take the work root from the host file when the environment has none ([#383](https://github.com/audiagentic/bigcherry/issues/383)) ([bf5ab6e](https://github.com/audiagentic/bigcherry/commit/bf5ab6e75e4cf8a5303922926bb392afc9ca5edd))
* **hardware:** AMD discovery reads amd-smi 26.x gpu_data and exact sysfs VRAM ([#518](https://github.com/audiagentic/bigcherry/issues/518)) ([9257168](https://github.com/audiagentic/bigcherry/commit/92571683ce3c4fec32ea6f5c79b7d55ed149dfea))
* host values expand ${VAR} (work-root ${HOME}/...), scans run once per PR ([#405](https://github.com/audiagentic/bigcherry/issues/405)) ([09195ae](https://github.com/audiagentic/bigcherry/commit/09195aecad896c7da69d84e11ee9580db2021c60))
* **lab:** helper-kld control arm gets the 44 GiB host pool the R9700-alone layout needs (MEN11) ([#257](https://github.com/audiagentic/bigcherry/issues/257)) ([68d63b0](https://github.com/audiagentic/bigcherry/commit/68d63b069e5cef7a347281a9963db68be130d18f))
* **lab:** helper-kld thresholds from the first two measured lab runs (MEN11) ([#259](https://github.com/audiagentic/bigcherry/issues/259)) ([0419ff1](https://github.com/audiagentic/bigcherry/commit/0419ff13ef10e3facee93b3fc1fe48aaa4c62412))
* **lab:** quick-ab compares a new binary's second run, not its first ([6084541](https://github.com/audiagentic/bigcherry/commit/60845417794e252417ef428f2cd2fa7ca462fa04))
* **lab:** quick-ab runs four arms so a new binary's first decode is not what gets compared; PA44 - 1281 slice record ([7c0f903](https://github.com/audiagentic/bigcherry/commit/7c0f903706a06392c7d681e24427016286307114))
* **patch:** 1281 - the MoE kernel edit's guard keeps its indent (without it the guard matches upstream's own line and the edit is skipped); test checks the barrier after the clear ([8572439](https://github.com/audiagentic/bigcherry/commit/8572439f7fc5400882a11955bfee4e327c7679b7))
* **patch:** 1361 puts its second-arena helpers in 1340's own header (the anchor moved there with the 1340 refactor) ([00e7bd1](https://github.com/audiagentic/bigcherry/commit/00e7bd1a3f1b96bba389b2c9c57214d6f13b54c1))
* **patch:** 1362 pads the pooled-block count too, so Qwen4Exp prompt batches reuse the graph ([06249b4](https://github.com/audiagentic/bigcherry/commit/06249b4d525d12bf584d0cd4436056232b2e5412))
* **patch:** reject 1362_kv_pad_prompt - graph reuse brings no prefill gain ([4cf9483](https://github.com/audiagentic/bigcherry/commit/4cf9483f77ecda11ddc2af3d1c890a33cac9dbba))
* **patch:** reject 1362_kv_pad_prompt - the graph is reused but prefill does not get faster (QFP53) ([546bd9b](https://github.com/audiagentic/bigcherry/commit/546bd9b7d8e6769267cb10f80f0b3d4040d49bae))
* **patch:** shelve 1356 and 1360 as evaluated (owner decision); QFP41 and QFP49 closed; main ruleset prepared for the owner (PA46) ([a0b7396](https://github.com/audiagentic/bigcherry/commit/a0b739620d730ee811bd095ed8fb7aa0aadc2a0b))
* **patch:** shelve 1356 and 1360 as evaluated; close QFP41 and QFP49 ([ebcaa69](https://github.com/audiagentic/bigcherry/commit/ebcaa69c988a7bf46b563d8d146994c0dc9c93ce))
* **patch:** stabilize Qwen4Exp k-pool padding in 1362 ([de88443](https://github.com/audiagentic/bigcherry/commit/de88443dfcb3f3532c45c5146d88a0ca9e3ea951))
* **radiance:** combined-step chunk poller always polls (RAD42 no-tail hang) ([#520](https://github.com/audiagentic/bigcherry/issues/520)) ([f4dd959](https://github.com/audiagentic/bigcherry/commit/f4dd959b2fd3edd4975f7a97364babc50a883b89))
* **radiance:** core tests after the 1.8.0 bump and a CI job that runs them ([#305](https://github.com/audiagentic/bigcherry/issues/305)) ([ec24034](https://github.com/audiagentic/bigcherry/commit/ec24034a271e6e7278d8ea600de3bce641e8a4fd))
* **radiance:** Flash-Next launcher starts the XTX helpers and uses the current best settings ([#392](https://github.com/audiagentic/bigcherry/issues/392)) ([450b36f](https://github.com/audiagentic/bigcherry/commit/450b36f4db50e0154fa6da9a53c8242846e4f95e))
* **radiance:** libr3 unit wrappers include their unit by relative path ([#385](https://github.com/audiagentic/bigcherry/issues/385)) ([1e0efac](https://github.com/audiagentic/bigcherry/commit/1e0efac6cf049a13325a14fdb42b18b1e562c224))
* **radiance:** placement simulator review fixes (RHP07-01) ([#294](https://github.com/audiagentic/bigcherry/issues/294)) ([5aeba08](https://github.com/audiagentic/bigcherry/commit/5aeba0869767750ffb0df2ba0cd82e08a9c3f50a))
* **radiance:** qwen27b-2xtx runs with 3 speculative draft tokens (single-stream decode 48 -&gt; 96.7 tok/s) ([#590](https://github.com/audiagentic/bigcherry/issues/590)) ([5c88f5c](https://github.com/audiagentic/bigcherry/commit/5c88f5c38ca8cdf27836c81c25ba17924b7264b6))
* **radiance:** r3_moe_worker replay runs the production bootstrap again ([#565](https://github.com/audiagentic/bigcherry/issues/565)) ([6eb31a7](https://github.com/audiagentic/bigcherry/commit/6eb31a708b92cefae2f90afed727e0638a449a50))
* **radiance:** rad0009 reads bc-serve's nested values.engine; test against a real bc-serve manifest and radiance's real option table ([#526](https://github.com/audiagentic/bigcherry/issues/526)) ([a234b35](https://github.com/audiagentic/bigcherry/commit/a234b35f91f28ee343faaf1e8ceb7f47122f3f70))
* **radiance:** RAD46 accurate gfx1030 rows kernels, selected per stage by request rows ([#577](https://github.com/audiagentic/bigcherry/issues/577)) ([5bcb0b0](https://github.com/audiagentic/bigcherry/commit/5bcb0b006cd0866500f733ba7691a3a810d52133))
* **radiance:** the burst-trace activation marker matches the line the plugin prints; THA34 records the HI17 test red on main ([#539](https://github.com/audiagentic/bigcherry/issues/539)) ([7c83538](https://github.com/audiagentic/bigcherry/commit/7c835384d3202660dcc3a6510b36b3217acf28d9))
* **release:** .release-please-manifest.json had committed conflict markers ([#363](https://github.com/audiagentic/bigcherry/issues/363)) ([3bf24e3](https://github.com/audiagentic/bigcherry/commit/3bf24e3d3361017c8e35a9ea1b00af5492a89301))
* **release:** add resumable release phases ([1c8239b](https://github.com/audiagentic/bigcherry/commit/1c8239ba72d36d6a630ff3957e7ce580e42008a9))
* **release:** bc-platform back to 1.0.1; its next release is 1.1.0 ([b4a4b77](https://github.com/audiagentic/bigcherry/commit/b4a4b77ed349da996cc7d17a328f371683f450c9))
* **release:** concise reviewed Radiance release notes ([#566](https://github.com/audiagentic/bigcherry/issues/566)) ([ccd5665](https://github.com/audiagentic/bigcherry/commit/ccd5665827c64f48518840593b7a4ae5ea8fa4f1))
* **release:** engine release notes no longer count for the platform release line ([#564](https://github.com/audiagentic/bigcherry/issues/564)) ([23f8d1e](https://github.com/audiagentic/bigcherry/commit/23f8d1e59942f2f1148210234a7a3bd5fdf33b8e))
* **release:** expose unpin and PR release flow ([ef3755e](https://github.com/audiagentic/bigcherry/commit/ef3755e3fb965eb333d8090dce5e46f5106b24bc))
* **release:** make publish dry-run concrete ([ea4e225](https://github.com/audiagentic/bigcherry/commit/ea4e2259666e3c37e2ea9e220c5ada045e80a769))
* **release:** merge release record through a pull request ([0481b19](https://github.com/audiagentic/bigcherry/commit/0481b19e89fcec57877e6aeb715441b5a4985cb8))
* **release:** patch-promote moves a promoted patch out of every other patch set (MEN26) ([#543](https://github.com/audiagentic/bigcherry/issues/543)) ([499db6e](https://github.com/audiagentic/bigcherry/commit/499db6e389ea38647a4d1dedf3cac313ed6cfe54))
* **release:** pin version in release-please config ([92e9cd9](https://github.com/audiagentic/bigcherry/commit/92e9cd9059a9e6a157b5faa3272d9fa6d97e7796))
* **release:** pin-release goes through pull requests and sets the package version in the release-please config ([53c8607](https://github.com/audiagentic/bigcherry/commit/53c86078f69c1579dd9ca246a3af1eddacad718e))
* **release:** pin-release names a release version from a squash-merged notes commit ([#297](https://github.com/audiagentic/bigcherry/issues/297)) ([5be0df2](https://github.com/audiagentic/bigcherry/commit/5be0df27be9a51da81f3449b8e7bdefc65575451))
* **release:** radiance compile gate configures with the build platform's compilers and options ([#262](https://github.com/audiagentic/bigcherry/issues/262)) ([7ec642d](https://github.com/audiagentic/bigcherry/commit/7ec642d7ba6e8fee0f437701db067381dee6c3e8))
* **release:** reject conflicting release-as before writing notes ([9b75926](https://github.com/audiagentic/bigcherry/commit/9b75926710838425b6f9485a732c4c9b1559c580))
* **release:** release-please acts as a real account, not GITHUB_TOKEN ([#362](https://github.com/audiagentic/bigcherry/issues/362)) ([5a1723d](https://github.com/audiagentic/bigcherry/commit/5a1723d8d6aa815882f497a012aefb844d939c1d))
* **release:** report post-tag notes as complete in dry run ([1d031af](https://github.com/audiagentic/bigcherry/commit/1d031af4845231d99ac1773c723a85003514d191))
* **release:** restore bc-platform to 1.0.1 after the mistaken 11552.0.0 release ([bb1884c](https://github.com/audiagentic/bigcherry/commit/bb1884c950008b5223b7bffb54c4eb2603c10866))
* **release:** resume cleanly after release tag exists ([bf74c98](https://github.com/audiagentic/bigcherry/commit/bf74c989fd90fbf7e2ee5d6bfecb8ed3aad6e7d3))
* **release:** select previous tag across release families ([a28aa21](https://github.com/audiagentic/bigcherry/commit/a28aa217ba3b12d27ddb89220015e5ab0589a1c3))
* **release:** show current manifest on completed dry runs ([7bfe6d5](https://github.com/audiagentic/bigcherry/commit/7bfe6d526fcec8423cedadda610c478251718288))
* **release:** tag pins in pin-release, one release-record key, pull fetches tags ([#264](https://github.com/audiagentic/bigcherry/issues/264)) ([d0a4447](https://github.com/audiagentic/bigcherry/commit/d0a44476be8c29ee8fe2a25cf29428b7fef6f612))
* **release:** the readable-tag push runs as the release PAT, which needs the Workflows permission ([#563](https://github.com/audiagentic/bigcherry/issues/563)) ([9fce55e](https://github.com/audiagentic/bigcherry/commit/9fce55ebc3e0fbda81195caf173c0993fa1ea613))
* **release:** validate release PR and clear release-as ([58f7b8a](https://github.com/audiagentic/bigcherry/commit/58f7b8a3c5d59245fde3926cece8b76606fdce2c))
* **security:** redact benchmark credentials and track repository privacy cleanup ([#544](https://github.com/audiagentic/bigcherry/issues/544)) ([ce04557](https://github.com/audiagentic/bigcherry/commit/ce045574b7b1877937830ed94c03765792da0a66))
* **tests:** repair the eight standing test failures on main ([#395](https://github.com/audiagentic/bigcherry/issues/395)) ([59faa2f](https://github.com/audiagentic/bigcherry/commit/59faa2fc9b0dfb3b4563402e623d47be971b7ab9))
* **tooling:** gpu-power setup messages: spaced steps, pasteable blocks ([#459](https://github.com/audiagentic/bigcherry/issues/459)) ([e52d50b](https://github.com/audiagentic/bigcherry/commit/e52d50ba1a73ccf7770c431f5552701746418956))
* restore the work-root paths a redaction had renamed to the cache-root variable ([#399](https://github.com/audiagentic/bigcherry/issues/399)) ([3d9430a](https://github.com/audiagentic/bigcherry/commit/3d9430a2881bc73cbdc0cfbcd5541f14d64d0bf5))
* **workspace:** block cleanup when any checkout has a lease (WSL03) ([#408](https://github.com/audiagentic/bigcherry/issues/408)) ([36fd60a](https://github.com/audiagentic/bigcherry/commit/36fd60add833696b8346ec99fc84a0a44a5eb1d1))
* **workspace:** harden harvest per GPT review; RCS10 stream-derived settings note ([#313](https://github.com/audiagentic/bigcherry/issues/313)) ([526a7d4](https://github.com/audiagentic/bigcherry/commit/526a7d4c25732fd3497dfc3f401231608bee1a04))


### Performance

* **ci:** trial bounded HIP compile concurrency with Ninja timings ([#586](https://github.com/audiagentic/bigcherry/issues/586)) ([83447c3](https://github.com/audiagentic/bigcherry/commit/83447c34567651fa6af90f61dbe3fdaa8a77530b))
* **radiance:** few-rows lane kernel 2x on gfx1030 (RAD23) ([#479](https://github.com/audiagentic/bigcherry/issues/479)) ([8455035](https://github.com/audiagentic/bigcherry/commit/84550350e14be4934345061ae127d99b5090903a))
* **radiance:** few-rows lane kernel without the accumulator spill, up to 2x on large 6900 XT tails (RAD23) ([#481](https://github.com/audiagentic/bigcherry/issues/481)) ([25a0cd9](https://github.com/audiagentic/bigcherry/commit/25a0cd9e33ae02d034158bd64d65af380fbe7e53))
* **radiance:** Flash-Next helpers own the hottest experts (RAD06) ([#272](https://github.com/audiagentic/bigcherry/issues/272)) ([45dd5ac](https://github.com/audiagentic/bigcherry/commit/45dd5acf89264402ac1f2efd6073a6a19056ce9d))
* **radiance:** fuse injection into the quant kernel, the real bottleneck (RAD17) ([#378](https://github.com/audiagentic/bigcherry/issues/378)) ([644bf0e](https://github.com/audiagentic/bigcherry/commit/644bf0e893597536598d4b7dc5b1356854c5648b))
* **radiance:** gated quantise between gate/up and down bounded by the route count (R3_MOE_WORKER_LANE_QUANT, RAD23) ([#484](https://github.com/audiagentic/bigcherry/issues/484)) ([1d74b12](https://github.com/audiagentic/bigcherry/commit/1d74b1206a1f7cdb8c888858b803d3d40cbc2a4b))
* **radiance:** gfx1030 few-rows MoE kernel on fp16 dot2, double-buffered (RAD23) ([#486](https://github.com/audiagentic/bigcherry/issues/486)) ([5798ec8](https://github.com/audiagentic/bigcherry/commit/5798ec88b0924c6caa80236cae88c76e526768ec))
* **radiance:** helper MoE GEMM tuning on gfx1100 - narrow down tile, request histogram, form/tile pins (RAD06) ([#273](https://github.com/audiagentic/bigcherry/issues/273)) ([c1352b9](https://github.com/audiagentic/bigcherry/commit/c1352b942383e0a85570c66441b890255a513520))
* **radiance:** RAD47 - gfx1100 MXFP4 GEMM rows: tiled from 17 rows up (27B 4 streams +27%, 6 streams +37%) ([#596](https://github.com/audiagentic/bigcherry/issues/596)) ([6195907](https://github.com/audiagentic/bigcherry/commit/6195907289734ba42f78a18e3962827b174e67ad))
* **radiance:** the helper-row combine goes chunk by chunk (protocol v22, RAD42) ([#497](https://github.com/audiagentic/bigcherry/issues/497)) ([d6647a3](https://github.com/audiagentic/bigcherry/commit/d6647a35f5074bf3bf8ca38edf039de907562590))
* **radiance:** the route histogram lives in pinned host memory, counted in LDS (RHP02) ([#500](https://github.com/audiagentic/bigcherry/issues/500)) ([7a40358](https://github.com/audiagentic/bigcherry/commit/7a4035827811bc97c0a74494fcc9719426892f06))


### Reverts

* RAD01 PR 2 engine-explicit registry ([#213](https://github.com/audiagentic/bigcherry/issues/213)) - offline-patch CI failed; merged in error ([#214](https://github.com/audiagentic/bigcherry/issues/214)) ([82434de](https://github.com/audiagentic/bigcherry/commit/82434de3b99bfb1979520fcfc9b722e391a6b8ef))

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
* **lab:** helper-kld thresholds from the first two dev runs (MEN11) ([#259](https://github.com/audiagentic/bigcherry/issues/259)) ([5aedfc1](https://github.com/audiagentic/bigcherry/commit/5aedfc1b7f42c5bff4dc9510e5084f8efa43c53c))
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
