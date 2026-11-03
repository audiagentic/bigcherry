# bc-radiance-1140.0.0

BigCherry on radiance `v1.14.0` (`4a61d7296bc5`), BigCherry commit `9b3eb87701cc`.
First tagged release.

## Bump record

pin-bump lane build: radiance:stock:linux-radiance build_plan_id=459d86718e56e15a3aea8860aa5a9ebc binary_sha256=b9233dc8cd1987a4db4f9f736061396717559bc1fe8d8e5a2dd2cab94194fcb0

release 1140.0.0 2026-11-03
upstream: radiance v1.14.0 (revision 4a61d7296bc5b4fec5932949dac4a2fadf9b61ca)

completion gate (pin-status --complete --all-remotes):
pin-status: local
  pin           v1.14.0 -> 4a61d7296bc5
  vendor HEAD   4a61d7296bc5 (tag v1.14.0)  [1 modified, 0 untracked]
  releases      0504396140d1 audited, 0adcc3bb5710 broken, 1e968f6e1d86 patched, 22dc605c4ead patched, 23b0202a189c patched, 2578138397d7 generated, 28ff0958291c patched, 370cb12e8bb7 patched, 434ddbbc0e30 patched, 4a61d7296bc5 patched, 50182a53fa2c generated, 6645337f695a audited, 685ec5bd858a patched, 6ea215d171fd patched, 7a7707a95559 patched, 91f6a6cf3613 patched, abc123 validated, b1ff4ca23630 patched, b9acf138a1e2 patched, c061df19838f broken, c841aeeb8bb2 patched, d77dd0806dc2 patched, d7bd3bfcad3e patched, d89651a7b205 patched
  bigcherry     f1d49c78c747
  marker        6645337f695a -> 4a61d7296bc5 (v1.14.0) [committed-clean]
  VERDICT       consistent  (content-unverified: 1 tracked file(s) modified (expected after patch application; verified by audit, not here))

COMPLETION  PASS

hardware evidence:
radiance v1.14.0 release evidence (bigcherry main after #538, #542, #546).
Build: radiance-lab:stock:linux-radiance lane on origin/main (production patch sets serving-core [rad0008, rad0002], host-compat, serving-manifest; moe-remote, serving, libr3, bc-gfx12 components).
Serving A/B vs the previous main (runs/rhp11-c2-ab7, Flash-Next R9700 engine + two 7900 XTX + 6900 XT tail, 9 interleaved arms): identical text; 1-stream decode +0.14% (24.47 vs 24.43 ms/step), 6-stream decode -0.46%; prefill TTFT 396 vs 395 ms (1 stream), 5.62 vs 5.65 s (6 streams); 28 placement epoch cycles per cycling arm with no hang.
A3 gate (#538, runs/a3-gate): identity vs main 8/8 arms text-identical, 1-stream 24.40 vs 24.41 ms/step; plan gate PASS (cold measure then serve with correct first completion, cached restart admitted, 1-byte plan drift refused before any pool); start canary passes on the card.
Quality (runs/men26-kld, KL mode vs a freshly recorded engine-local reference, 8240 positions, deterministic): two-XTX helper offload passes every helper-kld.json limit (p99.9 +0.41/0.5); the 6900 XT tail layout fails KLD p99.9 (+1.58), tracked as RAD46.
Known open items: RAD46 (RDNA2 tail-helper tail quality); rad0008 evidence recorded at 1.9.0, current-pin revalidation outstanding (MEN26); the one unexplained cold-start '!!!' run in the first A3 gate (1 of 10 cold launches), now guarded by the start canary (RSC08).

## New and changed

- Each engine now declares how its server is run, so the lab server runner can launch llama.cpp or radiance from the same code. (MEN03)
- Radiance is now built, patched, configured and A/B-measured through the same BigCherry patch sets, recipes, profiles and campaign harness as llama.cpp. (RAD01)
- The R9700 expert helper for Flash-Next on two XTX was shown to be numerically sound and raises decode speed by about 78%. (RAD03)
- Flash-Next can now run with the R9700 as the main engine and both RX 7900 XTX cards computing experts; prefill reaches ~1,880 tok/s, the fastest Flash-Next prefill so far on dev. (RAD02, RAD03)
- Flash-Next prefill on dev reaches about 2,640 tok/s with the R9700 running the model and both XTX cards computing the cold experts, more than double the original two-XTX setup. (RAD02)
- Flash-Next decode on dev reaches about 108 tok/s with the R9700 running the model and both XTX cards computing experts, four times the original two-XTX setup. (RAD05, RAD06)
- Added an experimental prefill kernel for the radiance expert helper; it measured slower and stays off. (RAD06)
- Radiance experiments now run through the standard benchmark harness, including the XTX expert helpers. (MEN11)
- Radiance quality checks report the same metrics as the model's published reference. (RAD08)
- Radiance has its own release line, versioned from the upstream radiance version. (MEN13)
- Radiance moved to upstream 1.7.0 with the same speed and quality, and its first release, bc-radiance 170.0.0, is out. (MEN12)
- Flash-Next prefill on the R9700 + two-XTX layout is about 48% faster (2,130 to 3,150 tok/s) with identical output and unchanged decode. (RAD06)
- Flash-Next helper kernels are a further 7-8% faster with identical output. (RAD06)
- Flash-Next prefill on the R9700 + two-XTX layout is another ~5% faster (about 3,310 tok/s) with identical output. (RAD06)
- Internal experiment: balancing expert ownership between the two helper GPUs per layer is available but off by default (no prefill gain). (RAD06)
- Flash-Next decode is about 5-8% faster (about 143-150 tok/s) with identical output, and the helper critical path can now be measured without a profiler. (RAD06)
- Flash-Next decode is about 1% faster again (about 151 tok/s at long generations), with identical output. (RAD06)
- Radiance is now on upstream 1.8.0 with the same build, patch and release workflow as llama.cpp. (MEN12, MEN22)
- The faster prefill and decode paths for the two-XTX helper layout are now validated settings with recorded evidence. (RAD06)
- Radiance 1.8.0 is released as bc-radiance-180.0.0 after passing hardware acceptance. (MEN12, MEN13)
- Found that Qwen3.6-35B-A3B on a single R9700 loses speculative-decode acceptance at 4+ concurrent streams unless the prefix cache is turned off, and added the probes and configs used to show it. (RAD14)
- Qwen3.6-35B-A3B served with the two XTX workers decodes much faster: about 2.7x less kernel time per step, +60-75% engine decode at 4 and 6 streams. (RAD14, RCS02, RCS03, RCS04)
- Radiance updated to 1.9.0 (new KV cache formats, int8 trunk GEMMs, GLM-5.3 and Gemma 3 support) with our gfx1100 kernels ported; Flash-Next and Qwen3.6-35B decode speeds unchanged. (MEN24)
- Added an opt-in faster MoE decode kernel for the R9700 (BC_GFX12_MOE_GEMV=1) and a decode-path quality check. (RAD15)
- Validated a faster FP8 matrix kernel for Flash-Next prefill works correctly on the R9700, ahead of wiring it in; added a one-command Flash-Next launcher. (RAD15)
- Measured whether the idle 6900 XT could help with decode: it can't, for the reasons we now have data on. (RAD16)
- Closed the remaining technical blocker on the Flash-Next prefill speedup's activation path, validated against real model data. (RAD15)
- Turned on the faster MoE decode kernel for the R9700 by default. (RAD15)
- Found and fixed the real bottleneck behind last round's confusing results: the FP8 kernel's grid was too small to fill the GPU. Fixing that alone cut the time by nearly a third. Still slower than what it would replace, so it stays off, but the gap is closing fast with each real measurement. (RAD17)
- Found the real bottleneck (too many small kernel launches, not slow math) by comparing directly against the production code path, fixed it, and proved the fix produces byte-identical results before trusting it. Nearly 37% faster this session, though still not faster than what it would replace. (RAD17)
- One more small real speedup from reducing kernel launches, verified against the actual production reference this time, not just the prior revision. (RAD17)
- Radiance's hyper-connection read is now about 3% faster for Qwen3.8-Flash-Next prefill (output unchanged), the first radiance patch in the production set. (RAD18, MEN25)
- bigcherry patch-promote now works for any engine, including radiance. (MEN25)
- An optional mode makes Flash-Next prefill about 5-8% faster on the helper layout with output quality within the gate; it stays off by default while one decode measurement is checked. (RAD18, RAD23)
- Flash-Next helper GPUs now idle at about 6-12 W instead of about 95 W between requests, for about 0.5% decode cost; switchable with R3_MOE_WORKER_POWER_SAVE. (RAD20)
- Flash-Next helper GPUs now idle at about 6-12 W instead of about 95 W between requests, for about 0.5% decode cost; switchable with R3_MOE_WORKER_POWER_SAVE. (RAD20)
- Flash-Next helpers use about 30 W less CPU power on CPUs with efficiency cores, with no speed change. (PWR03)
- Radiance updated to 1.13.0: the 27B prefills about 20% faster and decodes about 8% faster on two RX 7900 XTX, and Flash-Next decodes about 4% faster with slightly better output quality. Speculative decoding on the 27B is faster than before the fix but still below its earlier best; a follow-up is planned. (RAD36)
- The remote-MoE helper can now run on an RX 6900 XT (RDNA2). Nothing changes for the 7900 XTX and R9700 setup; using the 6900 XT as a third helper needs further work. (RCS07)
- Radiance updated to 1.14.0: Flash-Next decodes about 1.5% faster at the same quality, the 35B-A3B model now serves with speculative decoding (it failed on 1.13.0), and the Flash-Next launcher defaults to 1M tokens of context. (RAD37)
- The remote-MoE offload can now be laid out for up to four helper GPUs, with each helper's share of the experts configurable; two-helper behaviour is unchanged. (RAD23)
- The Flash-Next lab launcher can use the RX 6900 XT as an extra offload helper (off by default). (RAD23)
- Planning and a first offline building block for adapting expert placement to the real workload; no change in behaviour. (RHP01, RHP02, RHP03, RHP04, RHP05, RHP06, RHP07, RHP08, RHP09, RHP10, RHP11)
- Flash-Next prefill with the RX 6900 XT as a third helper is about 7% faster: the 6900 XT's MoE kernel uses its packed fp16 dot instructions. (RAD23)
- Flash-Next on four GPUs serves 8 concurrent sessions about 6% faster (2,513 tok/s): the helpers' results now stream to the engine in chunks instead of after the whole step. (RAD42)
- Multi-stream decode on Flash-Next is up to ~18% faster: decode steps of up to 32 rows now use the faster packed-f16 4-bit expert kernel on the helper GPUs. (RAD23)
- The faster packed-f16 4-bit expert kernel is now the helper GPUs' default. (RAD23)
- Groundwork for moving experts between helper GPUs while serving: the helper messaging protocol now carries versioned ownership. (RHP11)
- Long-running Flash-Next servers keep their prefill speed: finished sessions now move to a host KV tier instead of crowding experts out of GPU memory. (RSC03, RHP12)
- Groundwork for live expert placement: the state machine and slot bookkeeping that will move experts between helper GPUs safely while serving. (RHP11, RHP06, RHP04, RHP07)
- New bc-serve tool: one strict, typed serving configuration with hardware detection and an explainable manifest. (RSC01, RSC05)
- Groundwork for live expert placement: every model pass now carries the placement it was issued under, at no measurable cost. (RHP11)
- The engine can now start from a validated serving manifest instead of command-line flags. (RSC07)
- The serving component can produce the canonical form and content digest the v1 manifest format specifies. (RSC08)
- The serving manifest v1 schema, typed C++ model and field reference are now generated from one field registry, and CI fails if any generated copy drifts. (RSC08)
- bc-serve, the engine and the helpers can now share one strict reader for the v1 serving manifest, with stable error codes and the start-up memory admission check. (RSC08)
- The engine now checks, at every start and before it allocates anything, that the memory plan it is about to run still fits what was measured, and refuses with a clear reason if not. (RSC03, RSC08)
- The engine can now pause, drain and resume expert placement safely under load, the prerequisite for moving experts between GPUs while serving, at no measurable decode cost. (RHP11)
- The remote-MoE hand-off that lets Flash-Next spread experts across the helper GPUs is now a validated part of the production engine build. (RAD01, MEN26)

## Fixes

- Radiance builds now reuse the compiler cache for every component and share one work area per machine; a new up-mix kernel setting (RAD_HC_UPMIX_RT=128) makes the prefill hc_read step 1.5-3% faster on the R9700. (RAD18)
- If a Flash-Next helper process dies, the engine now fails the request within seconds instead of hanging. (RAD24, RAD20)
- Fixed a hang with many concurrent long sessions when helper rows are combined on the CPU. (RAD42)
- Radiance builds now compile and link against the selected ROCm SDK only, never whatever /opt/rocm points at, and builds against different ROCm SDKs no longer share a build directory. (RCS11)
- Fixed the engine's --manifest start-up to read the manifest format bc-serve actually produces. (RSC07)
- Burst-trace activation evidence now matches the line the plugin prints. (THA34)

## Internal changes

- Health checks and the job system now handle each engine separately. (MEN10)
- Internal cleanup of the Flash-Next expert helper; no behaviour change. (RAD20)
- Internal cleanup of the Flash-Next expert helper; no behaviour change. (RAD20)
- Internal preparation for using more than two helper GPUs with the remote-MoE offload; no change in behaviour. (RAD23)
- The RDNA2 (RX 6000) GPU kernels for the remote-MoE offload are now their own library; no change for existing setups. (RCS07, RAD23)
- Which GPUs the offload uses, and for what, is now set in configuration files instead of code; a new card type or layout is a config edit. (RAD23, RCS07)
- bc-serve and the MoE helpers now share one implementation of expert ownership, so the planner can compute the same owned sets the helpers will select. (RSC03)
