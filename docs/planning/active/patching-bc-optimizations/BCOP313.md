---
id: BCOP313
order: 313
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-19'
created-by: agent
priority: P2
work: S
---

# RNX02: vector-FA KQ WAR fence is non-HIP; retire redundant 1300 selector

**Discovery:** Upstream [llama.cpp #30573](https://github.com/ggml-org/llama.cpp/pull/30573) adds a post-V-loop `ggml_cuda_syncwarp()` to `fattn-vec.cuh` after NVIDIA racecheck reported 800,742 WAR warnings on sm_86. Pinned b11552 `common.cuh` makes that helper a no-op for HIP, so direct adoption changes no gfx1100/gfx1201 synchronization or physical work. Host lane mapping shows KQ scratch is wave-local; AMD race freedom remains unproven without ISA/device evidence. The pinned selector already chooses native Q8_0/D256 n=1 vector FA for legally padded KV; 1300 bypasses that padding guard without gaining dispatch.

**Owner/consolidation:** RNX02 owns the source proof and any future isolated vector-FA kernel test. Patch 1300 stays untested/inactive, not promoted. 1298 is rejected Q>2 vector evidence. QFP03/QFP17 QSA, PRBE54 Q5 conversion and other active RAD06/RAD10/RHP07/MEN/RCS work are separate and untouched. No new kernel, dispatcher, telemetry, allocator, config or queue.

**12-hour and novelty:** Selected independently active `feat/rad10-helper-counters` at `517d88cf8a9b8e8e35c12602d6a520e3d031ac1d` (2026-10-18 23:03:16 UTC), one commit ahead of `main`. No independent RNX02/1300 implementation, PR, benchmark or recorded queue in the preceding 12 hours (last matching commit 2026-10-07). Excluded RAD06 helper WMMA/AHALF/VECIO/fused staging, RAD10 streaming/small-row benchmarking, RHP07 placement, RCS/MEN release and test work. Unlike earlier RNX02 selector audits, #30573 supplies a new, exact KQ scratch WAR source diff and sanitizer report; its HIP no-op is the terminal AMD discriminator.

**Measured evidence:** Existing BigCherry 1298 MTP3 vector override 58.9 ms/step vs 56.1/54.3 ms baseline (~6% slower), unrelated to this fence. Upstream NVIDIA sanitizer 800,742 -> 0 warnings and 3,993/3,993 FA tests; **not AMD performance/correctness**. No new GPU measurement.

**Cheap validation:** Host-only 99/99 wave-local coverage cases and 216,576 modeled KQ reads passed; Python compilation passed. Inspected b11552 source and #30573 diff. No repository pytest, HIP build, AMD sanitizer, ISA or GPU benchmark.

**Unresolved action/terminal:** `UPSTREAM_NON_HIP_ONLY` and `NATIVE_VEC_ALREADY_SELECTED` are the default dispositions; do not port the no-op fence or promote 1300. Reopen only for a real gfx1100/gfx1201 WAR/ISA receipt or a separately measured >=3% exposed E2E n=1 Q8_0/D256 bottleneck. Then use RNX02's existing profiler/width sweep, test a disposable HIP-valid wave fence only if correctness evidence demands it, require logits/KLD/MTP, graph/multi-request, full work and CI95-low >=3% E2E with <=1% controls. Otherwise close `NO_AMD_RACE`, `BELOW_CEILING` or `NO_GAIN`. No active experiment queued.
