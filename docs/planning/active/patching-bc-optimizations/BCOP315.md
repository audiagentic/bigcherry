---
id: BCOP315
order: 315
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-19'
created-by: agent
priority: P2
work: S
---

# RNX08: qualify upstream F32 RMS_NORM float4/register reuse before local kernel work

**Discovery:** Merged llama.cpp [#29720](https://github.com/ggml-org/llama.cpp/pull/29720) adds aligned float4 loads and keeps F32 RMS_NORM input in registers across its reduction/output phases. Pinned b11552 still rereads x. This is a single kernel/dispatch slice, not a new norm+RoPE fusion. Qwen4Exp Q/K use IMRoPE; native three-op RMS_NORM+MUL+ROPE fusion only admits NORMAL/NEOX, while existing RMS_NORM+MUL may already absorb the norm. Actual F32/stride/vector route and exposed time remain unmeasured.

**Authoritative owner/consolidation:** RNX08's dormant norm subfeature owns the technical gate. Existing ggml-cuda norm/fusion kernels and a future approved upstream pin own implementation. 1004 remains rejected as duplicate generic norm+RoPE; PRBE15 IMRoPE SET_ROWS, QFP35 HC POST and RNX10 shared-expert tail are separate. No new kernel, patch package, selector, scheduler, allocator, cache, telemetry or GPU queue. The RNX08 PLE audit remains unchanged.

**12-hour/novelty:** Selected independently active `feat/rad12-lane-classes` from Audumla `61ab2ab2` (2026-10-19 00:53:37 UTC), newer than the independently active workspace branch at 00:41:31 UTC. Last matching RNX08 independent plan commit `a50af2b6` was 2026-10-11 19:15:57 UTC; no matching recent independent commit/PR or recorded queue found. Protected RAD12 helper lane classes, RAD10 streaming/counters, RAD06 helper kernels, RHP07 placement, RCS/MEN and workspace work untouched. #29720 merged 2026-10-16 and vLLM #51406 updated 2026-10-19 supply the novel source/measured mechanism, not a repeat PLE review.

**Evidence:** External #29720 RX 7900 XTX Qwen3.5-27B Q5_K_M pp128/ub1 +0.03%, pp1024/ub512 +1.18%; R9700 +0.55% and +1.31%, respectively; not BigCherry or confidence-qualified. Source-derived maximum x reread removed: `4*ncols` bytes/F32 row, with unchanged output work and launches. Host-only selector/coverage **4,734/4,734** cases, 44,700,957 elements, 4,478 fallback controls; simplified F32 reduction grouping differences 5/168. Python compilation passed. No HIP build, repository tests, GPU correctness, ISA, profiling or benchmark.

**Unresolved action/terminal:** WAIT_PIN. First prove actual eligible executed RMS_NORM+MUL route, per-rank critical-path share and 16-byte alignment on the approved production pin. Reject if no route or optimistically removable E2E time <2.9126%. If admitted, compare native upstream vector versus test-only scalar on isolated gfx1100/gfx1201 with full work, VGPR/spill/occupancy, multi-request/ubatch/graph, logits/KLD/greedy/MTP, and >=4 sessions x >=10 ABBA rounds. Require CI95-low >=3% E2E, <=1% controls. Otherwise close `UPSTREAM_ABSORBED`, `NO_GAIN`, `BELOW_CEILING` or `CORRECTNESS_FAIL`. No implementation or benchmark queued.
