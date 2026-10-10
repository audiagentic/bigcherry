---
id: BCOP77
order: 77
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-08T22:03:36+00:00'
created-by: agent
priority: P2
work: S
---

# PRVP02: native CM1 provenance and partial-K A-prefetch safety

## Discovery / disposition

llama.cpp #27952 merged 2026-09-24 and is an ancestor of pinned b11474. Native CM1 includes five coupled files; the planned 1246 transplant is absent/obsolete (PRVP01 terminal). Native Vulkan auto-selects Q8_1 CM1 on supported RDNA3/4; the old "available but unselected" assumption is false. Pinned and current upstream mul_mmq_cm1_funcs.glsl::PREFETCH_BLOCK lack an end_k guard for A while B is guarded. This is a source-supported potential OOB, not reproduced GPU corruption. External issue #29342 is open; its K=5184/128 arithmetic claim requires correction. Nine host index fixtures passed; no GPU or build run.

## Authoritative owners / protected work

PRVP02 owns the narrow A-prefetch safety and native capability reconciliation. PRVP01 source transplant is retired; TRVP14 owns optional strict route, TRVP15 owns later performance evidence. RRVP02's Vulkan implementation pause remains authoritative. PRVP03 Vulkan AllReduce, PKC04 lifecycle, active QFP41/HIP Meta, MTP/QFP, QSA and MoE work are separate and untouched. No new shader, scheduler, cache, allocator, registry or queued hardware lane.

## Bounded action / terminal gate

When Vulkan hardware work is permitted, source-instrument dense/ID partial-K final tiles (K=160/960/5184/2880 and aligned controls) on gfx1100 RADV, then gfx1201 separately. Prove actual bound CM1, descriptor ranges, work counts and CPU/greedy/logits/KLD parity across multi-request/ubatch/MTP. If needed, use the smallest A-index clamp matching the B guard; prefer an upstream fix and reject any separate kernel clone. A safety correction needs complete correctness and <=1% E2E regression, not an invented speedup. Close on upstream absorption or disproved OOB with reproducible evidence. A distinct performance proposal remains TRVP15 and needs isolated route controls plus CI95-low >=3% E2E. No action during the Vulkan pause.

## References

PRVP01; PRVP02; TRVP14; TRVP15; RRVP02; https://github.com/ggml-org/llama.cpp/pull/27952 ; https://github.com/ggml-org/llama.cpp/issues/29342 ; https://github.com/ggml-org/llama.cpp/blob/b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq_cm1_funcs.glsl .
