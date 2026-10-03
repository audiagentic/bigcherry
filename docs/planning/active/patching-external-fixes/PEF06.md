---
id: PEF06
order: 6
plan: patching-external-fixes
state: pending
created-at: '2026-10-03T21:10:00+00:00'
breadth: cross-plan
skill: advanced
created-by: agent
priority: P0
work: M
---

# Consolidate new AMD inference mechanisms into existing BigCherry owners

## Description

Deep scan of the active planning tree plus current upstream/fork inference work found several useful mechanisms, but the main code-reduction rule is: do not create parallel patch families where BigCherry already has an owner. The highest-value immediate experiment is upstream llama.cpp PR #29910 (opened 2026-10-03), which removes Q2_K MMQ VGPR spills by reducing unroll pressure and deleting an unnecessary temporary loop. Upstream reports spills dropping gfx1100 1387 -> 0, gfx1200 2041 -> 0, and gfx1030 836 -> 0. That exactly overlaps BigCherry's RDNA3/RDNA4 MMQ scope, so test/rebase the existing Q2_K owner rather than add a second implementation.

Secondary mechanisms should be folded into existing owners: vLLM RDNA3 long-prefill BLOCK_M shape tuning belongs under patching-hip-autotune; RDNA3 FP8-KV decode regression evidence belongs under KV/cache policy rather than a new FP8 kernel; vLLM's architecture/shape routing pattern belongs in the existing autotune selector; stew675/llama-cpp-rdna-boosts is a provenance source for fused MoE/k-quant, BF16/WMMA FA and attention-memory ideas, not a patch stack to duplicate.

## Steps

1. Q2_K first: resolve #29910 against the current llama.cpp pin and identify the existing BigCherry Q2_K/Q6_K MMQ patch/plan owner. Diff the upstream change against that owner. Replace/supersede overlapping Q2_K code rather than stacking both.
2. Compile gfx1030/gfx1100/gfx1201 with `-Rpass-analysis=kernel-resource-usage`/LLVM AMDGPU resource diagnostics (or equivalent saved compiler output). Record VGPR, scratch/spill bytes and occupancy for the exact Q2_K MMQ kernels.
3. Benchmark Q2_K `MUL_MAT` and `MUL_MAT_ID` at ubatch 16/32/64/128/256/512 on all three local architectures. Preserve Q6_K as a negative control so consolidation cannot regress the non-Q2_K half of an existing combined patch.
4. If #29910 wins and is source-compatible, delete the superseded local Q2_K implementation and retain only BigCherry-specific qualification/selection glue. Code-size gate: production LOC must decrease or remain neutral; any net increase requires a measured benefit not obtainable by direct upstream reuse.
5. Add a shape-tuning experiment to the existing HIP-autotune owner, not here: test FA/prefill query-block choices around 16/32/64 on gfx1100 and gfx1201. vLLM issue #52585 reports 1.1-1.3x TTFT from BLOCK_M=64 on gfx1100 long prefill; BigCherry must independently measure llama.cpp shapes before changing defaults.
6. Add a KV-policy experiment to the existing cache owner: compare F16/BF16/FP8 KV on gfx1100/gfx1201 with decode attention isolated. vLLM issue #56992 reports FP8 KV can be slower on gfx1100 because conversion cost exceeds bandwidth savings. Do not assume lower-byte KV is faster on RDNA3.
7. Scan the current `stew675/llama-cpp-rdna-boosts` patch blocks by patch-id/source seam. For each fused MoE/k-quant, BF16/WMMA FA, hybrid all-reduce, and attention-memory mechanism, map to an existing BigCherry owner. Only create a new item when no owner exists and the mechanism survives current-pin applicability + independent microbenchmarking.
8. Consolidation pass: for every active plan that targets the same source function/kernel and same performance mechanism, nominate one canonical owner; convert others to dependencies/evidence tasks or retire them. Prefer one selector + one kernel implementation + multiple evidence lanes over duplicated architecture-specific implementations.

## Detailed Solution & Technical Design

### A. Q2_K spill elimination (P0)

Upstream #29910 is unusually strong evidence because it reports compiler-resource deltas across AMD generations, including the exact BigCherry targets. The mechanism is simpler code: gentler unrolling and removal of an unnecessary temporary loop. This aligns with the project objective better than adding another specialized kernel.

Implementation rule:

```cpp
// Conceptual only: reuse upstream structure, do not fork a second Q2_K path.
// Before: aggressive compile-time unroll + temporary accumulation loop.
// After: low-pressure loop structure selected for HIP/AMD where proven.
#pragma unroll 1
for (...) {
    // existing Q2_K decode / dot operation
}
```

Do not hard-code `unroll(1)` globally without NVIDIA controls: upstream itself notes architecture/path sensitivity (GCN DP4A preferred a different unroll in its experiments). BigCherry's selector should remain architecture/path aware only if measurements prove the distinction. Prefer compile-time HIP specialization over runtime branching in the inner loop.

Acceptance performance gate: zero scratch/spill on gfx1100 and gfx1201 Q2_K MMQ where #29910 is applicable; no >1% median regression at any production-relevant ubatch; improvement at the known spill-sensitive shapes; Q6_K negative control within noise. Correctness: existing `test-backend-ops` Q2_K MUL_MAT/MUL_MAT_ID plus whole-model perplexity/output gate already used by the MMQ owner.

### B. Attention shape segmentation (fold into HIP autotune)

vLLM's gfx1100 result demonstrates that a single conservative query tile can leave substantial long-prefill performance unused. BigCherry should not import Triton code; import the mechanism: shape/architecture-keyed selection. Reuse the existing autotune key/cache and add FA candidate dimensions rather than another selector. Candidate key fields: `(gfx, head_dim, nq_per_kv, q_len_bucket, kv_len_bucket, dtype)`. Keep the candidate set deliberately tiny (e.g. query tile 16/32/64) and cache the winner. This reduces code by centralizing shape policy instead of accumulating per-kernel `if (gfx...)` branches.

### C. KV dtype is a performance policy, not a monotonic compression knob

The vLLM gfx1100 FP8-KV regression is directly relevant to RDNA3: lack of native FP8 arithmetic can make conversion dominate decode. Treat KV dtype selection as an architecture + attention-backend decision. On gfx1100, require measured decode win before FP8 auto-selection; on gfx1201 measure separately. Reuse the existing cache policy owner; do not add a second FP8 cache subsystem.

### D. Fork scan / consolidation

`stew675/llama-cpp-rdna-boosts` (rebased 2026-09-17) advertises MTP decode, chunked gated-delta-net prefill, BF16 KV + WMMA FA, fused MoE/k-quant decode, hybrid all-reduce and attention-memory work. BigCherry already has owners for most of these domains. Use the fork as a source of mechanisms and comparison commits only. Patch-id match and source-function overlap must be checked before any port. A fork mechanism that duplicates an active BigCherry kernel should become an A/B candidate under that owner, not a new production patch.

## Code Samples & Guidance

Resource evidence should be machine-readable beside benchmark rows, e.g. `{arch,kernel,vgpr,sgpr,scratch_bytes,occupancy,ubatch,tokens_per_s}`. Add no new telemetry subsystem if current benchmark artifacts can carry these fields.

For attention autotune, extend the existing candidate table/key rather than creating `rdna_attention_selector.*`. The desired code shape is data-driven candidates + one common benchmark/cache path.

For consolidation, use source-function ownership as the dedupe key: same backend + same function/kernel + same mechanism => one canonical plan owner.

## Files

Existing Q2_K/Q6_K MMQ patch and its plan item (resolve exact owner before edit); `ggml/src/ggml-cuda/mmq.cuh` / Q2_K MMQ source at current pin; existing HIP-autotune plan/selector; existing KV/cache policy owner; `config/external-sources.toml`; benchmark/evidence artifacts only as needed.

## Validation

Three-architecture matrix: gfx1030 RX 6900 XT, gfx1100 RX 7900 XTX, gfx1201 R9700. Q2_K compiler resource report + `test-backend-ops` + pp/tg benchmark; Q6_K negative control. Attention: long-prefill buckets including 2k/8k/32k plus decode control. KV: F16/BF16/FP8 decode at short/medium/long context with identical model/cache capacity accounting. Reject results that trade speed for correctness or silently change kernel/backend.

## Effort & Risk

Medium. #29910 is small and high-confidence but still open upstream; source drift and NVIDIA behavior require controls. Attention/KV findings are cross-engine mechanisms, not directly portable code, so they remain experiments until llama.cpp-specific evidence exists.

## Standards

Reuse-before-fork; measurable refactor benefit gate; fail-closed architecture selection; existing experiment-contract/evidence conventions.

## Acceptance Criteria

- #29910 is independently tested on gfx1030/gfx1100/gfx1201 and consolidated into the existing Q2_K owner if beneficial.
- Superseded local Q2_K code is deleted rather than stacked.
- Attention tile and KV-dtype mechanisms are assigned to existing canonical owners with no duplicate selector/cache subsystem.
- Fork scan produces owner mappings and patch-id/applicability evidence before any port.
- Any production change has correctness evidence and a measured speed/code-size benefit.

## Notes

Online references (verified 2026-10-03/04):
- llama.cpp PR #29910: https://github.com/ggml-org/llama.cpp/pull/29910
- vLLM gfx1100 long-prefill BLOCK_M report #52585: https://github.com/vllm-project/vllm/issues/52585
- vLLM gfx1100 FP8 KV regression #56992: https://github.com/vllm-project/vllm/issues/56992
- vLLM ROCm backend routing: https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
- RDNA patch collection: https://github.com/stew675/llama-cpp-rdna-boosts

Current branch context at scan start: patch-refactor @ 4fb725fe3e829cdcea7c40f4b49c9f429e7295de; head message records 1309 Q8_1 publication but no MMVQ hit, so this item deliberately does not create a competing RMSNorm/Q8_1 implementation.

## Change Log

- 2026-10-03T21:10:00+00:00 (created-by): Deep active-plan + upstream/fork scan; added consolidation-first AMD optimization work item.

## Ledger-events

