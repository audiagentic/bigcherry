---
id: PRBE54
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:14.838615+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-KV-001: Dedicated F16 dequant path for Q5 KV

## Description

Materialized as `1271_prbe54_q5_kv_dequant_f16`. b11126 already has dedicated Q4_0/Q4_1 converter kernels; legacy Q5_0/Q5_1 still use the generic `dequantize_block_cont_cuda` scalar-callback path in `ggml_get_to_fp16_cuda()`. The patch adds a HIP half2 F16 Q5 path but exposes it only through a new FlashAttention-specific selector, so weight/general dequant callers remain unchanged. Direct vector-FA Q5 dequant helpers are already specialized; 1271 targets only the `need_f16_K`/`need_f16_V` staging fallback.

## Steps

1. Add one 32-thread Q5 F16 kernel: two Q5 blocks/workgroup, one thread per low/high nibble pair, Q5_0 scale and Q5_1 scale+minimum applied with half2 arithmetic.
2. Add Q5_0/Q5_1 launch wrappers and `ggml_get_to_fp16_fattn_cuda()`. On HIP it selects the new wrappers only for Q5_0/Q5_1; every other type delegates to `ggml_get_to_fp16_cuda()`.
3. Replace exactly the contiguous K and V F16-staging selector calls in `fattn-common.cuh`; no global selector table entry changes.
4. Add package-local activation/correctness/performance producer, contract, README/SUMMARY, and fail-closed mechanics tests.
5. Qualify on gfx1100 and gfx1201. Keep `untested` until >=4 sessions/architecture establish positive Q5 prefill effect and <=1% Q5 decode control regression.

## Detailed Solution & Technical Design

The existing generic Q5 path materializes a `float2` through `dequantize_q5_0`/`dequantize_q5_1`, then scalar-casts both values to F16. `1271` computes the same legacy-Q5 low/high values per thread and writes the two F16 outputs through half2 arithmetic. A 32-thread workgroup handles two 32-value Q5 blocks; an odd final block is bounds-checked.

The new selector is intentionally call-site scoped. `ggml_get_to_fp16_cuda()` stays byte-for-byte unchanged, preventing the experiment from changing non-FA weight/general conversion. The K and V selector replacements use `replace_all` with `expect_matches=1`, so upstream adding/removing a staging site fails closed.

Activation is a once-per-process `BIGCHERRY_PATCH_TRACE` marker emitted only when the FA staging selector receives Q5_0/Q5_1.

## Code Samples & Guidance

Primary source files:
- `ggml/src/ggml-cuda/convert.cu`: Q5 half2 kernel, wrappers, FA selector.
- `ggml/src/ggml-cuda/convert.cuh`: FA selector declaration.
- `ggml/src/ggml-cuda/fattn-common.cuh`: exactly one K and one V staging-selector substitution.

No Q4 changes. No edits to global `ggml_get_to_fp16_cuda()` cases. No change to direct vector-FA Q5 helpers.

## Files

- `patches/1271_prbe54_q5_kv_dequant_f16/{patch.toml,patch.py,README.md,SUMMARY.md,validation.toml}`
- `patches/1271_prbe54_q5_kv_dequant_f16/validation/{producer.toml,producer.py}`
- `tools/tests/patch/test_1271_prbe54_q5_kv_dequant_f16.py`
- `config/experiment-contracts.toml`
- upstream targets listed above.

## Validation

Offline: patch catalog load, contract registry load, mechanics apply/idempotence, missing-anchor/cardinality failures, patch lint/rebase check.

Hardware: gfx1100 + gfx1201; Q5_0 KV with FlashAttention for full-vocabulary backend-reference. Positive lane is pp512 prefill, which selects tile/MMA and therefore the F16 staging path; Q5_0 tg128 decode is the control because quantized decode selects the direct vector kernel and bypasses staging. Marker must be subject-only during the prompt/correctness path. Policy: `improvement_no_regression_v1`, 10 paired rounds/session, >=4 sessions/architecture.

## Effort & Risk

M / medium. Main risks are numerical drift from half2 arithmetic and accidentally broadening the conversion path. Full-vocabulary correctness gates the first; the FA-only selector and exact two call-site edits gate the second.

## Standards

Architecture-scoped evidence; no extrapolation. Small wins count only when established; no materiality bar.

## Acceptance Criteria

Q5 FA staging selector activates only in subject, full-vocabulary backend reference passes, session-bootstrap Q5 pp512 CI95 low > 0, Q5 tg128 control CI95 high <=1%, minimum four sessions on each declared architecture.

## Notes

Supersedes: RD64
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd64

2026-09-24 review corrected the original premise: Q4_0/Q4_1 are already specialized at b11126; Q5_0/Q5_1 are the remaining legacy generic converter cases. The global type selector has non-FA callers, so the implementation is FA-specific rather than globally replacing Q5 conversion.

2026-09-27: materialized as fresh `untested` package `1271_prbe54_q5_kv_dequant_f16` with contract `PRBE54-Q5-KV-DEQUANT-F16`; no hardware evidence is inherited. The positive lane is prefill because b11126's quantized decode path chooses the direct vector FA kernel and does not execute the F16 staging selector.
