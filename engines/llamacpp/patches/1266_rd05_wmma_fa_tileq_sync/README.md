# 1266 RD05 WMMA FA tile_Q synchronization

Status: **untested**. Plan: PRBE110. Contract: `PRBE110-RD05-WMMA-TILEQ-SYNC`.

## Scope

Extracts only RD05's two synchronization fixes from rejected package `1203_rd050607_rdna4_wmma_fa_q6k_mmq`: the end-of-`k00` barrier condition and the inter-`process_tile` barrier protecting the reused `tile_Q` buffer. It does **not** carry RD06's RDNA4 configuration table, head-dimension gate, softcap changes, or RD07 MMQ work. The only non-source-port change is the env-gated, once-per-process activation marker at the real WMMA-F16 dispatch.

All edits are anchored with `expect_matches=1`; a missing or duplicated anchor fails closed. This package conflicts with rejected 1203 because their RD05 edits overlap textually.

## Validation

The producer requires gfx1201, full-vocabulary backend-reference correctness, subject-only activation marker, and combined llama-bench decode/prefill measurement. Contract policy is `improvement_no_regression_v1`, 4 sessions, 10 paired rounds. Historical 1203 evidence is not reusable: this package has a fresh patch and contract identity.

## Sources

- Fork source commit (RD05/RD06/RD07 mixed change): https://github.com/stew675/llama.cpp/commit/1d525bd45f9e8f844856ecbc5dd8ae33c8d34eff
- Upstream flash-attention WMMA implementation: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-mma-f16.cuh
- Upstream flash-attention dispatcher: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu


## 2026-10-11 b11552 compatibility / qualification correction

**Historical README above is not a b11552 application instruction.** Merged upstream [#30103](https://github.com/ggml-org/llama.cpp/pull/30103) replaces the original end-of-`k00` guard with `if (np > 1 || nbatch_combine != DKQ/2)`. This differs from this patch's `k00 + nbatch_combine < DV/2` condition, so the first `expect_matches=1` anchor no longer exists at b11552. The second `kbc`-boundary barrier is still a distinct, **unproven** residual hypothesis.

The existing once-per-process WMMA dispatch marker is not a per-`k00` or per-`kbc` activation counter. Four historical gfx1201 sessions proved full-vocabulary parity but no throughput benefit and no reproduced stock race. Do not apply or qualify this two-edit package on b11552 without a newly demonstrated post-#30103 failure and a separately attested residual-only identity. Authoritative decision: PRBE02 / BCOP136; do not modify PRBE110/1267 while current MMQ/pin work is active.
