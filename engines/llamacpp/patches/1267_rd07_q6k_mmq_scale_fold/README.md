# 1267 RD07 Q6_K MMQ scale fold

Fresh `untested` extraction of the RD07 sub-piece from rejected package 1203. It hoists Q6_K row base scales and folds the per-k01 sub-scale before the j0 accumulation, removing repeated scale work without importing RD05/RD06 or diagnostic timing code.

## Provenance

- Source fork commit: https://github.com/stew675/llama.cpp/commit/1d525bd45f9e8f844856ecbc5dd8ae33c8d34eff
- Pinned upstream dispatch verified at `ggml-org/llama.cpp` b11126: `ggml/src/ggml-cuda/mmq.cu` uses `mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);` with no `forced_J` argument.
- Rejected source package: `1203_rd050607_rdna4_wmma_fa_q6k_mmq`; its validation evidence is not inherited.

## Scope

Includes only RD07 Q6_K scale hoist/fold, a `BIGCHERRY_PATCH_TRACE` Q6_K MMQ dispatch marker, `GGML_CUDA_MMQ_J_MAX` qualification knob, and Q6_K-only perf cases with Q8_0/F16/F32 controls. It explicitly excludes all RD05/RD06 flash-attention changes and 1203 op-timing instrumentation.

## Validation

Contract `PRBE110-RD07-Q6K-MMQ-SCALE-FOLD`: gfx1100/gfx1201/gfx1030, backend-reference correctness, subject-only marker, prefill positive and decode control, 10 paired rounds/session, 4 sessions, `improvement_no_regression_v1`, combined llama-bench invocation. State remains `untested` until fresh evidence is produced.
