# 1210_rd26_bitidentical_decode_verify_standalone

**Status:** untested
**Plan item:** RD26

**State:** untested (not promoted). **Authoritative plan:** PRBE20. **Audit:** 2026-10-13 / BCOP184.

The patch is **not** just the original two-hunk subset. `patch.py` includes plain/fused MMVF selection, CPU sgemm gate, FlashAttention WMMA/tile selection and RDNA3/RDNA4 MMVQ nwarps changes. The source provenance remains the stew675 fork; the historical “remaining three hunks deferred” statement is obsolete.

**Actual evidence:** dev gfx1100 `llama-results` width 1 versus width 5 raw-logit comparison still failed at the first logit row (historical offset 480); widths 2/3/4 differ later (~1,987,040). Historical PPL parity does not qualify this invariant. No new GPU run or performance gain.

**Unresolved source-level discriminator:** llama.cpp b11552 `mmvq.cu::get_mmvq_mmid_max_batch_rdna3` returns 4 for Q4_K/Q5_K/Q6_K, and `ggml-cuda.cu::ggml_cuda_mul_mat_id` enforces that limit. Width 5 may leave MMVQ while 1210 only normalizes `calc_nwarps`; the historical model's actual quant type/route are **not yet proven**. Do not override the tuned threshold without tracing first divergence.

**Next gate:** reuse `validation/producer.py` and `tools/lab/prbe20-rd26-bisect`; first check GGUF quant type and actual per-node dispatch, then a bounded diagnostic `cb_eval` first-divergent-op trace (not a performance run). Separate early width-5 from late width-2/3/4 divergence. Only after causal proof consider an opt-in selector change; require full-logit/greedy/MTP correctness, same-process replay and paired AMD performance. Do not duplicate routing, scheduling or telemetry. See PRBE20 for exact gates and terminal outcomes.
