---
id: PRBE04
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:53:43.516065+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# RD07 Q6_K MMQ safety and dense-shape qualification (downstream of PRBE110)

## Current disposition (2026-10-10)

The earlier description that RD07 is blocked on PRBE110 *creating* a patch is obsolete: `1267_rd07_q6k_mmq_scale_fold` already exists, with mechanics tests, an experiment contract, a producer, and four-session historical b11126 evidence. It remains **untested** on the current b11474 pin and is **not** in production. Rejected 1203 evidence is historical only. PRBE110 is the sole technical owner of 1267, including current-pin composition and promotion/retirement; PRBE04 does **not** author another Q6_K patch, selector or benchmark producer.

PRBE04 owns only two mandatory gates before any PRBE110 performance campaign:

1. **PEF01 / memory safety:** exact composed b11474+0300+1006+1267 backend-op Q6_K boundaries, with particular attention to Q8_1 staging and J allocation. Upstream llama.cpp [#29953](https://github.com/ggml-org/llama.cpp/pull/29953) merged 2026-10-08 to fix MMQ OOB reads from inconsistent allocation/launch tile selection; [#30168](https://github.com/ggml-org/llama.cpp/pull/30168) merged 2026-10-09 to align host precision config. Pinned b11474 retains the old `ggml_cuda_mmq_get_J_max` allocation logic. Source-derived RDNA4 Q6_K `ne11=17` can reserve J=16 and dispatch J=32 if both configurations pass the shared-memory gate. This is **not a reproduced GPU fault**. Require current-pin correction/equivalence evidence and bounded allocation/launch fixtures before hardware performance tests; coordinate the generic fix with its existing upstream-fix owner, not PRBE04.
2. **HI71 / dense-shape and real activation:** require actual gfx1201 Q6_K MMA execution with 1267, not just the host `ggml_cuda_mul_mat_q_switch_type` once-marker. The patch modifies `ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma`; gfx1030 DP4A is unchanged. Verify J, fallback, K and M shape, source precision, baseline/subject build identity, and non-Q6/DP4A controls. Use the existing PRBE110 producer and profiling infrastructure.

The b11126 1267 gfx1201 pp512 +2.4675% (CI95 [2.2776%,2.6508%]) was measured without validated 1006 in its stored baseline composition. Current production `validated-enhancements` includes 1006, independently measured around +18% pp512; **do not add effects or reuse the old 1267 delta as incremental**. First measure non-overlapped Q6_K MMA contribution against the actual 1006 baseline. If its theoretical end-to-end ceiling is below 3%, close 1267 without further A/B. Otherwise PRBE110 owns four-session >=10-pair ABBA gfx1201 qualification, full-vocabulary/logit/greedy/MTP parity, multi-request/graph safety, CI95-low >=3% end-to-end improvement and <=1% controls.

No new patch, test queue, telemetry system, allocator, scheduler, or configuration surface is authorised here. Historical 1203/RD07 +3.4% pp512/+6.2% pp2048 bundled evidence remains confounded and not promotable.
