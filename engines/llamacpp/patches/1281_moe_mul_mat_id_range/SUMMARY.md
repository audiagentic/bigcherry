# 1281_moe_mul_mat_id_range

**Status:** validated (existing BigCherry qualification; not re-run in the 2026-10-11 MET09 audit)
**Plan item:** MET02

`ggml_mul_mat_id_range(ctx, as, b, ids, id_base)` is the range variant of `GGML_OP_MUL_MAT_ID`: a local weight bank holds experts `[id_base, id_base + ne02)`, selected IDs remain global, and nonresident lanes produce exact +0 without reading foreign weights. Ordinary `ggml_mul_mat_id` remains unchanged. The variant is used by the opt-in expert-tier graph (1283); it does not change normal graphs when unused.

## Current implemented scope

The patch contains CPU/reference range semantics **and HIP/CUDA MMVQ/MMQ range dispatch**, including global-to-local ID translation and a conditional QFP30 broadcast-dedup scatter path. For non-native-FP4 broadcast MMQ, `bc_range_dedup = bc_range && !use_native_fp4 && ne11 == 1 && n_expert_used > 1`; the inverse map is initialized to -1 and `quantize_scatter_range_mmq_q8_1_cuda` skips those inactive entries. Native-FP4, nonbroadcast, MMVQ and fallback paths do not all share that branch. Never describe all range MMQ as dedup-disabled or as CPU-only.

## Existing evidence and limitations

See `README.md` for the *previously measured* b11402/d89651a7 CPU 432/432, GPU 720/1440/1152/432/432 cases by token band, explicit backend-policy declines, unused-path greedy MD5 f80ea4df and opt-in EP fidelity/performance. Those results were **not** reproduced on b11474 in the 2026-10-11 MET09 audit. The previous Phase-A-only and pending-reference-test wording here was stale.

The README explicitly does **not** qualify duplicate expert IDs within one token on MMQ; repacked-weights CPU and fused-launch coverage remain limited. Upstream [#30262](https://github.com/ggml-org/llama.cpp/pull/30262) proposed a duplicate-ID compact-offset correction but closed unmerged; treat it as a negative-test mechanism, not an adopted fix. MET09 owns first-divergence attribution, MET02 owns any confirmed range-kernel defect. No new patch, build or hardware result is claimed.
