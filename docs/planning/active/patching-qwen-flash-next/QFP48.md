---
id: QFP48
order: 48
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-10T13:53:50.737727+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P2
work: M
---

# Fusion round 2: choose kernel fusions from the 2026-10-10 prefill trace and a decode trace

## Description

Owner, 2026-10-10: "can we fuse more kernels". The earlier fusion candidates came from an external report (PRBE11 / 37 / 38 / 39 / 40: GEMV epilogue, residual-add, K+V projection; parked in QFP35, now closed). This item picks fusions from our own trace instead.

What the trace says (run dp1, Flash-Next production profile on main at 1042e84d, rocprofv3 kernel trace, share of a target card's kernel time, the same at 24K and 98K): AllReduce 39-46% (QFP49), MMQ incl. MoE 25-30% (mul_mat_q 19.4 s, quantize_mmq_q8_1 1.37 s at 24K), flash attention about 10%, norm / activation / elementwise 4.6% (rms_norm_f32 1.10 s, k_bin_bcast 0.82 s, dsv4_hc_pre_grid 0.63 s), set / get rows and mask build 3.8% (k_get_rows_float 0.82 s, k_get_rows_float_vec 0.74 s), float matmul 3.8%, GDN 2.9%, other 6.1% (bc_moe_router_splitk_partial 1.93 s, bc_dsv4_hc_post_gate_grid 1.15 s, bc_mm_ids_helper_mw 0.82 s).

Fusion pays where two memory-bound kernels read or write the same data back to back. Candidates to check, each against the trace order of kernels (not assumed): quantize_mmq_q8_1 into its producer (rms_norm or the HC pre-grid), the two get_rows passes, k_bin_bcast into its neighbour, and the 1.93 s router partial against its reduce.

## Steps

1. Add a decode trace mode to long-ctx-profile.sh (prefillprof exists; decode has none) and take it on the current build.
2. From both traces list adjacent kernel pairs by total time (same stream, consecutive, same tensor), per card.
3. For each pair above 1% of prompt wall or of decode step time: check existing fusions (1307-1313, 1344, 1355, native GLU) do not already cover it; write the candidate with its expected saving.
4. One patch per fusion, default off, lightweight tier: mechanics tests, marker, ABBA with complete separation, greedy identity.
5. Read the ExLlamaV3 RDNA3 ports (wtogami/exllamav3-rocm, phoenixhaxor/exllamav3-rocm-moe: "fused-core prefill kernels") for fusions we have not considered; reference only.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per fusion: offline mechanics + patch-lint, BIGCHERRY_PATCH_HIT marker or census showing the launch removed, ABBA on one binary with complete separation at 8K / 24K / 98K, greedy identity (GGML_CUDA_DISABLE_FUSION=1 control on both arms where the fusion pass is involved), no q4 KV.

## Effort & Risk



## Standards



## Acceptance Criteria

- A decode trace exists for the current build, with the same family table as the prefill one.
- Each fusion candidate from the trace is either promoted (lightweight tier) or rejected with its measurement.
- No candidate is taken from an external timing without our own trace showing the pair is adjacent and material.

## Notes

Depends on nothing; the AllReduce work (QFP49) is larger and comes first for prefill. Bandwidth per kernel (weight bytes read over kernel time, against 950 GB/s for a 7900 XTX and 640 for the R9700) is to be added to the trace table so a fusion's ceiling is known before it is written. Related: QFP44 (MMQ tile geometry), QFP46 (prefill concurrency contracts).

## Change Log

- 2026-10-10T13:53:50.737727+00:00 (created-by): Created by claude
