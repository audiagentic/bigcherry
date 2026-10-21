---
id: RAD14
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-19T12:49:22.719925+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Shape-aware MoE execution geometry for the worker (FP8 GEMV, per-projection tiles, per-expert row density)

## Description

GPT review 2026-10-19 (follow-up to RAD13). Optimise execution geometry by expert shape, projection and the ACTUAL routed rows per expert, keeping four decisions separate: expert geometry, quantisation geometry (FP8 checkpoint blocks stay 128x128), execution geometry (GPU tiles, tuned independently) and GPU placement. Primary dispatch variable is rows_e = offsets[e+1] - offsets[e], not total request rows. For Qwen3.6-35B-A3B (256 experts, top-k 8, hidden 2048, intermediate 512) uniform routing averages request_rows / 32 rows per expert, so a 512-token prefill averages 16 and global 32/64-row tiles waste work; real routing is skewed, so decide from observed histograms.

## Steps

1. RAD14-01 (S): instrument per-expert row distributions (routes-per-expert histogram, p50/p95/max run, active experts per layer) and per-projection GPU latency (GU, gated quant, DN, full lane, launch count). Gate: reliable geometry and timing evidence.
2. RAD14-02 (M): benchmark the existing moe_gemm_fp8a8 launcher geometry; sweep GU and DN tiles separately. FIRST inspect the FP8 row's real launcher in pinned radiance 1.8.0 (available M/N/K tiles, tile selection, wave count, LDS, registers, gfx1100 path, whether runtime tile selection exists): native/r4d_moe.hip.rw rewrites the W4 grouped GEMM and R3_MOE_WORKER_MOE_FORM/_DOWN_TILE/_LONG_ROWS must not be assumed to tune the FP8 row. Extend the existing FP8 launcher before adding a kernel family. Candidates (not measured winners): decode 1-2 rows GEMV (M1-2, N32-64, K128); 3-8 rows multi-row GEMV; 9-16 rows M16 grouped GEMM N64-128; 17-32 rows M32; 33-64+ rows M32-64 N128-256; narrow experts I<=256 narrow tiles; wide I>=1024 N128. Gate: reproducible winners.
3. RAD14-03 (M): shape-aware dispatch table keyed (weight_format, arch, H, I, projection, rows_bucket) -> geometry (GU: long K, M16xN64xK128 / large-run M32xN128; DN: more columns, shorter K, M16xN128 / M32xN128-256). Gate: better full-lane latency, no accuracy regression.
4. Native FP8 decode GEMV lane_moe_gemv_fp8<TOKEN> for 1-8 rows (RAD13-04) is the first-priority kernel; avoid WMMA for single-token decode unless measured faster.
5. RAD14-04 (M): adaptive per-expert work bucketing (GEMV_1_2, GEMV_3_8, GEMM_16/32/64 from compact work maps over the sorted offsets, no launch per expert) ONLY if RAD14-03 leaves measurable performance unexplained; a global tile choice can win on small workloads because dispatches cost launch and work-map overhead.
6. RAD14-05 (M/L): heterogeneous expert shapes only for an actual target model: group experts by compatible (N,K,format) and dispatch each group (preserving route and output order) before considering a per-expert-dimension plugin ABI; do not repurpose the even/odd parity classes as arbitrary geometry buckets without updating routing, indexing and pointer tables.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Per layer collect active experts, routes-per-expert histogram, p50/p95/max run length, GU/gated-quant/DN/full-lane latency, launch count, LDS, VGPR, effective weight bandwidth, occupancy. Benchmark H 1024/2048/4096 x I 256/512/1024 x rows-per-expert 1,2,4,8,16,32,64 with uniform and skewed routing; compare (A) existing FP8 grouped GEMM, (B) projection-specific geometry, (C) adaptive per-expert bucketing, (D) native FP8 GEMV, standalone and in the full worker lane; a kernel gain without an end-to-end latency gain is not adopted. Correctness: E4M3 codes, bf16 128x128 scales, group-128 plain activation quantisation preserved; compare to the CPU reference and the R9700 native FP8 engine (max abs, relative L2, p95 element error, token log-prob KL, routing and selected-token differences); W4 Flash-Next replay stays bitwise identical.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Expert placement vs tensor slicing: prefer whole experts across GPUs; split an expert only when memory, severe imbalance or measured latency requires it, align FP8 slices to 128-element scale boundaries, and account for GU placement, DN partial reductions, synchronisation, communication and scale-plane slicing; decide placement from measured per-layer/expert activity and latency (RHP07), not geometry alone. Tuning profiles are keyed (weight_format, arch, H, I, projection, row_bucket) and follow the owner's rule: derive, calibrate once per model x device x build and cache (RCS10), not hand config. Depends on RAD13. Related: RCS10, RHP07.

MTP-collapse bisection, closed out 2026-10-20 (owner: enough time spent). Engine-alone Qwen3.6-35B-A3B FP8 on the R9700 loses MTP draft acceptance at 4+ streams (0.02-0.07 vs 0.61 at 1-2) only with the prefix cache on; `--no-prefix-cache` restores 0.55-0.62 and 3-4x engine decode. Reproduces on near-native radiance (rad0005 compile fix only); absent on the dense Qwen3.8-27B MXFP4 and on Flash-Next (2xXTX). Ruled out: the expert mover (rad0006 freeze), our workers, checkpoint slots, speculation depth, visible stream mixing. Untested: the 35B at `--tp 2` on the XTX; staggered starts (config written); a fully stock build (libr3 may still load); no upstream issue filed. Workaround: `--no-prefix-cache` in every 35B MTP profile. Run specifics, configs, builds and numbers: docs/reference/engines/radiance/models/qwen3.6-35b-a3b.md.

## Change Log

- 2026-10-19T12:49:22.719925+00:00 (created-by): Created by agent
- 2026-10-19T12:49:39.590595+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:notes

## Ledger-events












- chg_20261020_084720_qwen36-35b-a3b-served-with-th_8103
- 2026-10-21T03:57:21.239912+00:00 (updated-by): Updated: section:ledger-events
