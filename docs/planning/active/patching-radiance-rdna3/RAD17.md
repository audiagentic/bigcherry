---
id: RAD17
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-21T07:41:01.749670+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P2
---

# hc_read prefill FP8 WMMA down-product: production port into r4d_hc_bf16.hip

## Description

Port the already-validated FP8 WMMA down-product (RAD15) into libr4d's actual hc_down_pf2_kernel. The WMMA mechanics, the cross-thread amax reduction, and their composition are all proven correct on real ground-truth data in standalone harnesses (tools/lab/radiance/hc-down-fp8-validate.hip, hc-pf2-quant-reduce.hip, hc-down-fp8-e2e.hip). What remains is splicing that proven logic into the real kernel's existing double-buffered, prefetch-ahead pipeline without disturbing its scheduling or its bit-identical-reduction-order guarantee against the wide/per-row forms.

## Steps

1. Real-weight ground-truth extraction: same ndump technique used for the activation (N), applied to W_down, to validate the weight side the same way the activation side was validated.
2. Port the proven quantise_n + hc_down_fp8 kernels into hc_down_pf2_kernel's actual fetchA/stageA double-buffered slice loop (kTAP, barriers, prefetch-ahead bp[]), behind a default-off flag.
3. Patch package (radNNNN_*, patch.py/toml/README/SUMMARY/tests), rad-kbench correctness at rows 17/32/64/.../4096, then a Flash-Next clone of qwen35b-helper-kld.json's quality_only harness (thresholds: max_kld_mean_increase 0.02, median 0.01, p99 0.2, top1_drop 1.0, ppl_ratio 0.05).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Continuation of RAD15's hc_read prefill FP8 WMMA work, split into its own item since RAD15 covers broader engine-decode-row scope. The design/correctness work is done (see RAD15's notes for the full validation history); this item is specifically the production-port engineering, held at low priority pending either explicit acceptance of the correctness risk under continued low-effort work, or a dedicated higher-effort session for it.

## Change Log

- 2026-10-21T07:41:01.749670+00:00 (created-by): Created by agent
