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

2026-10-21 real-weight ground-truth extraction, step 1: found a genuine layout error before writing any dump code, by reading r4d_hc_fp8.h directly rather than reusing the MoE op's weight-format assumption. hc_read's own fp8 weight storage is NOT the separate-codes/separate-scale-array layout the moe_gemm_q op uses (which the validated standalone harness assumed for W_down). The real format: each output row is one packed blob [K E4M3 codes][K/128 f32 scales][pad to 16 bytes] (r4d_hc_fp8.h, r4d_hc8::row_bytes), scale as f32 not bf16, codes and scale interleaved per row rather than in separate tables.

Consequence: row_bytes(K=10240, G=128) = 10240 + 4*(10240/128) = 10240 + 320 = 10560 bytes (already 16-byte aligned), NOT 10240. The validated standalone harness (hc-down-fp8-validate.hip) used ldm=HN=10240 for matrix_b's col_major load, assuming a clean [N,K] codes array -- that is WRONG for the real on-disk layout; the real stride between consecutive weight rows (= WMMA's 'columns' in col_major terms) is 10560, not 10240. Porting the harness's ldm assumption straight into the real kernel would have silently read scale bytes as if they were weight codes for columns near a row boundary -- the exact silent-wrong-output failure mode already caught twice this session, now caught a third time before it reached real code.

This changes the real kernel's B-fragment load: ldm must be r4d_hc8::row_bytes(K,128) (computed from the real K per class, not assumed equal to K), and the scale read for a given column's group g is a plain f32 at codes_row + K + g*4 (immediately after that row's K code bytes), not from a separate scale buffer. Also note mix_down's group is fixed at 128 (r4d_hc8::kDownGroup) matching the weight's own existing granularity -- no mismatch there, only the stride assumption was wrong.

Next: dump real W_down bytes from the container (same ndump-style technique, reading directly from the op's 'down' operand pointer in hc_pf2_launch -- no kernel instrumentation needed this time since the weight is a stored input, not computed inline) to confirm this row_bytes math against actual container bytes before updating the standalone harness or touching the real kernel.

2026-10-21 real W_down ground truth extracted and the corrected row stride confirmed directly against the live container (not just arithmetic from r4d_hc_fp8.h). Same throwaway-scratch-rebuild technique as the N dump (BC_HC_WDUMP_ROWS env hook added right at the top of hc_pf2_launch -- dld, the real per-row stride, is already a parameter there, so no kernel instrumentation was needed, just a host-side hipMemcpy+fwrite of the raw 'down' operand bytes during a live rad-kbench hc_read call on the real Flash-Next container; never touched the pinned tree).

Result: dld=10560 (exactly matching the predicted row_bytes(K=10240,G=128)=10240+320=10560), f8=1. Dumped 4 real rows (42240 bytes) and parsed them: all 10240 code bytes nonzero in every row, and the 80 f32 scales per row (read at the predicted offset, right after the K code bytes) are sane, small positive numbers (range ~5e-5 to ~1.3e-3 across the 4 rows) -- confirms both the corrected stride AND the scale's position/type (f32 inline, not a separate bf16 table) against real data, closing step 1 of RAD17.

Step 1 of RAD17 (real-weight ground-truth extraction) is now done. Next: step 2, porting the proven quantise_n + hc_down_fp8 kernels into hc_down_pf2_kernel's actual double-buffered loop -- using dld=row_bytes(K,128) (not K) for the weight's B-fragment ldm and reading each column's group scale as f32 at codes_row+K+g*4, both now confirmed against real data rather than assumed. A GPT dev-agent design review for this exact splice is in flight (req_bigcherry_757a112f3c07).

## Change Log

- 2026-10-21T07:41:01.749670+00:00 (created-by): Created by agent
- 2026-10-21T07:44:23.057374+00:00 (updated-by): Updated: section:notes
- 2026-10-21T07:48:10.715264+00:00 (updated-by): Updated: section:notes
