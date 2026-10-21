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

GPT dev-agent design (req_bigcherry_757a112f3c07, read both the throwaway diagnostic copy and the actual 1.9.0 build-tree source before answering):

1. SEPARATE KERNEL PATH, not fused into hc_down_pf2_kernel's existing double-buffered loop. The BF16 path's acc[MT][NW] accumulates across the whole K chunk with one scalar activation factor; FP8 needs per-128-group raw WMMA then per-(row,col)-scale rescale-and-add -- fundamentally different accumulation shape, so do not template/refactor the original kernel. A NEW kernel (hc_down_pf2_fp8_kernel) alongside it, selected by a runtime flag; the original BF16 kernel stays byte-for-byte unchanged.

2. QUANTISATION PRE-PASS (hc_pf2_quant_a8_kernel), launched after hc_pf2_prep_kernel, before the FP8 down kernel. One wave owns (row, 128-K group): read bf16 h, apply scalein[row,c]*(w[k]+wadd), round to bf16 exactly as the existing stageA, reduce 128 abs values via r4d_wave_max (a wave-level op already used elsewhere, not LDS), write 128 E4M3 codes + one f32 scale to new scratch arrays A8[rows][HN] uint8 and Ascale[rows][HN/128] f32. This keeps rocwmma's loaders at ldm=HN for both A and B (no new fragment-layout risk) -- a fused two-slice-lookahead version (what I'd been planning) is explicitly deferred to AFTER this simpler version is proven, not attempted first.

3. INJECTION COLUMNS, a new finding: PF2 has 21 n-tiles (lr=320/16 + hc=4/16-ish = 20 W_down tiles + 1 injection tile at [lr,ncol)). The FP8 path must NOT touch injection -- needs its own small kernel (hc_inj_pf2_bf16_kernel, copied from the existing down kernel's frag/mma arithmetic, only active for n-tiles intersecting [lr,ncol)). FP8-down and injection write disjoint output columns, both feed the unchanged hc_down_reduce_kernel. Do not silently quantise injection weights to FP8.

4. SPLIT-K ALIGNMENT HAZARD, a new correctness finding: the existing split-K scheduling only guarantees 64-K alignment between independent split-K workgroups, but an FP8 128-wide scale group must never straddle two of them (would silently sum across a boundary it shouldn't). Dispatch guard must include kc % 128 == 0.

5. Scratch budget: at HN=10240, lr=320, hc=4, f8=true, ntt=21: existing u capacity 40/80 MiB (M=2048/4096), existing BF16 weight planes+udec 12.81 MiB (both), new A8+Ascale 20.63/41.25 MiB -> totals 33.44/54.06 MiB, fits. Check actual u capacity at runtime (allocated from Mband, not necessarily rows); fall back to BF16 if insufficient rather than changing the scratch ABI.

6. Dispatch guard (full): RAD_HC_PF2_FP8_DOWN=1 (default off) && f8 && n==2560 && hc==4 && lr==320 && rows>=2048 && HN%128==0 && kc%128==0 && sufficient_u_scratch. Emit a diagnostic when requested but ineligible, so a benchmark can't silently measure the BF16 fallback thinking it measured FP8.

7. Patch structure: engines/radiance/patches/rad0007_hc_pf2_fp8_down/{patch.py,patch.toml,SUMMARY.md} (order 7, next available). Anchors in r4d_hc_bf16.hip: after the #include "r4d_hc_fp8.h" line (add a rocWMMA include); insert the new kernels immediately before 'static int hc_pf2_launch('; replace the single hipLaunchKernelGGL(hc_down_pf2_kernel,...) call with a guarded dispatch (prep kernel unchanged -> if eligible: quant_a8 -> fp8 down -> injection (if ncol>lr) ; else: original down kernel -> existing reduce/up/mix/quant unchanged). Edit() calls with expect_matches=1 and distinct guards per insertion.

8. Acceptance sequence: (a) flag off -> byte-identical to existing r4d_selftest; (b) quantiser's A8 codes/scales match the validated offline quantiser, zero mismatches; (c) raw FP8 WMMA per-group products (pre-scale) match standalone harness; (d) full down value (pre-SiLU, including split-K) matches; (e) injection output byte-identical to baseline BF16; (f) end-to-end hc_read numerics/quality -- use the already-measured 2.58% RMS as the expected reference, not an automatic pass criterion; (g) separately benchmark prep/quant/fp8-down/injection/total at 2048 and 4096 rows.

Not yet started: actual implementation of any of these kernels/the patch package. This is now a concrete, buildable plan (substantially more de-risked than before -- the injection-column and split-K-alignment hazards would very likely have caused silent wrong output if discovered only after writing code, not before).

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

2026-10-21 GPT-designed quantisation pre-pass (hc_pf2_quant_a8_kernel: one wave of 32 lanes owns one (row, 128-of-K group), 4 elements a lane, full-wave fmaxf butterfly reduction matching r4d_wave_max's result -- max is order-independent so a plain shuffle loop is numerically identical, unlike sum reductions) validated against the real ground-truth N (tools/lab/radiance/hc-pf2-quant-a8.hip): 0 mismatches across 80 scales and 10240 codes, 2.53% RMS error (same as every prior measurement of this exact quantisation). This replaces the earlier, more complex 8-lane/two-slice-lookahead scheme (hc-pf2-quant-reduce.hip) as the design to actually implement -- GPT's pre-pass approach needs only a full-wave reduction, not cross-thread coordination across two pipeline slices.

Status: activation quantisation pre-pass design is now validated end to end (wave-per-group bookkeeping correct; already proven this composes correctly with the WMMA down-product via hc-down-fp8-e2e.hip, and the weight-side row stride is confirmed against the real container via #366). Remaining for RAD17: the FP8 down-product kernel itself (reading the real packed-codes-then-scales weight layout, ldm=row_bytes(K,128)), the injection-columns kernel (GPT's finding -- a separate small kernel reusing existing BF16 injection arithmetic for the 21st n-tile), the dispatch guard (including the kc%128==0 split-K alignment check), and the actual radNNNN_* patch package with its anchors into r4d_hc_bf16.hip per GPT's design (see detailed_solution).

## Change Log

- 2026-10-21T07:41:01.749670+00:00 (created-by): Created by agent
- 2026-10-21T07:44:23.057374+00:00 (updated-by): Updated: section:notes
- 2026-10-21T07:48:10.715264+00:00 (updated-by): Updated: section:notes
- 2026-10-21T07:56:15.246271+00:00 (updated-by): Updated: section:detailed_solution
- 2026-10-21T07:59:14.580428+00:00 (updated-by): Updated: section:notes
