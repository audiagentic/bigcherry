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

2026-10-21 FP8 down-product validated against the REAL packed weight layout, on real data both sides (tools/lab/radiance/hc-down-fp8-real-layout.hip): real ground-truth N (quantised with the validated wave-per-128-group scheme) against real W_down bytes (dumped #366, 16 rows), using matrix_b's col_major load at ldm=DLD=10560 (not K=10240) and reading each column's group scale as a plain f32 at weight_row_base+K+g*4. Result: max rel diff 3.2e-6 against a quantised-operand reference built the same way -- exact (fp32 noise floor). This closes the last open correctness question on the WMMA mechanics themselves: the real weight layout, the real activation quantiser, and the proven rocwmma::fragment approach all compose correctly together on real data.

RAD17 status: every individual piece of GPT's design is now validated standalone on real data -- the quantisation pre-pass (wave-per-128-group), the WMMA down-product against the real weight layout, and (from RAD15) the activation-side end-to-end composition. What remains is assembly, not more correctness research: (1) the injection-columns kernel (not yet built or validated -- GPT's finding, a separate small kernel reusing existing BF16 injection arithmetic for the 21st n-tile); (2) wiring all pieces into the actual radNNNN_* patch package with real anchors into r4d_hc_bf16.hip, the dispatch guard (including kc%128==0), and the flag; (3) the in-kernel integration itself (porting these validated standalone kernels' logic into properly-scheduled HIP kernels matching the real grid/block dimensions hc_pf2_launch uses, not just 1-workgroup proof-of-concept tests); (4) the full acceptance sequence against r4d_selftest and the KL harness.

2026-10-21 injection-columns piece simplified by reading the real epilogue (r4d_hc_bf16.hip ~line 2478-2495). Injection is NOT structurally separate from the down-product's K-sum -- it's the SAME WMMA accumulation (acc[m][i][e], same plane, same K dimension), just a different column range (col>=lr) with a different epilogue: silu(v*inv_hc) -> dout[r*lr+col] for col<lr (W_down), vs 2*sigmoidf_(v*inv_hc) -> inj[r*irow+(col-lr)] for col>=lr (injection). The injection weight (injw) has no FP8 variant in the schema at all (RadOperandSpec's inject_w is a plain weight operand, loaded by hc_pf2_prep_kernel's plane-builder via the bf16 branch unconditionally, never through the f8-decode path) -- so there is no new numerics to validate for injection: it is exactly the existing, already-shipped bf16 WMMA arithmetic, just needing to run as its own kernel (or its own code path) covering only the injection column range once W_down's columns are peeled off into the new FP8 kernel. No standalone validation harness is needed for this piece specifically (unlike every FP8 piece, which all needed and got one) -- it is a direct, verbatim extraction of existing proven code, not new logic.

RAD17 status: ALL correctness-risk pieces are now validated on real data (quantisation pre-pass, WMMA mechanics, real weight layout, their end-to-end composition). The injection piece carries no correctness risk of its own (verbatim existing arithmetic). What remains is pure integration engineering: assembling the validated FP8 kernel, the extracted injection kernel, and the dispatch guard (kc%128==0 etc.) into the real hc_down_pf2_launch flow as an actual radNNNN_* patch package with real anchors, sized for the real grid/scheduling (not the 1-workgroup proof-of-concept harnesses used for validation), then running r4d_selftest and the KL harness.

2026-10-21 first patch package increment, rad0007_hc_pf2_fp8_down (PR #370), compiled and verified on real hardware: adds hc_pf2_quant_a8_kernel (validated) and hc_down_pf2_fp8_kernel (WMMA/layout math validated, rescale-and-accumulate epilogue at the real RT=64/kWaves tiling left as an explicit TODO) to libr4d/r4d_hc_bf16.hip. Deliberately inert -- hc_pf2_launch untouched, no dispatch wiring, zero behaviour change. patch-rebase-check --focal-overlay reports CLEAN against the pinned upstream; applied programmatically (bigcherry.patch.apply.apply_patch, never the shared wt-accept checkout) to a scratch copy and compiled for gfx1201 with zero errors; confirmed both kernels are real compiled device code objects (mangled names + .kd descriptors present via nm/strings).

Next: finish hc_down_pf2_fp8_kernel's epilogue (the TODO), implement hc_inj_pf2_bf16_kernel (extraction, no new numerics), then wire the dispatch guard (RAD_HC_PF2_FP8_DOWN, kc%128==0, the other eligibility checks from GPT's design) into hc_pf2_launch -- only once that's done does this patch do anything at runtime, and only then can r4d_selftest / the KL harness actually exercise it.

2026-10-21/22 implemented GPT's grid-underfill diagnosis (req_bigcherry_46f3e71a16c4, session ses_0e45b277c600463c -- the session we're reusing for this whole iterative pass). GPT read the actual kernel code and found the real, confirmable cause of the occupancy sensitivity from the previous two reverted attempts: at M=2048/ks=2, RT=64 gave the FP8 down kernel's grid only 64 workgroups -- matching the R9700's CU count exactly, a genuine grid-underfill bottleneck, independent of and untested by the accumulator-hoisting attempts.

Halved RT to 32 for the FP8 kernel only (own constant, not kPf2RT -- bf16 kernel unaffected). Measured: 1203.7us->999.6us at M=2048 on real hardware -- the single largest win this entire session. Also tried RT=16 per GPT's own stopping rule ('if RT32 improves, test RT16 next'): measured 999.2us, statistically identical to RT=32 -- confirms grid underfill is now FULLY resolved, not partially; smaller tiles plateau. Kept at RT=32. PR #373 open.

Session running total: 1457.6us->999.6us (~31.4% faster), now ~1.9x slower than the 530.7us bf16 baseline -- closer, still not a win. Flag off re-verified byte-identical/unaffected at every single step across this whole optimization pass.

Workflow note: per explicit user instruction, continuing this as an iterative implement->test-on-hardware->report-to-GPT-in-the-same-session->repeat loop. Next: report this RT=32/RT=16 result back to GPT in ses_0e45b277c600463c and ask for the next concrete step.

2026-10-21/22 MAJOR FINDING AND WIN: GPT dev-agent profiling (req_bigcherry_f0af77769407), a full paired BF16-vs-FP8 stage comparison using the same instrumented binary (flag off vs flag on, event markers after every kernel including output-quant), found the real architectural problem that all the prior down-kernel tiling/hoisting work was NOT addressing: BF16's hc_down_pf2_kernel does the down product AND injection together in ONE fused kernel, costing 0.159ms total. This patch's equivalent logical work -- hc_pf2_quant_a8_kernel + hc_down_pf2_fp8_kernel + hc_down_pf2_fp8_reduce_kernel + the (then-separate) hc_inj_pf2_bf16_kernel, FOUR kernel launches -- cost 0.642ms, roughly 4x more expensive for the same computation, DESPITE reading less weight data from global memory than BF16's decoded plane. prep/up_mix/output-quant measured nearly identical between both paths, confirming the entire gap was concentrated in down+inject specifically, not spread across the pipeline.

Fix: fused injection's per-group partial dot product into hc_pf2_quant_a8_kernel's own pass, reusing the quantised activation while still in registers instead of re-reading it from global memory a third time (new hc_inj_pf2_fp8_reduce_kernel finalises the accumulated partials, piggybacked on the same dispatch step as the down-product's own split-K reduce; ks==1 keeps the old standalone hc_inj_pf2_bf16_kernel unchanged since there's no reduce launch to piggyback onto in that case -- every real PF2 shape found this session has ks>1 anyway, so this is effectively always the fused path in practice).

Measured on real gfx1201 hardware: 999.6-1030.6us -> 920.3us at M=2048 -- a real, substantial win. CRITICALLY, validated BIT-FOR-BIT against the previous (unfused) injection kernel's own output, same input, same run, across every test case the harness ran: rms_rel=0, max_abs=0 -- every single comparison, not merely close -- confirming the fusion refactor is mathematically identical to the already-reviewed prior kernel, not just plausible-looking, before being trusted. (Built a throwaway host-side bf16 comparison helper since bf2f is device-only -- real bf16 bit-layout conversion via union, not an approximation.)

Also closed out accumulator-hoisting as a dead end for good: retested the B-fragment hoist at the NEW RT=32/MT=2 shape (only 2 live accumulators needed, grid no longer underfilled by the earlier RT fix) per GPT's own suggestion that the earlier RT=64 hoist regressions might not transfer -- it STILL regressed (999.6us -> 1244.9us), a THIRD independent data point in the same direction. This closes the question definitively: more rocwmma::fragment tiling/hoisting variants are not the path forward on this kernel shape.

Session running total: 1457.6us -> 920.3us (~36.9% faster overall this session), now ~1.73x slower than the 530.7us bf16 baseline -- meaningfully closer, still not a win. PR #378 open (branch feat/rad0007-fuse-injection, on top of merged #371-#376).

Still outstanding: a real quality metric (RMS/KL) against the ACTUAL bf16 reference output (not just the bit-for-bit self-consistency check above, which is a narrower, different question -- confirms the refactor didn't introduce a NEW bug, says nothing about how close the lossy FP8 path's inj/down/x outputs are to the real bf16 numbers). GPT's earlier review flagged this as still required before any promotion decision, independent of how much faster the path gets.

Workflow note: continuing the implement->test-on-hardware->report-to-GPT-in-the-same-session (ses_0e45b277c600463c)->repeat loop per explicit user instruction.

2026-10-22 fused the two FP8 reduce kernels (hc_down_pf2_fp8_reduce_kernel now also finalises injection partials inline, removing hc_inj_pf2_fp8_reduce_kernel's separate launch for ks>1). Measured 920.3us->~911-915us at M=2048 across 4 repeated runs -- small (~1%) but real and reproducible. Validated bit-for-bit against the REAL hc_inj_pf2_bf16_kernel reference directly this time (not just the prior patch revision): rms_rel=0, max_abs=0. Pushed as a new commit on PR #378 (same PR as the quant+inj fusion).

Session running total: 1457.6us->~912us (~37.4% faster overall), still ~1.72x slower than the 530.7us bf16 baseline.

Reported back to GPT (req_bigcherry_a3f02be7f186, same reused session ses_0e45b277c600463c) asking: whether launch-count fusion is now exhausted (quant_a8 and down_fp8 are the only two FP8-specific kernels left, and GPT already said quant-into-down fusion needs substantial LDS/loop restructuring, not a cheap win); what the smallest testable first step for a raw-WMMA rewrite of down_fp8 would be, isolated as a single-variable change; and whether a 2x improvement on down_fp8 specifically (needed to make the whole pipeline competitive) is a realistic expectation for raw WMMA on this hardware, or optimistic. Response pending.

Unrelated but valuable finding this round, in response to the user's direct question about CI build times: confirmed via real log inspection that the ~20-30min radiance-build CI job is NOT wasted on repeated apt-get/docker setup (that's only ~40 seconds) -- it's a genuine 147-object-file C++/HIP compile from scratch every time, because ccache has never actually hit (GitHub Actions scopes caches per-branch, and this workflow never ran ON main itself to populate a cache other branches could reach -- PR #377, open, fixes this by also triggering on push to main).

2026-10-22 STRATEGIC PIVOT (user-relayed GPT assessment): paused incremental FP8 down-product work to investigate whether the BF16 hc_up_mix_wide_kernel (measured 261us of the ~531-540us full BF16 call, roughly half) has real headroom, before continuing to chase the FP8 path's remaining ~1.72x gap.

Step 1 (GEMM-only): built tools/lab/radiance/hc-upmix-gemm-bench.hip -- a tuned hipBLASLt BF16 GEMM at the up-product's real shape (M=2048, K=lr=320, N=HC*n=10240). Measured (gfx1201, device idle-confirmed, 8 candidate algos): best = 148.12us/call, vs the fused kernel's 261us total. A real device-side kernel hang was hit and fixed during this step (hipblasLtMatmul operand order must match the col-major-transpose-trick layouts: pass B then A). Committed on feat/rad0007-fuse-injection (9b4db17bc), merged to main via PR #378.

Step 2 (GEMM + separate epilogue, per GPT's own decision framework, which requires this before concluding anything): built tools/lab/radiance/hc-upmix-epilogue-diag.hip -- hipBLASLt GEMM (fp32 output, matching the production fp32 WMMA accumulator) + a separate HIP kernel reproducing hc_up_mix_wide_kernel's exact mixing formula (sigmoid, narrow to bf16, multiply by N = h*scalein*(w+wadd), sum over HC=4, divide by HC, narrow to bf16), reproduced from r4d_hc_bf16.hip read directly on dev (vendor/radiance/libr4d), not re-derived from memory. GPU-event timing (not host wall-clock), real random (non-constant) activation/weight data.

RESULT: gemm=168.81us, epilogue=226.19us, COMBINED=395.01us -- SLOWER than the production fused kernel's measured 261us. The epilogue dominates and is bandwidth-bound: it round-trips the full ~84MB fp32 intermediate U (plus ~42MB of h) through global memory, which the production kernel never materializes (keeps U in WMMA accumulator registers/LDS, applies the epilogue before writeback). This is exactly GPT's own predicted failure mode for this branch of its decision tree.

Correctness: checked against a host double-precision reference of the identical formula (64 rows x 2560 cols x HC=4 sampled). rms_rel=0 (effectively exact); max_abs=0.140625 on a handful of outlier elements, traced to the device's fast __expf() vs host's double-precision exp() diverging at saturated sigmoid inputs -- not a structural bug (same formula, same operand order, same bf16 narrowing points as production). LIMITATION noted explicitly in the harness: this checks the epilogue's own arithmetic against the formula, not bit-identity against the real hc_up_mix_wide_kernel binary (which would need a full radiance build+link -- a separate, heavier harness not yet built).

CONCLUSION per GPT's own decision tree (combined >= 261us -> profile separately -> epilogue bandwidth-bound -> investigate eliminating the full intermediate via a custom fused GEMM, OR fall back to improving the existing kernel's WMMA scheduling in place): the naive hipBLASLt-GEMM-plus-separate-kernel approach is NOT a viable replacement as-is. Reported this result back to GPT (new session ses_c5c9f14085424707 after the prior reused session ses_0e45b277c600463c got permanently wedged on a stuck "running" request and was closed -- req_bigcherry_a3f02be7f186 cancelled, session closed, fresh session opened) asking for a concrete next step: either abandon the hipBLASLt angle and focus on improving hc_up_mix_wide_kernel's existing WMMA tile scheduling in place, or try a custom WMMA-based fused-epilogue kernel that keeps U in registers (skipping hipBLASLt entirely) before giving up on the GEMM-redesign angle. Response pending (req_bigcherry_46bf68059a11).

Both new harnesses are committed to tools/lab/radiance/ and registered in docs/reference/tooling/TOOL_DISPOSITION.md as TRANSITIONAL. Neither touches production code; RAD17's FP8 down-product patch (rad0007_hc_pf2_fp8_down) remains exactly as it was (default off, not promoted) -- this pivot is a separate, parallel investigation into the BF16 path, not a continuation of the FP8 work itself.

## Change Log

- 2026-10-21T07:41:01.749670+00:00 (created-by): Created by agent
- 2026-10-21T07:44:23.057374+00:00 (updated-by): Updated: section:notes
- 2026-10-21T07:48:10.715264+00:00 (updated-by): Updated: section:notes
- 2026-10-21T07:56:15.246271+00:00 (updated-by): Updated: section:detailed_solution
- 2026-10-21T07:59:14.580428+00:00 (updated-by): Updated: section:notes
- 2026-10-21T08:02:59.795128+00:00 (updated-by): Updated: section:notes
- 2026-10-21T08:05:29.531266+00:00 (updated-by): Updated: section:notes
- 2026-10-21T08:13:09.220715+00:00 (updated-by): Updated: section:notes
- 2026-10-21T19:39:39.409088+00:00 (updated-by): Updated: section:notes
- 2026-10-21T21:32:59.492280+00:00 (updated-by): Updated: section:notes
- 2026-10-21T21:46:55.745667+00:00 (updated-by): Updated: section:notes
- 2026-10-21T22:44:34.980663+00:00 (updated-by): Updated: section:notes
