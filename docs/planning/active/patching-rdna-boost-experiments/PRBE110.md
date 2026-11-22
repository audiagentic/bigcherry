---
id: PRBE110
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-23T09:19:09.142442+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# RD07 Q6_K MMQ scale fold — b11552 source and admission gate

## Description



## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

**Transferred PRBE04 acceptance (retain as separate gating outcome):** After PRBE110's source/route admission on current b11552 (1267 is known_broken, historic b11126 +2.4675% gfx1201 is not current qualification), compose only existing 1267 with 0300 J selection and 1006 Q6_K codegen; no new dispatcher/J knob. Carry PRBE04's PEF01/HI71 downstream safety gate: J packing/padding, allocation and bounds, activation, exact Q6_K/other-quant numerical and graph-replay controls. Only if safely admitted on gfx1201 run four independent sessions with >=10 paired ABBA rounds/session pp512/pp2048/tg128 and other-quant controls; require pp512 CI95-low >=2%, representative E2E CI95-low >=3%, <=1% unrelated regression. Negative outcomes are explicit NO_RELEVANT_Q6K, SOURCE_REBASE_BLOCKED, SAFETY_OR_NUMERIC_FAIL, NO_EXPOSED_GAIN and NO_VALIDATED_GAIN; leave patch unpromoted unless all pass.

## Notes

2026-11-22 owner handoff from decision-only BCOP211: Unsectioned source gate. Source-bounded action/stop rule: **Discovery/change:** 1267's Q6_K dispatch-marker and `mmq.cuh` J-max anchors no longer exist on pinned b11552; 0300 now chooses forced J before activation padding. Repository `engines/llamacpp/dispositions/1267_*.json` already marks `known_broken`. PRBE04 still claimed PRBE110 had not delivered the package despite b11126 hardware receipts. Archived gfx1201 pp512 +2.4675% CI95 [2.2776,2.6508], but validation record remains incomplete/ineligible despite a contract-pass field; no new GPU measurement. **Owner/consolidation:** PRBE110/1267 owns the one patch and architecture gate. PRBE04 owns PEF01/HI71 safety and final acceptance. 0300 owns J/eligibility; 1006 owns Q6_K codegen. No new MMQ dispatcher, J knob, allocator, scheduler, cache, telemetry or hardware queue. Existing Radiance RAD02/RAD03/RAD04 work untouched. **12-hour/novelty:** selected branch `feat/rad03-gpu-lanes`, independent HEAD `42388e880e089668db3c4fdbf1ff7aeb25ef62b3` (2026-10-14 13:43 UTC). Latest independent PRBE110/PRBE04 plan change 2026-10-08; 1267 relocation 2026-10-09; no independent related work/PR/queue in preceding 12 hours. BCOP190–210 do not own Q6_K scale-fold. **Unresolved action:** first prove executed See docs/planning/completed/patching-bc-optimizations/BCOP211.md for the complete source/shape/negative case and evidence. PRBE110 is the only active implementation owner; this handoff does not qualify a kernel, change a source pin or authorize extra dispatch code.

2026-11-22 final BCOP authority reconciliation (BCOP115): 1267 Q6_K MMQ subscale-fold is known_broken/untested on b11552; first prove a real current-pin Q6_K route and >=3% exposed opportunity, then compose existing 0300/1006 and validate J padding/allocation plus PEF01/HI71. Old b11126 +2.4675% gfx1201 cannot qualify a new source. If no relevant route/ceiling, close 1267 unpromoted, with no second Q6_K J/selector patch. Original source and reviewed decisions: docs/evidence/2026-11-22-bcop-final-disposition/README.md

CONSOLIDATION 2026-11-22 PRBE-QFP: receiving PRBE04. Prior acceptance remains mandatory; this is a planning ownership consolidation, not an executed implementation or evidence/quality pass. Independent hardware routes and production-safe fallbacks remain conditional. Full original design, source/negative receipts and historical review survive in each archived source plan.

## Authoritative decision (2026-10-14)

**Pending / 1267 remains `untested` and `known_broken` on b11552. Do not adopt, queue a GPU campaign, or resurrect 1203.** The tracked `engines/llamacpp/dispositions/1267_rd07_q6k_mmq_scale_fold.json` already records `FAILED_NEEDS_RECONCILIATION` against pinned llama.cpp `23b0202a189c44a54625aadcb37a946dd1d6278d`. The existing PRBE04 prose incorrectly says PRBE110 has not delivered the package: 1267 exists and was hardware-exercised on b11126. This item owns only the 1267 source/rebase and architecture-selection decision. PRBE04 owns downstream PEF01/HI71 safety qualification.

**12-hour protection:** no independent PRBE110/PRBE04/1267 implementation, plan, evidence, PR or queued experiment was found in the 12 hours preceding this audit (latest relevant independent plan commit 2026-10-08 06:47 UTC; package/evidence relocation 2026-10-09 04:20 UTC). Active RAD02/RAD03 helper/mailbox and RAD04 two-half prefill remain untouched. Recent BCOP190–210 do not own this Q6_K scale-fold path.

## Implementation and failure trace

1. Pinned `ggml/src/ggml-cuda/mmq.cu::ggml_cuda_mul_mat_q_switch_type` dispatches Q6_K as `mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ctx, args, stream)`. Current 0300/HI06 chooses an eligible `J_best` **before** MMQ activation padding and workspace allocation, with `GGML_HIP_MMQ_FORCE_J` for HIP diagnostics; it deliberately does not thread `forced_J` through `mul_mat_q_case` or `mul_mat_q_switch_J`.
2. 1267 `patch.py` still inserts its activation marker before the obsolete `mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream, forced_J)` call. Its `rd07-jmax-env` edit also anchors on `int ret = std::min(ne11, int64_t(512)); ret -= ret % 8;` in `mmq.cuh`; that old J-max selector is absent at b11552. **Both are exact missing anchors.** Merely changing the marker is insufficient to make the package applicable.
3. The three Q6_K kernel edits still match the source mechanism in `ggml/src/ggml-cuda/mmq-vec-dot.cuh::ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma`: hoist `x_df` per output element outside `k01`; load `x_sc` and fold `x_sc * x_df` outside the `j0` loop; accumulate `(float) C.x[l] * x_s2_reg[n][l] * dB`. Existing code instead accumulates `C.x[l] * sc[k01/4] * x_df[i*sram_stride] * dB`. This reduces repeated scale work but **changes floating-point association**. It does not remove a matrix product, output row or required accumulation.
4. The MMA branch is shared by `AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE`; the present 1267 kernel edit has no gfx1201-only guard. A default-on patch would affect gfx1100 and gfx1030 code paths too, despite their negative/neutral historical receipts. Qualify a compile-time architecture gate using the existing `RDNA4`/backend configuration and preserve stock arithmetic for unqualified devices. Do not add a new MMQ selector, J table or runtime dispatcher. The exact gfx1201 code-object gate must be verified; `RDNA4` alone is a family, not proof of every member.
5. The patch's `GGML_CUDA_MMQ_J_MAX` knob is redundant/unsafe to re-anchor at a late selection point. **Remove that edit from the candidate** and reuse 0300's preallocation `J_best`/eligibility path. Preserve PEF01's gfx1100 Q6_K J=112/fallback=0 quarantine; never use an unqualified forced-J to bypass it.

**Host-only numerical discriminator:** 200,000 bounded int32-C/int8-subscale/F32-scale/F32-activation samples, evaluated in explicit F32 arithmetic, yielded 38,735 bit-different results between the old and folded association (maximum absolute difference 0.0625). This proves general host expression non-equivalence, **not** an observed GPU/logit failure. A 12-predicate live-source static audit passed; no C++ compilation, HIP run, patch application or GPU benchmark was performed.

## Historical measured evidence — do not re-label as current

At **b11126**, `1267/evidence/validation.json` contains four-session gfx1201 evidence: pp512 +2.4675%, CI95 [2.2776%, 2.6508%]; tg128 -0.0331%, CI95 [-0.0592%, -0.0071%]. Gfx1100 and gfx1030 completed four-session campaigns without established prefill gain. The gfx1201 record's **contract verdict says pass**, but its `validation_disposition` remains `incomplete`, `eligible_for_validated_state=false`, and the `performance`/`controls` checks report fail. These fields must be reconciled against the authoritative contract evaluator before promotion; do not silently equate a contract-pass field with validated state. The archived full-vocabulary check reports 12,869,632 log-probabilities, max KL/material difference zero, but does not establish current-pin or all-route bit identity. The rejected 1203 bundle's older +3.4%/+6.2% numbers are non-transferable.

**Production applicability:** current Flash-Next IQ4_XS/Q8_0 files do not exercise Q6_K weight MMQ. A standalone Q6_K model or executed Q6_K tensors are required to justify any deployment-specific benefit. A single-R9700 pp512 improvement is not a three-GPU Flash-Next throughput claim.

## One bounded implementation path (only if admitted)

- **Gate 0 — actual route and opportunity:** record exact Q6_K GGUF tensor types, physical GPU and `ggml_cuda_mul_mat_q` MMQ/BLAS dispatch at the tested M/N/K, `J_best`, fallback, activation Q8_1 layout and graph mode. Estimate the *exposed* model E2E ceiling from kernel time and critical-path overlap; close `NO_RELEVANT_Q6K` if no representative Q6_K lane, `NO_EXPOSED_GAIN` if optimistic E2E <3%. No speculative port or four-session queue before this.
- **Gate 1 — one existing-package rebase:** update only 1267's marker to the pinned `<GGML_TYPE_Q6_K, GGML_PREC_Q8>` dispatch, drop the stale J-max edit/knob, retain the three kernel transformations and Q6_K-only backend-op cases, and gate the transformed branch to qualified gfx1201 code objects. Keep `patch.toml` `untested`, opt-in and 1203 conflict. Re-run `patch-lint`, `patch-rebase-check`, missing/duplicate-anchor negative controls, and composition with 0300 and 1006/related Q6_K codegen owners. Do not alter 0300.
- **Gate 2 — cheapest correctness:** host scale/address oracle (including tail rows and all legal J), float32 association/logit tolerance, reference Q6_K MMQ backend ops, graph on/off, multi-ubatch and repeated same-process requests. Require subject-only marker, actual executed Q6_K kernel, full-vocab/KLD/greedy and unchanged output/work accounting. PEF01 quarantine and HI71 launchability must remain effective on gfx1100.
- **Gate 3 — measured gfx1201 lane only:** if prior gates pass, one existing validation producer, matched control and candidate with same pin/model/quant/physical placement. Four independent sessions, >=10 paired rounds each, pp512 and pp2048, tg128 and non-Q6_K controls, short/24K/80K when supported, GPU kernel/launch/VRAM counters. Promote only with current-pin pp512 CI95-low >=2%, controls <=1% regression, no safety/numerical failure and >=3% exposed end-to-end gain for any production feature. Otherwise `NO_VALIDATED_GAIN` or `NOT_PRODUCTION_RELEVANT`; leave 1267 untested/parked.

## Ownership and external comparison

PRBE110/1267 is the **sole** RD07 code/architecture owner; PRBE04 owns the downstream PEF01/HI71 and campaign acceptance fence. `0300_mmq_forced_j` owns preallocation J/eligibility; existing MMQ configs own tile geometry; 1006 owns Q6_K codegen/casts; PEF01 owns the gfx1100 J112 quarantine. Do not fork any of these. Upstream llama.cpp master (inspected 2026-10-14) still uses the unhoisted Q6_K MMA expression. The current stew675 RDNA-boosts block 07 carries the same scale-fold mechanism, but moves `GGML_CUDA_MMQ_J_MAX` to the native **preallocation J search**; do not port its knob on top of 0300. SGLang's `gguf.py::fused_mul_mat_gguf` routes Q6_K to its GGUF MMQ when available, but its implementation/AMD performance is not a BigCherry gfx1201 receipt.

References: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq-vec-dot.cuh ; https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/patches/0007-rdna-boosts-block-07-k-quant-boosts-mmvq-VDR-Q6_K-MM.patch ; https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/gguf.py .

## Terminal disposition

Keep `pending` until Gate 0. Outcomes: `NO_RELEVANT_Q6K`, `NO_EXPOSED_GAIN`, `SOURCE_REBASE_BLOCKED`, `SAFETY_OR_NUMERIC_FAIL`, `NO_VALIDATED_GAIN`, `NOT_PRODUCTION_RELEVANT`, or `PROMOTABLE`. No new patch, hardware experiment, scheduler, cache, allocator, dispatcher or telemetry system was created by this audit.

## Change Log

- 2026-11-22T05:23:21.770716+00:00 (updated-by): Updated: section:notes
- 2026-11-22T06:45:50.153211+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4482
- 2026-11-22T09:43:04.662940+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
