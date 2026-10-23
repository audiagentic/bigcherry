---
id: RAD18
order: 0
plan: patching-radiance-rdna3
state: in_progress
created-at: '2026-10-21T23:17:12.131804+00:00'
breadth: ''
skill: ''
created-by: agent
---

# hc_up_mix_wide_kernel PF2 RT sweep (occupancy/register pressure)

## Description

Two prefill levers for Qwen3.8-Flash-Next in the working layout (R9700 runs the engine; the two XTX run r3_moe_worker helpers that hold the experts):

1. rad0008_hc_upmix_rt_sweep: launch the PF2 up-mix at 128 rows per workgroup instead of 256. DONE: promoted to validated in serving-core (PR #389).
2. R3_MOE_WORKER_COMBINE: helper 0 adds both helpers' bf16 rows on the CPU into plane [0][1], so the engine reads one plane instead of two (protocol v15). Not bit-identical (bf16 add order changes). Initial combine merged in #394; FP16/F16C refinement merged in #404 (still opt-in).

## Steps

1. [done] rad0008: RT sweep, kernel A/B on RDNA4 and RDNA3, in-engine --profile-ops A/B, end-to-end A/B, promotion (PR #389).
2. [done] Combine: implemented with blocking waits (futex, combine runs on helper 0's large-lane thread, generation in Request::combine_generation; commit 7902a1ae9); stop/fault paths never wait on a join (41f3cae42).
3. [done] Combine end-to-end A/B, 4 pairs (helper-ab layout): pp4096 3354 -> 3597 (+7.3%), pp1024 3404 -> 3648 (+7.1..8.1%), tg512/tg2048 flat, tg128 124.2 -> 110.4 (-11%). The tg128 drop is a content effect, not the combine: a control with the combine on but never firing gave tg128 124.3, the same as off. CPU combine cost about 300-340 us a step (176 steps, 53-60 ms a run).
4. [todo] Quality gate for the combine: paired KLD and top-1 agreement against the uncombined output (tools/lab/radiance/helper-kld.json), plus MTP draft acceptance per cell; tg128 content check on the same prompt both arms.
5. [todo] Resolve the post-#404 FP16 evidence gate in its own follow-up PR; retire the never-fire diagnostic only after it is measured and archived.
6. [todo] If the gate fails: keep the code on main as evaluated, default off, reason in SUMMARY (shelved-work rule).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/patches/rad0008_hc_upmix_rt_sweep/ (merged)
engines/radiance/components/moe-remote/r3_moe_worker.cpp, moe_lane_common.h, moe_remote_gpu.h (combine, branch feat/moe-remote-combine)
tools/lab/radiance/helper-ab.json, helper-kld.json, combine control config (branch)
docs/reference/engines/radiance/models/qwen3.8-flash-next.md (results)

## Validation

Combine: helper-kld.json paired KLD / top-1 within the engine's existing quantisation noise; MTP acceptance equal between arms; helper-ab.json 4 pairs, prefill gain holds, tg512/tg2048 not worse; component tests (r3_moe_shm_test, r3_moe_backend_test) pass.

## Effort & Risk



## Standards



## Acceptance Criteria

rad0008 validated in serving-core (met). Combine either merged default-on/selected with the quality gate passed and prefill gain confirmed, or shelved on main as evaluated with the reason recorded; results on the Flash-Next model page.

## Notes

Context: this is a sibling investigation to RAD17 (FP8 down-product), targeting the BF16 up-mix kernel instead. See RAD17's notes for the full GEMM/epilogue diagnostic history that led here. Patch: rad0008_hc_upmix_rt_sweep, default-off via RAD_HC_UPMIX_RT env var (unset = bit-identical to current production RT=kPf2UpRT=256).

2026-10-22. rad0008 (PF2 up-mix RT=128 default, RAD_HC_UPMIX_RT=256 restores the original): kernel level vs unpatched build -2.1%/-3.2% of hc_read on the R9700 (M=2048/4096), -5.8%/-2.8% on the XTX, byte-identical. END TO END in the working layout (R9700 engine + 2 XTX helpers, ab-benchmark tools/lab/radiance/helper-upmix-rt-ab.json, one binary built with --experiment helper-layout-upmix-rt, 4 rotated pairs, --host-pool-mib 22528 in both arms because the 30 GiB pool plus the pinned n-gram table no longer fits free host RAM): pp4096 3351 (RT256) vs 3355 (RT128) tok/s (+0.1%), pp1024 3406 vs 3416 excluding one low RT128 run (+0.3%), tg2048 152.3 vs 152.2 -- no measurable end-to-end gain. Matches the share: hc_read is 8.9% of R9700 prefill device time in this layout (--profile-ops, 7306-token prompt, TTFT 3.49 s), so 2-3% off hc_read is ~0.2-0.3% end to end. Prefill device time in this layout: moe_mailbox_add 51.7% (helper wait + merge), gemm_nt_q 11.1%, hc_read 8.9%, moe_mailbox_publish 6.2%, local moe_gemm_q 5.9%. Next: R3_MOE_WORKER_TRACE decomposition of the mailbox wait (GPT req_bigcherry_ceee9968e4b5): exposed helper wait vs helper GPU time vs merge (the merge reads each helper's full-width bf16 result straight from mapped host memory). Hardware layout is fixed (owner).
Prefill device-time split before these (--profile-ops, 7306 tokens): moe_mailbox_add 51.7%, gemm_nt_q 11.1%, hc_read 8.9%, mailbox_publish 6.2%, local moe_gemm_q 5.9%; TTFT 2.36 s; exposed helper wait 0.65 s. rad0008 in-engine hc_read 216.6 -> 210.0 ms (-3.0%); end-to-end below noise because hc_read is ~9% of prefill. Follow-ups split out: RAD19 (derive helper slice split), RAD20 (replace lane polling), RAD21 (micro-batch pipelining of the helper round trip).

## Notes (handoff)

Status 2026-10-23 (handoff): rad0008 promoted (done). Combine: bf16 row failed KLD (p99 +0.275); rounding-only control
proved it was the bf16 rounding; fp16 row passes KLD (mean +0.0027, p99 +0.039, top-1 -0.28 pt); scalar fp16 was too
slow (~1 ms CPU a step, prefill -2..-5%); AVX2/F16C restores prefill (+5-8% pp4096, +6-7% pp1024, CPU ~0.32 ms a step),
tg512/tg2048 flat, tg128 -21% still to confirm as a content effect. Code: branch feat/combine-fp16 (protocol 16,
combine_f16.h + r3_combine_f16_test, kCombineFallback marker, engine reads fp16). Runs on the build server:
${BC_WORK_ROOT}/runs/rad18-kld (reference), rad18-fp16 (KLD + scalar A/B), rad18-f16c (F16C A/B).
Remaining for RAD18: (a) tg128 check: helper-combine-nofire-ab.json + per-cell MTP acceptance, re-run the excluded
pair; (b) then retire the never-fire control config. Method selection, GPU method and N helpers continue in RAD23.

## Execution gate review (2026-10-23, post-merge)

- **Baseline correction:** the initial combine merged in #394; the FP16/F16C improvement merged in #404, default off. Neither should be reimplemented or reopened. This item remains **in progress** solely for the FP16/F16C evidence/acceptance close-out.
- **RAD18 follow-up slice:** use the already merged binary (fixed SHA/build ID); rerun tg128 off/on and the never-fire control on an identical prompt/seed; capture generated-token counts, accepted MTP draft counts/rates **per cell**, and step/latency measurements. Re-run the previously excluded A/B pair rather than dropping an outlier. Use the existing lab/AB harness and resource queue; do not run ad-hoc concurrent GPU workloads.
- **Quality/throughput proof:** archive paired KL mean/p99, top-1, continuation hashes, fixed-text decode latency, pp1024/4096, tg128/512/2048 and arm order. Prior FP16 p99 delta +0.039 is supporting evidence, not a substitute for this run. A divergent continuation or draft-acceptance rate may explain tok/s changes but is not a pure performance win.
- **Finish condition:** if quality and controlled performance hold, update the model evidence, retire the diagnostic never-fire config in the RAD18 follow-up PR, and close RAD18. If any gate fails, leave combine off, record a negative result, and do not promote. Keep one new PR for this evidence/cleanup slice, separate from merged #404 and RAD20/RAD23.

Handoff 2026-10-24: code merged (#404). Remaining is only step (a)/(b) above: tg128 never-fire control + MTP
acceptance on the build server (bigcherry slice lab), then retire helper-combine-nofire-ab.json.

## Change Log

- 2026-10-21T23:17:12.131804+00:00 (created-by): Created by agent
- 2026-10-22T07:32:45.588047+00:00 (updated-by): Updated: section:notes
- 2026-10-23T02:13:51.871271+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:notes
- 2026-10-23T02:14:52.569644+00:00 (state-transition): State: pending → in_progress
- 2026-10-23T02:15:07.427515+00:00 (updated-by): Updated: section:acceptance_criteria
