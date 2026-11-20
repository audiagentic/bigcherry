---
id: PVPS15
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-29T22:19:04.336861+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Producers must prove the patch fires BEFORE timed sessions

## Description

Every real producer uses trace_probe="skip" and decides activation itself, after its paired timed benchmarks (1206 rd13 runs run_paired_llama_benchmark before run_trace_probe). A model or config where the patch does not fire therefore burns the full timed run and only then reports not_executed. Found 2026-09-30: lab A/Bs of rd13, 1245 and 1263 on the 27B were run without proving they fire; the 1245 gate (ncols==6, MTP n_max=5) cannot fire at n_max=4. The trace_probe="run" dispatcher mode (probe after producer) is used by no real patch.

## Steps

1. Add a runtime helper (ProducerRuntime.require_fires) that runs the two-probe activation check and raises ValidationProducerError when status is not executed.
2. Call it first in every producer that declares a trace-marker check (1204, 1206, 1237, 1265, 1241, 1253, 1254), before any timed lane.
3. Give patches without a marker (1245) a trace marker or an explicit declared scope (n_max=5).
4. Declare per-patch test requirements (target model, n_max or layout, marker) in validation.toml so a campaign cannot be planned on a model that cannot fire it.
5. Update tests that encode probe failure as a persisted record (rd13 test_probe_failure_still_exit_zero_with_failed_record) and decide whether a failed pre-flight still persists a BLOCKED record.

### Inherited workload-contribution profiling gate from PVPS10

After paired verdict lanes, run an untimed rocprofv3 kernel trace for control and subject on the declared positive-lane workload. Declare patched kernel families/symbol patterns and required workload in validation.toml `[profile]`; `bigcherry kernel-fraction` records per-kernel calls, GPU time and share. Require the subject's declared kernels to appear and the control to show the pre-patch path or no patched kernels. Run the scaffold tune build with 0200/0700 coverage counters once and retain exact per-path hits per token. Record expected-effect bound = patched-kernel share × measured kernel-only change. If patched share is below a declared floor, mark the session INVALID (not FAIL); if kernel gain is large but end-to-end share is small, report that the effect is not measurable on this workload and require a contract/workload review. Publish top kernels, occupancy, launch counts and neighboring-kernel regressions with the promotion evidence.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/bigcherry/patch/campaign/producer.py, patches/*/validation/producer.py, tools/tests/patch/test_patch_validation_campaign_rd13_contract_cli.py, tools/lab/plan-qualification/queue.sh

## Validation

Unit test: a producer whose marker never fires aborts before any timed lane runs and never calls run_paired_llama_benchmark. Interim queue-level gate: PREFLIGHT and REQUIRES= rows in queue.sh (commit 6fd507e2).

### Inherited PVPS10 validation

Offline tests cover kernel-fraction parsing, declared-kernel matching, hit counts and INVALID-versus-FAIL classification. Future hardware traces cover 1256/1257, 1263, 1262 and 1265 gfx1030; known winners 1237/1253 should show patched-kernel share consistent with their observed gains. This handoff performs no GPU profiling.

## Effort & Risk



## Standards



## Acceptance Criteria

Before performance interpretation or promotion, each patch declares the expected kernel family and positive-lane workload, has a control/subject trace and exact coverage counts, and includes a measured kernel-only effect and share-scaled expectation. Under-floor share is INVALID, never FAIL; a low end-to-end contribution despite a large kernel gain triggers workload review. Profile artifacts include neighboring-kernel effects and do not replace correctness, activation or paired end-to-end gates.

## Notes

Interim mitigation landed: queue.sh PREFLIGHT rows, REQUIRES= gate, tools/lab/native-vs-patched/preflight-fire.sh. 1245 ncols=6 Q8_0 fused kernel measured 60 VGPR, no spill, no scratch, so the register-pressure explanation for its -6.4% at n_max=5 is ruled out.

2026-09-30 progress: steps 1-2 done. Instead of a new runtime API, added vp.ValidationProducerBlocked, vp.require_activation_ok and vp.require_fires; producers 1204/1206/1237/1265/1241/1253/1254 now abort (non-zero, no record, no timed lane) when the marker does not fire. 1204 and 1206 reordered so the probe runs before any timed lane. Step 3: 1245 REJECTED instead of instrumented (-6.4% at n_max=5, identical acceptance; spill ruled out). REMAINING: step 4 (per-patch firing requirements in validation.toml, e.g. 1263 single-GPU, 1245-style n_max scope), persisted BLOCKED verdict + policy handling in patch-verify-evidence (GPT req_f8ad9fe280fd47eb), rd13 27B activation preflight run on dev. Note: the rd13 test test_probe_failure_still_exit_zero_with_failed_record covers the dispatcher trace_probe=run mode, which no real patch uses; it is unchanged.

2026-09-30 27B dual-XTX findings: rd13 (1206) firing on Qwen3.8-27B Q8_0 is PROVEN (preflight-rd13-27b: BIGCHERRY_PATCH_HIT patch=1206_rd13 path=mul_mat_add_view_fusion_q, hits=1), and the 4-round paired A/B on 27B -sm tensor (rd13-ab1, mtp-dual) shows no effect: pp1024 -0.03%, pp4096 -0.03%, tg512 +0.02%, tg2048 +0.06%, all CIs straddle 0. So rd13 fires but is neutral on 27B; the 4B contract passes do not transfer. 1254 (nro05) shows no gain on gfx1100 in 5 sessions. 1263 is single-GPU only (aborts under -sm tensor). None of these three helps the dual-XTX 27B production config; no lifecycle mutation made (lifecycle decisions remain explicit).

2026-09-30 BLOCKED handling done: ValidationProducerBlocked now propagates unwrapped from execute_validation_producer; _run_validation_producer catches it, writes run_dir/producer-blocked.json (verdict=BLOCKED, reason, timed_lanes_run=false, evidence_record=null), prints it, and returns exit 3 (BLOCKED_EXIT_CODE). Deliberately NO tracked validation record is persisted for a BLOCKED run, so it cannot satisfy improvement_no_regression_v1 / patch-verify-evidence (neither PASS nor FAIL, ineligible for promotion). Tests: ProducerBlockedDispatchTests. Remaining: per-patch firing requirements in validation.toml (1263 marker, 0860 provider marker), ProducerRuntime.require_activation API to replace per-producer require_fires plumbing, an end-to-end test of the exit-3 path, rd13 27B requirement in validation.toml (fires; neutral).

Consolidation update 2026-11-20: PVPS10 profiling follows PVPS15's actual-patch-activation gate. Activation is proved before timed lanes; contribution profiling follows the verdict lanes and classifies workload coverage before interpreting end-to-end effect. Source remains incomplete until the offline and hardware validations above are recorded.

## Change Log

- 2026-09-29T22:19:04.336861+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260929_222105_patch-qualification-queues-can_5436
- 2026-09-29T22:21:08.894835+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-30T00:04:54.693739+00:00 (updated-by): Updated: section:notes
- chg_20260930_000454_qualification-runs-no-longer-s_6036
- 2026-09-30T00:04:57.988566+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-30T00:17:47.477124+00:00 (updated-by): Updated: section:notes
- 2026-09-30T01:03:14.283546+00:00 (updated-by): Updated: section:notes
- chg_20260930_010324_validation-campaigns-that-cann_9359
- 2026-09-30T01:03:27.769967+00:00 (updated-by): Updated: section:ledger-events
- chg_20260930_021419_builds-firing-checks-and-ab_8646
- 2026-09-30T02:14:26.106622+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-20T04:10:01.939133+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
