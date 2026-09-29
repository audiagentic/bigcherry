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

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/bigcherry/patch/campaign/producer.py, patches/*/validation/producer.py, tools/tests/patch/test_patch_validation_campaign_rd13_contract_cli.py, tools/lab/plan-qualification/queue.sh

## Validation

Unit test: a producer whose marker never fires aborts before any timed lane runs and never calls run_paired_llama_benchmark. Interim queue-level gate: PREFLIGHT and REQUIRES= rows in queue.sh (commit 6fd507e2).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Interim mitigation landed: queue.sh PREFLIGHT rows, REQUIRES= gate, tools/lab/native-vs-patched/preflight-fire.sh. 1245 ncols=6 Q8_0 fused kernel measured 60 VGPR, no spill, no scratch, so the register-pressure explanation for its -6.4% at n_max=5 is ruled out.

## Change Log

- 2026-09-29T22:19:04.336861+00:00 (created-by): Created by agent
