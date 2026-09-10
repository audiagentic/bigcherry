---
id: PEF01
order: 0
plan: patching-external-fixes
state: pending
created-at: '2026-09-09T10:47:40.567257+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# MMQ q6_K/J=112 illegal-memory-access incident on gfx1100 -- quarantined, root cause open

## Description

Maintain the exact gfx1100 Q6_K J=112/fallback=0 quarantine while preserving an open root-cause investigation. Clean generic runs do not close the incident.

## Steps

- Accept only recovered original canonical signature/digest or complete attempt record as historical identity evidence; candidate-only matches are insufficient.
- If exact identity is recovered, replay through the exhaustive tuner path with quarantine bypass, HIP_LAUNCH_BLOCKING=1, graphs disabled, tracing and rocgdb; otherwise do not run generic sweeps.
- Identify the faulting instruction/address and distinguish kernel, workspace/allocator and harness causes before changing eligibility.
- Apply a scoped fix only after root-cause proof, then rerun exact trigger, native control and fresh full tune with zero errors/failures/attribution gaps.
- Keep the diagnostic bypass opt-in and exact-candidate scoped; never widen or remove quarantine from non-reproduction evidence.

## Detailed Solution & Technical Design

The current mitigation quarantines only src0 Q6_K, primary J=112, fallback=0 on gfx1100. Preserve this narrow containment. Treat the reopened Y-staging-overread theory from HI71 as a hypothesis requiring exact-tail-vs-available-tail proof, not as closure or permission to generalize.

## Code Samples & Guidance



## Files

src/ggml/src/ggml-cuda/hip-autotune-dispatch.cu; hip-autotune-journal.{h,cpp}; hip-autotune-tuner.cu; dispatch/replay tests; docs/reference/FINDINGS.md; exact historical attempt artifact if recovered.

## Validation

Exact identity recovery conditional; rocgdb/faulting instruction; HIP launch-blocking/graphs-disabled replay; native control; fresh full tune; zero HIP/assertion/measurement/attribution failures; retained negative/non-reproduction evidence.

## Effort & Risk



## Standards

Fail closed on safety incidents; exact candidate identity; provenance-bound diagnostics; no generic sweep substitution.

## Acceptance Criteria

Quarantine remains active until exact candidate-level fault identity and a verified fix exist. Clean generic or alternate-model runs are non-reproduction evidence only; no broad eligibility change is allowed.

## Notes

Supersedes: EX02
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-external-fixes-ex02

## Change Log

- 2026-09-09T10:47:40.567257+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:36.780012+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.758728+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.209055+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:41:10.569527+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_024129_build-and-external-fix-success_1105
- 2026-09-10T02:41:29.980880+00:00 (updated-by): Updated: section:ledger-events
