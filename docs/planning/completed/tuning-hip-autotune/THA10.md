---
id: THA10
order: 0
plan: tuning-hip-autotune
state: completed
created-at: '2026-09-09T10:49:18.675456+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P0
---

# Diagnostics are not zero-cost: enabled() does an atomic RMW on every call

## Description

Diagnostic atomic-removal changes landed, but the frozen item still specifies audit/compile-out/measurement validation and contains no completed acceptance record.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: tuning

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/tuning-hip-autotune-hi159.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: none extracted.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (1); preserve historical references (0) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: HI159
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi159

CLOSURE 2026-09-10. Audit (step 2) done and clean: verified by direct source inspection that dispatch_counters_enabled(), native_select_timing_enabled() (hip-autotune-dispatch.cu), and trace_launch_attempt (hip-autotune-tuner.cu) all use the magic-static pattern from HI159's fix (commit e8142e30), not the old checked.exchange atomic-RMW bug.

Key finding via git history that changes the closure argument: the #ifndef GGML_HIP_DISPATCH_DIAGNOSTICS compile-out gate was added in a SEPARATE, LATER commit (4125a6f3, "Strip diagnostics out of the production build, not merely switch them off") -- not part of e8142e30 itself. So the real production-relevant history is: (1) e8142e30 fixed the atomic-RMW bug but the function still ran at runtime in every production build (fast static-bool read, but still executed); (2) 4125a6f3 later removed the function from production builds ENTIRELY via compile-time exclusion. Current HEAD has both.

Empirically confirmed on Brutus: `strings ~/bc-build-rccl/bin/llama-server | grep -c 'GGML_HIP_DISPATCH_COUNTERS\|GGML_HIP_NATIVE_SELECT_TIMING'` returns 0 -- the diagnostic code is not merely disabled at runtime, it is ABSENT from the compiled production binary. This is by-construction zero cost, not a statistically-inferred zero cost.

DECISION: did not run a full hardware tg balanced A/B campaign (building a pre-e8142e30 binary, deploying, running paired interleaved rounds). Two dev-gpt requests for the exact harness command both failed with gateway INT-AGW-098 initial-activity-timeout (gateway capacity contention, not a real answer failure) -- did not retry a third time. Rather than spend a full build+campaign budget chasing a comparison whose 'after' state has zero bytes of the relevant code in the binary (a tg A/B could only ever return HI159's own accepted null result, never anything else, since there is nothing left to measure), treating the binary-absence evidence as sufficient closure. This is consistent with HI159's own validation note ('Expect tg movement or nothing; a null result here is a legitimate outcome and must be reported as such') -- the null result is proven by construction here rather than by a statistical test failing to detect a nonzero effect.

STATUS: audit clean, compile-out verified empirically, atomic-RMW fix confirmed landed. Closing as satisfied. If a future session wants the full paired-build hardware campaign anyway (e.g. to also validate the magic-static fix's effect during the e8142e30..4125a6f3 window when the code still ran but was cheap), that remains a legitimate follow-on but is not required for THA10's own stated closure bar.

## Change Log

- 2026-09-09T10:49:18.675456+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:25.203556+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.872504+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.397460+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T01:39:30.452547+00:00 (updated-by): Updated: section:notes
- 2026-09-10T01:39:30.826583+00:00 (state-transition): State: pending → completed
- chg_20260910_013936_closed-the-diagnostic-overhead_1190
- 2026-09-10T01:39:36.478076+00:00 (updated-by): Updated: section:ledger-events
