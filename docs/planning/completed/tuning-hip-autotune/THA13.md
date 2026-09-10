---
id: THA13
order: 0
plan: tuning-hip-autotune
state: completed
created-at: '2026-09-09T10:49:31.810764+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Separate fast runtime cache identity from the persistent cryptographic digest

## Description

Frozen HI163 explicitly records that the fixed-source retest did not shrink the replay host-side gap and says the real cause remains open; it is not terminal despite the independent terminal suggestion.

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

successor-specs/tuning-hip-autotune-hi163.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: HI64.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (6); preserve historical references (1) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: HI163
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi163

CLOSURE-WITH-CAVEAT 2026-09-10, verified by direct source/history inspection (via subagent, to keep the large diff out of primary context). The code fix is DONE and CORRECT: commit 4295e05e rekeys g_bindings to a fingerprint-keyed unordered_map<uint64_t, vector<L2Entry>>, moves signature_digest/dispatch_digest computation to the genuine L2-miss point only (the cheap FNV-1a fingerprint now keys both L1 and L2, matching HI163's own design), plus a dedicated test (tools/tests/tuning/test_l2_cache_collision_safety.py). HI163.md's own 'IMPLEMENTED 2026-09-09' note confirms this matches the design exactly.

BUT: real hardware evidence ALREADY EXISTS via HI163's own 'RETEST RESULT 2026-09-09' section, and it is a documented NEGATIVE result -- the gap did NOT shrink. Real rebuild-and-swap retest on Brutus (verified the rebuilt .so was actually mapped via /proc/PID/maps): native mean 9.374us vs replay(fixed) mean 9.971us across 65,175 calls/arm -- replay is STILL +6.36% slower (vs +5.2% before this fix). HI163's own interpretation: this workload's steady-state decode hits L1 almost every dispatch, so L2's keying mechanism (what this fix touches) rarely executes in the measured window -- the fix is real and worth keeping for thrashing workloads, but is NOT the dominant contributor to the observed +5-6% hipLaunchKernel host-side gap. HI163 explicitly: 'That regression's real cause is still open... its promotion should NOT be described as having fixed GPU0's e2e regression.'

DISPOSITION: closing THA13 as its own stated scope (separate fast runtime cache identity from the persistent cryptographic digest) is COMPLETE and CORRECTLY IMPLEMENTED, with real hardware evidence already attached via HI163 -- but explicitly NOT claiming this resolved the replay host-side gap, because it demonstrably did not. The still-open replay-gap investigation is a DIFFERENT question than THA13's own scope and should not be conflated with it going forward. Per HI163's own notes, candidates for that separate investigation include an extra branch/atomic/bookkeeping difference in ggml_hip_dispatch_resolve()/launch(), or a GGML_HIP_DISPATCH_REPLAY compile-time effect unrelated to cache keying -- worth a fresh plan item if pursued, not a reopening of this one.

## Change Log

- 2026-09-09T10:49:31.810764+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:41.474973+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.887487+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.418979+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T01:40:50.494776+00:00 (updated-by): Updated: section:notes
- 2026-09-10T01:40:50.907667+00:00 (state-transition): State: pending → completed
- chg_20260910_014055_closed-the-l2-cache-keying-fix_4531
- 2026-09-10T01:40:55.984842+00:00 (updated-by): Updated: section:ledger-events
