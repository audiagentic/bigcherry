---
id: RHA01
order: 0
plan: run-hip-autotune
state: pending
created-at: '2026-09-09T10:48:34.167227+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: null
---

# profile-campaign CPU call-graph capability via perf (enhancement, deferred -- perf currently non-functional on Brutus)

## Description

Perf availability was unblocked, but perf.py integration and real-target CPU call-graph validation are explicitly not done.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: run

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/run-hip-autotune-hi133.md

## Validation

Deferred by explicit capability constraint: no usable perf event source on Brutus. When resumed, require a perf.py integration test plus a real-target CPU call-graph capture with provenance; absence of perf data is not a pass.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: HI133
Migration: capability-rebaseline-v3-2026-09
Successor key: run-hip-autotune-hi133

Evaluated against the completed RHA04/RHA10/RHA11 production admission path. This is an optional profiling enhancement, not a prerequisite for the parity/admission mission. Brutus currently exposes no usable perf events for the requested CPU call graph; implementing perf.py integration without a functioning target would produce no decision-grade evidence. Leave pending/deferred until a perf-capable host or kernel configuration is available; do not reopen the completed GPU admission items.

## Change Log

- 2026-09-09T10:48:34.167227+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:37.889270+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.818915+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T21:02:58.731913+00:00 (updated-by): Updated: section:validation, section:notes
- chg_20260909_210309_the-only-remaining-active-plan_2565
- 2026-09-09T21:03:09.280906+00:00 (updated-by): Updated: section:ledger-events
