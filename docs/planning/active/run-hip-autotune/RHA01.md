---
id: RHA01
order: 3
plan: run-hip-autotune
state: pending
created-at: '2026-09-09T10:48:34.167227+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: P2
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

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): GO after a tiny contract freeze — design is settled enough to implement now. Freeze before coding: (1) configured direct perf binary path + explicit sudo policy (per this item's own prior verified findings: /usr/lib/linux-tools-6.8.0-139/perf with sudo); (2) normalized perf artifact schema including tool-version/command capture; (3) a failed sampling pass is failed diagnostic evidence, never silently replaced by the control run. Keep perf and rocprofv3 mutually exclusive per the existing design. THA16 (dispatch-resolver-overhead question) confirmed as the right first real validation target. Execution order: ranked #3.

## Change Log

- 2026-09-09T10:48:34.167227+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:37.889270+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.818915+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T21:02:58.731913+00:00 (updated-by): Updated: section:validation, section:notes
- chg_20260909_210309_the-only-remaining-active-plan_2565
- 2026-09-09T21:03:09.280906+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:32.121390+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:04.125826+00:00 (updated-by): Updated: order=3, priority='P2'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.930653+00:00 (updated-by): Updated: section:ledger-events
