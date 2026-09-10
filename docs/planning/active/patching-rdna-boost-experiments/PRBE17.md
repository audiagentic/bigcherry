---
id: PRBE17
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:37.108784+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Defer HIP integrated-GPU host-buffer backout

## Description

Retain the HIP integrated-GPU host-buffer backout as deferred hardware-scoped work; never apply it globally to discrete production.

## Steps

- Confirm a relevant integrated-GPU target exists before implementation.
- Reproduce asynchronous host-buffer behavior and compare blocking/non-blocking and load modes.
- Check PPL/output integrity and host-buffer ownership/lifetime.
- Keep discrete GPUs on current behavior and run explicit non-selection controls.
- If target hardware remains unavailable, retain deferred disposition rather than infer from discrete results.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

HIP host-buffer policy source; integrated-GPU fixture; async/blocking/load and PPL/output tests; ownership/lifetime evidence; discrete-GPU non-selection controls.

## Validation

Integrated GPU only; async correctness; PPL/output; blocking/non-blocking; load; ownership; discrete non-regression.

## Effort & Risk



## Standards

Hardware-scoped change; no global workaround without evidence.

## Acceptance Criteria

Promotion requires affirmative integrated-GPU evidence with no discrete regression; until then this remains explicitly deferred and no global workaround is introduced.

## Notes

Supersedes: RD22
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd22

## Change Log

- 2026-09-09T10:54:37.108784+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:50.083646+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.205711+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.915767+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:50:24.214993+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025049_rdna-successors-prbe1719-now_5726
- 2026-09-10T02:50:49.204728+00:00 (updated-by): Updated: section:ledger-events
