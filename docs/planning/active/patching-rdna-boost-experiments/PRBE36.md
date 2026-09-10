---
id: PRBE36
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:57.373909+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-STREAM-006: Enable graph-opt by default on RDNA3.5

## Description

Consider graph-opt default-on for gfx1151 only after the complete AMD-STREAM prerequisite chain passes; current hardware is not sufficient.

## Steps

- Require PRBE32, PRBE33, PRBE34 and PRBE35 validated identities.
- Keep default change scoped to gfx1151 architecture; gfx1100/gfx1201 behavior must remain unaffected.
- Run broad dense/MoE/GDN and MTP on/off regression suite with capture stress and long-context edge cases.
- Compare correctness, capture failures, output divergence and performance distribution across models, not a single winner.
- If hardware-wide confidence is absent, retain graph-opt opt-in and document the blocking evidence.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

Architecture default configuration; prerequisite identities; gfx1151 broad regression recipes; dense/MoE/GDN/MTP/capture evidence; non-selection controls.

## Validation

Full prerequisite chain; gfx1151 hardware; broad model suite; graph capture stress; long context; correctness/output; performance distribution; non-gfx1151 controls.

## Effort & Risk



## Standards

Last-in-chain policy change; architecture-scoped; broad evidence; no single-model extrapolation.

## Acceptance Criteria

Default-on is allowed only after hardware-wide gfx1151 confidence with no capture/output regression; otherwise remain opt-in.

## Notes

Supersedes: RD44
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd44

## Change Log

- 2026-09-09T10:55:57.373909+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:10.094682+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.290852+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.044114+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:59:46.152511+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_030005_amd-streamfus-successors-prbe_8761
- 2026-09-10T03:00:05.767790+00:00 (updated-by): Updated: section:ledger-events
