---
id: PRBE73
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:39.838631+00:00'
breadth: ''
skill: ''
created-by: capability-rebaseline-v3
priority: null
---

# Hardware-blocked patch retirement policy (RD21/RD22-class items)

## Description

Introduce reviewable retirement/recheck metadata and policy for hardware-blocked patch items.

## Steps

Add hardware_blocked_since, last_source_recheck, last_upstream_equivalence_check and next_review_date to deferred items; define threshold and automatic flagging; audit RD21/RD22-class items and require explicit retire/recheck decision; recheck whether current gfx1201 qualifies RD22.

## Detailed Solution & Technical Design

Prevent indefinite maintenance of hardware-deferred patches. Flag stale items for human disposition while preserving evidence and never auto-deleting a potentially useful patch.

## Code Samples & Guidance



## Files

Plan/evidence schema, validator/report, hardware-blocked item metadata and RD21/RD22 audit records.

## Validation

Schema/validator tests, threshold date tests, stale-item report and explicit decisions.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Every hardware-blocked item has review metadata and is flagged after the threshold for explicit retire/recheck; no silent accumulation or automatic deletion.

## Notes

Supersedes: RD93
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd93

## Change Log

- 2026-09-09T10:58:39.838631+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:50.884404+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.458618+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.298240+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:20:26.481445+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032047_repaired-five-more-active-succ_6361
- 2026-09-10T03:20:47.960815+00:00 (updated-by): Updated: section:ledger-events
