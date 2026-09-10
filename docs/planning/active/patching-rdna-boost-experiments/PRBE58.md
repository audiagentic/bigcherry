---
id: PRBE58
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:31.760072+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# FORK-MTP-002: Remove four-copy Vulkan MTP handoff

## Description

Evaluate removing redundant four-copy Vulkan MTP handoff after coupled RD71/RD72 analysis, preserving correctness and PP/VRAM budgets.

## Steps

Recheck source commit 1fcc05da and MrLordCat registry; diff with RD71 commit 41a8ca78 before implementation; measure copy path variants, staging alternatives, draft depth and context on dual RDNA4 Vulkan; compare single-GPU controls; verify hidden-state/output identity and record copy count/bytes/time and effective TG.

## Detailed Solution & Technical Design

Reduce Vulkan MTP handoff copies where NextN/output placement and pipeline-parallel choices allow it. Determine whether RD71 supersedes/overlaps this path before porting. Keep host staging and original four-copy fallback when correctness or VRAM/PP cost changes.

## Code Samples & Guidance



## Files

Vulkan MTP scheduler/copy path; topology and pipeline controls; hidden-state/output tests; dual-RDNA4 copy and E2E evidence.

## Validation

Hidden-state/output identity; copy count/bytes/time, effective TG, PP and VRAM across path variants, draft depths and contexts.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only if copies fall without hidden-state/output changes, PP/VRAM cost increase, or acceptance loss; retain fallback otherwise.

## Notes

Supersedes: RD72
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd72

## Change Log

- 2026-09-09T10:57:31.760072+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:43.820256+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.389761+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.197952+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:13:26.102767+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031346_repaired-four-more-migrated-pa_4345
- 2026-09-10T03:13:46.510818+00:00 (updated-by): Updated: section:ledger-events
