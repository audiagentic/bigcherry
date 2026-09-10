---
id: PHA06
order: 0
plan: patching-hip-autotune
state: completed
created-at: '2026-09-09T10:49:06.721439+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: null
---

# CORRECTED -- no real regression: RD13 test assumed an out-of-band local mutation that pull() legitimately resets

## Description

Fix the RD13 patch compatibility test so it validates the patch against an isolated copy of the current pinned vendor source rather than requiring an out-of-band patch application in the mutable checkout.

## Steps

1. Read the current pinned vendor source.\n2. Copy it into a temporary isolated target at the patch's declared path.\n3. Run the real patch apply dry-run and assert exactly one applied result.\n4. Keep the package's independent anchor/near-miss tests unchanged.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

tools/tests/patch/test_patch_1206_rd13_mul_mat_add_view_fusion.py

## Validation

Focused pytest: tools/tests/patch/test_patch_1206_rd13_mul_mat_add_view_fusion.py — 5 passed.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

The test passes on a clean checkout without requiring patch 1206 to have been applied out-of-band; it fails if the current pinned source no longer accepts the real patch anchor.

## Notes

Supersedes: HI149
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-hip-autotune-hi149

Implemented 2026-09-09. Replaced direct post-apply/live-tree assumptions with an isolated-copy dry-run against the current vendor source.

## Change Log

- 2026-09-09T10:49:06.721439+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:12.312356+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events




- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.857574+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T12:28:35.585358+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-09-09T12:28:51.135157+00:00 (state-transition): State: pending → in_progress
- chg_20260909_122907_fixed-the-rd13-test-process-de_6932
- 2026-09-09T12:29:07.294612+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:58:51.678906+00:00 (state-transition): State: in_progress → completed
- chg_20260909_135901_closed-the-corrected-rd13-test_9901
- 2026-09-09T13:59:01.913922+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.371396+00:00 (updated-by): Updated: section:ledger-events
