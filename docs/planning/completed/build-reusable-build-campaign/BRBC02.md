---
id: BRBC02
order: 0
plan: build-reusable-build-campaign
state: completed
created-at: '2026-09-09T10:59:11.980443+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: null
---

# Structurally enforce work-root/checkout non-overlap, not just default-topology safety

## Description

Implemented the missing structural guard for explicit work-root/upstream-checkout configuration. ProjectContext now rejects equal or nested explicit roots using component-aware commonpath semantics, while preserving the intentional host-local default cache nested under work_root.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Add a central _paths_overlap helper in tools/bigcherry/core/context.py after path resolution. It uses os.path.commonpath on resolved paths, treats different Windows drives as disjoint, and fails before derived roots are constructed. Apply the check only when upstream_repo is explicitly supplied; the default host-local upstream cache remains a supported nested topology.

## Code Samples & Guidance



## Files

- successor-specs/build-reusable-build-campaign-re35.md
- tools/bigcherry/core/context.py
- tools/tests/core/test_context.py
- tools/tests/campaign/test_campaign_build_flip.py

## Validation

Focused tests: `$env:PYTHONPATH='tools'; python -m pytest tools/tests/core/test_context.py tools/tests/campaign/test_campaign_build_flip.py -q` — 9 passed, 2 subtests passed. Tests cover equal roots, nesting in both directions, valid siblings, explicit path precedence, and the default non-overlapping campaign topology.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Explicit work_root/upstream_repo roots are rejected when equal or nested, sibling roots remain valid, and the default host-local topology remains compatible. The guard is centralized in ProjectContext.resolve() and no per-backend duplicate checks are introduced.

## Notes

Supersedes: RE35
Migration: capability-rebaseline-v3-2026-09
Successor key: build-reusable-build-campaign-re35

2026-09-10: Closed the RE35 gap. The default upstream cache is intentionally under the derived work_root, so only explicit upstream_repo topology is fail-closed; this preserves existing defaults while protecting caller-owned checkouts.

## Change Log

- 2026-09-09T10:59:11.980443+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:21.156759+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.489189+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:26:20.146265+00:00 (state-transition): State: pending → in_progress
- 2026-09-09T15:26:34.704911+00:00 (updated-by): Updated: section:description, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260909_152645_added-a-centralized-fail-close_2080
- 2026-09-09T15:26:45.350902+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:26:53.529383+00:00 (state-transition): State: in_progress → completed
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.352534+00:00 (updated-by): Updated: section:ledger-events
