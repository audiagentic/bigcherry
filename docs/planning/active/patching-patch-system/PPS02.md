---
id: PPS02
order: 0
plan: patching-patch-system
state: pending
created-at: '2026-09-10T02:09:30.604686+00:00'
breadth: ''
skill: advanced
created-by: gpt-semantic-carry-forward-audit
work: L
priority: P0
---

# Semantic successor carry-forward audit and migration gate

## Description

Audit every continuing capability-native successor against its frozen predecessor and preserve all actionable implementation, correctness, performance, fallback, safety, negative-evidence, provenance, and dependency requirements. Structural lineage alone is insufficient.

## Steps

1. Build a clause-level predecessor/successor inventory from the locked 524-item source and accepted semantic review.
2. For every continuing item, classify each predecessor requirement as carried forward, completed with evidence, superseded by a named successor, redundant with a named owner, obsolete with rationale, or historical-only; silence is invalid.
3. Replace generic migration scaffolding and successor-specs placeholders with concrete scope, paths, gates, and negative constraints.
4. Repair high-confidence losses first, including RD04→PRBE01, RD17→PRBE14, and RD62/NRO06→PRBE52.
5. Add a semantic validator that flags generic acceptance/steps, placeholder files, and missing disposition mappings while exempting reference-only and terminal items.
6. Re-run structural validation and obtain GPT/manual review for disputed mappings.

## Detailed Solution & Technical Design

Use the frozen predecessor as the source of truth and the active successor as the current authority. The validator should parse lifecycle state, successor lineage, actionable predecessor sections, and successor sections; emit machine-readable findings with predecessor ID, successor ID, missing clause/category, and required disposition. It must not infer completion from terminal state or from the existence of a successor-specs path. Keep historical predecessor files unchanged.

## Code Samples & Guidance



## Files

tools/lab/planning-capability-rebaseline-v3/scripts/validate_semantic_carryforward.py; docs/planning active successor specs; review-evidence semantic carry-forward manifest

## Validation

Passes on a fixture set covering a fully repaired successor, a generic placeholder successor, a completed/reference-only predecessor, explicit obsolete/completed dispositions, and duplicate consolidation. Full source inventory has zero unclassified actionable clauses.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance; fail closed on omitted actionable requirements.

## Acceptance Criteria

Every continuing successor has explicit semantic carry-forward/disposition coverage for all actionable predecessor requirements; no live successor relies solely on generic migration scaffolding or successor-specs placeholders; validator and structural checks pass; disputed items are manually resolved and recorded.

## Notes

Created in response to dev-GPT audit of planning-refactor@04bc4a8. This is a merge-blocking semantic preservation gate, not a request to move or delete predecessor history.

## Change Log

- 2026-09-10T02:09:30.604686+00:00 (created-by): Created by gpt-semantic-carry-forward-audit

## Ledger-events

- chg_20260910_021313_the-semantic-audit-is-now-trac_4827
- 2026-09-10T02:13:13.301450+00:00 (updated-by): Updated: section:ledger-events
