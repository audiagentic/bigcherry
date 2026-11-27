---
id: MEN29
order: 0
plan: run-multi-engine
state: in_progress
created-at: '2026-11-27T04:19:22.689462+00:00'
breadth: ''
skill: intermediate
created-by: codex
work: M
priority: P2
---

# Typed quality metrics and versioned run-record extensions

## Description

Extend platform run-record v1 with queryable quality metric groups (KLD, top-1 agreement, perplexity) and namespaced versioned extension payloads. Migrate archived quality measurements where source data supports exact reconstruction, preserving unverified status when execution attestation is incomplete.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

- tools/bigcherry/core/schemas/run-record-v1.schema.json
- tools/bigcherry/core/run_records.py
- tools/bigcherry/core/results.py
- docs/reference/tooling/TOOLING.md

## Validation

- Validate representative existing v1 records and migrated quality records against the schema.
- Query quality metrics by group and confirm legacy speed metrics retain their existing query names.

## Effort & Risk



## Standards



## Acceptance Criteria

- Existing records remain readable and unchanged.
- Metric groups are typed numeric maps with explicit direction and gate semantics.
- Versioned namespaced extension payloads are accepted without weakening core record validation.
- Archived KLD/top-1/perplexity results are recorded only when source JSON and workload identity support reliable reconstruction; records with missing provenance remain unverified.

## Notes

## Change Log

- 2026-11-27T04:19:22.689462+00:00 (created-by): Created by codex
- 2026-11-27T04:23:54.939980+00:00 (state-transition): State: pending → in_progress
