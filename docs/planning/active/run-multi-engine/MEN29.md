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

- Python compilation passed for the changed modules; run-record schema JSON parsing passed; git diff --check passed.
- Ruff reported two unused imports in the existing CLI module; tests were not run.
- Run KLD migration remains deferred until the extended schema is deployed so the current store reader continues to accept all published records.

## Effort & Risk



## Standards



## Acceptance Criteria

- Existing records remain readable and unchanged.
- Metric groups are typed numeric maps with explicit direction and gate semantics.
- Versioned namespaced extension payloads are accepted without weakening core record validation.
- Archived KLD/top-1/perplexity results are recorded only when source JSON and workload identity support reliable reconstruction; records with missing provenance remain unverified.

## Notes

The archived aggregate KLD JSON is reconstructable, but writing quality groups to the live v1 store before deployment would make records unreadable by older strict v1 readers. Publish the historical quality records after the schema/query update is released.

## Change Log

- 2026-11-27T04:19:22.689462+00:00 (created-by): Created by codex
- 2026-11-27T04:23:54.939980+00:00 (state-transition): State: pending → in_progress
- 2026-11-27T04:32:01.158047+00:00 (updated-by): Updated: section:validation, section:notes
