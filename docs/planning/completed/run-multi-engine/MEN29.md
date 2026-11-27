---
id: MEN29
order: 0
plan: run-multi-engine
state: completed
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
- Migrated 183 KLD evidence records; the final external store contains 209 records with typed `quality` metric groups, including KLD, top-1 agreement, and perplexity fields where retained.
- Completed the broader legacy evidence sweep. The final external store has 1,739 record documents, including 166 unique RUM09 stream results and 78 unique RUM09 long-document prefill measurements; the inventory and indexes agree.
- The 166 RUM09 stream results include ten successful workload results from an interrupted runner attempt. Seventy-two duplicate prefill files were hash-matched and linked rather than counted again.
- Imported results retain source artifacts and available launch context. Historical records remain unverified where sealed execution attestations are absent or execution ended early. MRC records, indexes, and blobs remain outside Git.

## Effort & Risk



## Standards



## Acceptance Criteria

- Existing records remain readable and unchanged.
- Metric groups are typed numeric maps with explicit direction and gate semantics.
- Versioned namespaced extension payloads are accepted without weakening core record validation.
- Archived KLD/top-1/perplexity results are recorded only when source JSON and workload identity support reliable reconstruction; records with missing provenance remain unverified.

## Notes

Historical quality and performance-record migration is complete after the schema/query release. The broader source sweep was classified by evidence type, deduplicated by content hash, and documented in the host-local migration inventory. Failed or unscored attempts, non-model telemetry, kernel microbench evidence, corpus calibration files, and nested results already represented by launch-level records remain excluded. All historical reconstructions are marked unverified.

## Change Log

- 2026-11-27T04:19:22.689462+00:00 (created-by): Created by codex
- 2026-11-27T04:23:54.939980+00:00 (state-transition): State: pending → in_progress
- 2026-11-27T04:32:01.158047+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-11-27T06:39:06.163920+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-11-27T06:39:16.555948+00:00 (state-transition): State: in_progress → completed
- 2026-11-27T06:50:07.960233+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-11-27T06:58:06.470564+00:00 (updated-by): Updated: section:validation, section:notes
