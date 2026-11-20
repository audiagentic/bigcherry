---
id: MRC01
order: 1
plan: run-measurement-records
state: completed
created-at: '2026-11-19T00:50:23.229499+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Platform run-record v1: one engine-agnostic measurement record schema and content-addressed store

## Description

One platform-owned, engine-agnostic record of a measurement run, kept in a content-addressed store under $BC_WORK_ROOT. Today three unrelated stores describe runs: the run manifest (WSL01: lifecycle, no metrics), the workload result index (metrics, but the effective manifest is stored as a PATH plus digest) and engine-lab parity files. When the 2026-11-17 purge deleted the referenced manifests, every workload record lost its configuration; the v1.17.0 Flash-Next baseline never had a manifest or heat file at all (docs/evidence/2026-11-19-flashnext-4gpu-regression). A record must carry what it needs inside itself, by content.

Record contents: identity (engine, build id / plan / binary hash, source revision, model, layout); workload definition and metrics, with decode and prefill/TTFT always separate metrics; a provenance bundle (owned here: effective config/manifest, every input file that changes results such as heat file, calibration record and policy file, exact argv and environment, power settings), each stored by sha256 and never by path; host facts (MRC02); feature states (MRC03). Records and blobs live under $BC_WORK_ROOT keyed by sha256; a reader/query library is the only way tools read them.

## Steps

- 1. Schema: a platform JSON schema (record v1, blob reference, provenance bundle) under tools/bigcherry/core with validation and canonical serialisation; tests with fixture records.
- 2. Store: content-addressed writer (blobs and records by sha256, atomic write like run_manifest.write_atomic, idempotent put) and reader/query library (by engine, layout, workload, metric, time).
- 3. Link to the run manifest: a record names its run_id; run_manifest inputs/references gain a typed record edge so a run and its record resolve each other. Keep run_manifest.py (WSL01) as the lifecycle owner.
- 4. Retention: tools/bigcherry/core/retention.py treats every blob and record referenced by a live record as protected (typed reference roots); add the same protection to the WSL09 build GC. Tests: a referenced blob is never a cleanup candidate.
- 5. CLI: `bigcherry results put|show|query` handlers in tools/bigcherry/cli; docs in TOOLING.md ownership table (Core row).

## Detailed Solution & Technical Design

Placement follows docs/reference/tooling/TOOLING.md: schema, store and reader are reusable core (`bigcherry.core`), CLI handlers live in tools/bigcherry/cli/, tests in tools/tests/core/. Lab or engine-specific producers stay out of core and only call the library (MRC06). The record never stores a path as the only reference to an input: it stores the sha256 of the bytes in the blob store and may add a path as a hint. The run manifest stays the lifecycle record; a record is the measurement record. Engines supply adapters (a function mapping their own outputs to the record fields); they own no schema.

## Code Samples & Guidance



## Files

- tools/bigcherry/core/run_manifest.py (WSL01 lifecycle: manifest with execution, host, inputs [{role, uri, sha256}], typed references; kept, linked)
- tools/bigcherry/core/retention.py (plan_cleanup: protected vs candidate, typed-reference roots; extend with blob/record roots)
- tools/bigcherry/bench/workloads.py (current store: DEFAULT_INDEX results/index.jsonl, manifest_info() records manifest path plus manifest_digest; migrated in MRC06)
- docs/reference/tooling/TOOLING.md (placement rules; ownership table to update)
- docs/evidence/2026-11-19-flashnext-4gpu-regression/README.md (motivation)

## Validation

Merged: #1188 (run-record v1 schema, content-addressed store, queries, retention protection), #1204 (GC honours run-record references), #1213 (typed bidirectional WSL01 run-manifest <-> run-record link, bigcherry results put|show|query, production recovery dry-run). Tests: tools/tests/core/test_run_records.py (self-contained after source deletion, schema-required decode/prefill/TTFT groups, missing blob or bad schema refused before persist, store survives deleted source), the #1213 link/CLI tests, and the GC classification tests. The 2026-11-20 closure audit found all four acceptance criteria met; #1213 delivered the remaining steps.

## Effort & Risk

Medium. Risk: scope creep into one record per engine; keep the schema generic and push everything engine-specific into provenance blobs. Do first; MRC02 to MRC06 depend on it.

## Standards



## Acceptance Criteria

A run record is self-contained: it still verifies and shows how the run executed after its manifest, heat file and run folder are deleted. Every record carries separate decode, prefill and TTFT metric groups. Records are linked both ways to their WSL01 run manifest. Cleanup never deletes content a record references.

## Notes

Increments already on main (2026-11-19): the stored manifest and run folder of a native `bc-serve serve` are the provenance a record must hold by content (the replay key hashes the binary, engine, model, heat file, calibration store and every operator `--set`, `named_cache` in `session.cpp`); the evidence of 2026-11-19 (Flash-Next regression, TP3 no-regression) was assembled by hand from run folders, which is the use case. Build-flake receipts (`bigcherry-compile-retries.jsonl`, #1175) are the first append-only machine-readable run facts written beside a build.

## Change Log

- 2026-11-19T00:50:23.229499+00:00 (created-by): Created by agent
- 2026-11-19T00:54:03.018574+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-19T07:38:50.554679+00:00 (updated-by): Updated: section:notes
- 2026-11-19T21:42:26.289135+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria
- 2026-11-19T21:42:31.869229+00:00 (state-transition): State: pending → completed
- 2026-11-19T21:42:49.717875+00:00 (updated-by): Updated: section:validation
