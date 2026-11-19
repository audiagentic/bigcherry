---
id: MRC06
order: 6
plan: run-measurement-records
state: pending
created-at: '2026-11-19T00:50:46.203067+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: L
---

# Migrate producers onto run-record v1: Radiance workloads store, parity harness, bc-serve lab runs, llama.cpp bench rows and experiment bundles

## Description

Migrate every producer of measurement results onto the platform record (MRC01) so there is one store. Old stores become read-only readers; no new legacy fallbacks (the compatibility-shim rules in docs/reference/tooling/TOOLING.md apply: scan consumers, prove parity, record the retirement).

Producers named today: the workload store (tools/bigcherry/bench/workloads.py, results/index.jsonl); the Flash-Next parity harness (tools/lab/radiance/flashnext_parity.py); bc-serve lab runs (--calibration-run and the measure runs); tools/lab/radiance/rsc02_s15_ab.py; llama.cpp bench rows (the run_bench.py results described in docs/planning/completed/tuning-hip-autotune/HI115.md and similar items); and the experiment bundles (tools/bigcherry/experiment/bundle.py, HI47) with execution attestation (tools/bigcherry/experiment/attestation.py, VA25).

## Steps

- 1. workloads.py: write records instead of index rows (manifest stored by content through the bundle, not a path); keep read_index() as a read-only reader for old rows.
- 2. Parity harness: produce records and baselines through MRC05.
- 3. bc-serve lab runs (--calibration-run, measure): emit a record with the provenance bundle (effective manifest, heat file, calibration record, argv, env, power).
- 4. rsc02_s15_ab.py: write records.
- 5. llama.cpp bench rows: the run_bench.py result appends go to the platform store (the engine supplies an adapter from its row to the record).
- 6. Experiment bundles: bundle.py provenance documents link to records; fold the VA25 execution attestation (ExecutionAttestation, require_execution_identity) into the record as a validity field, so an unattested measurement is marked invalid in the record itself.
- 7. Retire each old writer per the shim rules once its consumers are scanned; list the old stores as read-only in TOOLING.md.

## Detailed Solution & Technical Design

One producer per PR. Each step ends with the old writer removed (not dual-writing) and a reader that still opens old data. Evidence that carries across bumps is the rule, so old records are read and never rewritten.

## Code Samples & Guidance



## Files

- tools/bigcherry/bench/workloads.py
- tools/lab/radiance/flashnext_parity.py
- tools/lab/radiance/rsc02_s15_ab.py
- tools/bigcherry/experiment/bundle.py (HI47 managed bundle runner, validate_provenance_document)
- tools/bigcherry/experiment/attestation.py (VA25 ExecutionAttestation, require_execution_identity)
- docs/reference/tooling/TOOLING.md (shim policy, ownership table)
- docs/reference/tooling/TOOL_DISPOSITION.md (disposition registry rows to update)

## Validation

- Per producer: a run writes a record that round-trips and validates; the old writer no longer exists or is read-only.
- tools/tests passes (BC_WORK_ROOT set); bigcherry check --quick clean.
- The attestation test suite still passes with the field moved into the record.

## Effort & Risk

Large across six PRs. Risk: a half-migrated state with two stores; mitigate by removing each old writer in the same PR as its migration.

## Standards



## Acceptance Criteria

- There is exactly one write path for measurement results.
- Old stores open read-only and have an entry in the disposition registry.
- Attestation validity is a field of the record.

## Notes

Producers added on 2026-11-19 that already write their own receipts and should migrate: the compile-crash retry receipts (`bigcherry-compile-retries.jsonl` in the build dir and `<work-root>/build-compile-retries.jsonl`, #1175) and the ad hoc hardware-proof scripts of the day (two-start replay proof, dense TP3 sweep) that stream-bench JSON and `session.json` were stitched together for.

## Change Log

- 2026-11-19T00:50:46.203067+00:00 (created-by): Created by agent
- 2026-11-19T00:54:27.377473+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-19T07:39:11.871195+00:00 (updated-by): Updated: section:notes
