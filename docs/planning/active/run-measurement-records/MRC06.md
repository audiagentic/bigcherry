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

# Migrate Radiance measurement producers onto run-record v1

## Description

Migrate every producer of measurement results onto the platform record (MRC01) so there is one store. Callers are migrated to the one immutable store and the old active result path is deleted; there are no aliases, summary writers, or old-index readers.

Producers named today: the workload store (tools/bigcherry/bench/workloads.py, results/index.jsonl); the Flash-Next parity harness (tools/lab/radiance/flashnext_parity.py); bc-serve lab runs (--calibration-run and the measure runs); tools/lab/radiance/rsc02_s15_ab.py; and experiment bundles (tools/bigcherry/experiment/bundle.py, HI47) with execution attestation (tools/bigcherry/experiment/attestation.py, VA25).

## Steps

- 1. workloads.py: publish immutable records, pinning the exact manifest; remove the previous measurement-index writer and consumers.
- 2. Parity harness: produce records and baselines through MRC05.
- 3. bc-serve lab runs (--calibration-run, measure): emit a record with the provenance bundle (effective manifest, heat file, calibration record, argv, env, power).
- 4. rsc02_s15_ab.py: write records.
- 5. Deferred to MRC07: llama.cpp bench rows, run-bench adapter and engine-specific producer migration.
- 6. Experiment bundles: bundle.py provenance documents link to records; fold the VA25 execution attestation (ExecutionAttestation, require_execution_identity) into the record as a validity field, so an unattested measurement is marked invalid in the record itself.
- 7. Migrate consumers and delete retired writer, reader, CLI alias and output paths; list retained evidence artifacts in tooling documentation.

## Detailed Solution & Technical Design

One producer per PR. Each migrated producer has a single record write path. Retained historical lab evidence is unchanged, but old measurement outputs are not active readers or public aliases.

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
- Migrated producers have no second measurement store or active old-data readers; disposition rows describe the surviving canonical path.
- Attestation validity is a field of the record.

## Notes

Producers added on 2026-11-19 that already write their own receipts and should migrate: the compile-crash retry receipts (`bigcherry-compile-retries.jsonl` in the build dir and `<work-root>/build-compile-retries.jsonl`, #1175) and the ad hoc hardware-proof scripts of the day (two-start replay proof, dense TP3 sweep) that stream-bench JSON and `session.json` were stitched together for.

## Radiance-only migration rule (2026-11-20)

Parity writes a verified run record for every completed launch, and
`bigcherry results` owns baseline/compare/list/show operations. The
`parity.json` writer and readers, launcher baseline/compare commands,
record-enablement flags, old results-index listing and llama.cpp run-bench
CLI are removed. No historical raw lab evidence is rewritten. The platform
registry and core producer remain engine-neutral, but llama.cpp adapters
and acceptance are assigned to MRC07.

## Slice 3: native calibration, probes, RSC02 and results views (2026-11-20)

- `bc-serve --calibration-run` stays Python-free. The C++ session owner saves
  actual child argv and curated overrides, plus exact A/B prompts, heat,
  freshly ingested MRC04 TOML, helper evidence and native hashes/receipt.
  Source mutation during measurement is refused. The post-run platform
  `bigcherry results ingest-calibration --run-dir RUN` validates a completed
  native session and the matching MRC04 `native_v1` selected record (run ID,
  key, revision, engine, host-facts digest) and publishes one immutable
  MRC01 record. A SHA-only pointer remains in the run directory. Collecting
  host facts at ingest does NOT equate to measurement-start observations.
- `needle_probe.py` and `image_probe.py` now check native owner and matching
  local HTTP port before requesting, retain their scoring, and store exact
  raw result and corpus/transmitted-image bytes as MRC01 blobs. `--out`
  contains a digest pointer, not another result index.
- `rsc02_s15_ab.py` remains an active leased RSC02 S15 acceptance driver,
  not retired. Its original manifest bytes are snapshotted (base64 + SHA);
  each cleanly stopped arm gets a v1 record with native feature states on
  the B arm. Existing comparison/lifecycle JSON remains read-only source
  evidence for the active comparator.
- `results list --today --engine radiance --workload parity` displays today
  in **UTC**; `results show SHA256` verifies the full MRC01 record and blobs
  without reopening run folders. All new fixture gates are CPU-only.

## MRC01–MRC06 acceptance audit: can the plan be closed?

**No.** Several core contracts are implemented, but cross-producer and
hardware acceptance criteria remain incomplete:

| Item | Acceptance criteria met | Not met / still to prove |
| --- | --- | --- |
| **MRC01** | Content-addressed self-contained records (including deletion fixtures); mandatory distinct decode/prefill/TTFT groups; single MRC01 library reader; GC/retention reference protection tests (#1204) | WSL01 two-way typed lifecycle-record edge and original results put/query CLI are incomplete; production recovery proof pending |
| **MRC02** | Engine-neutral host collector; unreadable/zero-cap markers; required record host facts; workloads delegate sysfs to core | Native calibration post-run collector records host state at ingest, not start; real host vs ROCm/sysfs validation remains |
| **MRC03** | Native Radiance startup warnings/status, MRC03 files and populated Radiance record snapshots | llama.cpp and legacy RSC02 Bash arms can still have empty feature-state arrays; llama feature adapter and registry-only extension/real orphan-warning gate not finished |
| **MRC04** | Declared narrow build-independent key and provenance, native TOML/platform reader/store, legacy refusal; fresh native calibration linked as an MRC01 content blob | Cross-build/pin-bump live activation, typed calibration-record reference and large quantity-blob interop pending |
| **MRC05** | Portable baseline by value, one-sided spread gates, single-launch refusal, provenance diff, record-only drift history | Hardware-generated provenance-bearing replacement for historical v1.17.0 baseline missing; parity legacy baseline/compare consumers still in transition |
| **MRC06** | Migrated parity, workloads, managed llama rows, HI47 typed bundles, native finite calibration, needle/image and active RSC02 adapters; VA25 validity and legacy workload read-only listing | **Global one-write-path criterion not met:** direct external run_bench still writes old stores if wrapper bypassed; non-calibration bc-serve lab measure, S8, TP3/replay and compile retry producers remain; some legacy result consumers/writers not retired or fully dispositioned |

Remaining slices must inventory and migrate other producers, enforce the
managed external llama entrypoint, complete cross-engine feature states and
typed WSL01 edges, retire old writers after consumer scans, and run explicit
leased hardware acceptance. Deleted 2026-11-17 manifest bytes must never be
recreated or silently trusted as new evidence.

## Change Log

- 2026-11-19T00:50:46.203067+00:00 (created-by): Created by agent
- 2026-11-19T00:54:27.377473+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-19T07:39:11.871195+00:00 (updated-by): Updated: section:notes
