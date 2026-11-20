---
id: MRC06
order: 6
plan: run-measurement-records
state: completed
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

Merged: #1203 (parity launches publish run records), #1205 (workload, experiment producers on run-record v1), #1209 (native calibration ingest and needle/image probes as records, bigcherry results list/show), #1223 (old JSONL workload index, legacy listing, parity.json and the harness's baseline/compare removed; 138 retired rows kept as evidence), #1231 (one canonical Radiance results path; native lab ingest refuses sidecars without typed metric specs). Tests: tools/tests/bench/test_mrc06_producers.py, tools/tests/lab/test_mrc06_labruns.py, tools/tests/lab/test_flashnext_parity.py, tools/tests/core/test_results.py; core/bench/lab suites 763 pass.

## Effort & Risk

Large across six PRs. Risk: a half-migrated state with two stores; mitigate by removing each old writer in the same PR as its migration.

## Standards



## Acceptance Criteria

Every Radiance measurement producer writes only immutable run-record v1 through core.measurement_producers; no old result store, summary file, record toggle, compatibility output or legacy listing remains; retired results are kept only as evidence. llama.cpp producers are tracked separately in plan run-llamacpp-measurement (LCM01).

## Notes

Producers added on 2026-11-19 that already write their own receipts and should migrate: the compile-crash retry receipts (`bigcherry-compile-retries.jsonl` in the build dir and `<work-root>/build-compile-retries.jsonl`, #1175) and the ad hoc hardware-proof scripts of the day (two-start replay proof, dense TP3 sweep) that stream-bench JSON and `session.json` were stitched together for.

## Slice 1: Flash-Next parity producer (2026-11-20)

The hardware harness now produces one immutable MRC01 record per completed
launch by default (`--records`; `--no-records` is an explicit opt-out).
The native engine remains Python-free; Python consumes the run's strict
manifest, feature-state snapshot, per-launch measurements, source heat and
selected calibration evidence after the owner has stopped the session.
The platform store independently validates every content-addressed blob and
record. Its MRC05 direction/gate/repeat metadata matches the parity harness
exactly; the card-free fixture runs both results compare and the harness's
unchanged compare across two baseline and two candidate launches.

Transition boundary retired 2026-11-20: `OUT/parity.json`, the harness `summarise`, `baseline` and `compare` wrappers and
`--no-records` are removed; the harness calls `bigcherry results` over the launch records. The origin/main base for this slice
contains MRC05 but not the newly pushed MRC04 reader, so selected calibration
TOML was independently verified in that slice; slice 2 replaces the
temporary reader with MRC04's shared validator.
Other producer migrations and hardware activation remain pending.

## Slice 2: workload, llama.cpp, experiment producers, MRC04 consolidation (2026-11-20)

- Replace the temporary parity calibration digest logic with MRC04's strict current/legacy/invalid TOML reader. Only the selected current calibration row can become measured evidence.
- A native Radiance workload run now publishes immutable MRC01 content, with manifest bytes captured while the source exists. Historical index paths were invalidated by the November 17 purge: the old results/index.jsonl is read-only. "results list" exposes historical rows as legacy / missing_content_provenance, with no baseline activation and no automatic conversion.
- The BigCherry-owned llama.cpp wrapper invokes the unmodified external bench/run_bench.py and intercepts its canonical result append calls. This preserves the benchmark process and measured quantities, writing a single MRC01 record rather than new JSON/family/SQLite results. Direct upstream invocations still write their own legacy stores, so operators must use the managed wrapper.
- HI47 experiment bundles publish a result record only when a typed measured-results sidecar exists. The current experiment.json and measured values are content blobs. VA25 canonical execution evidence supplies the optional MRC01 validity field and invalid/missing evidence blocks results baseline, compare and history activation.
- Tests use synthetic hosts/manifests/llama rows/external run_bench and actual MRC04 calibration fixtures; no GPU needed.

Remaining separate slices: bc-serve calibration/lab producers, RSC02 S15, other campaign/compile receipts and migration of consumers still reading old stores. No historic row is rewritten; unrepresentable llama PPL/standard deviation metrics remain verbatim in source-row provenance until a schema extension can represent those phases faithfully.

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

## Radiance-only owner-rule merge update (2026-11-20)

The #1223 main retirement remains authoritative: no parity summary
writer or reader, no workload JSONL index, no record-disable option,
and no managed llama.cpp run-bench CLI. Shell parity baseline/compare
commands delegate directly to the canonical `bigcherry results` API.
The additional MRC06 native-lab path is explicit typed sidecar ingestion
from a cleanly stopped Radiance owner, with run ID and original manifest
digest verification. It pins the complete source evidence as immutable
blobs and marks unattested measurements as unverified. The CLI, sidecar
and owner-state refusal tests are CPU-only. The earlier slice notes and
acceptance audit describe historical state; llama.cpp work is MRC07.
- 2026-11-20T01:49:57.796375+00:00 (updated-by): Updated: section:validation, section:acceptance_criteria
- 2026-11-20T01:50:05.332411+00:00 (state-transition): State: pending → completed
