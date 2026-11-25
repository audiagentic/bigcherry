# BigCherry tooling

This is the normative map for maintained tooling. Before adding, moving, or
retiring a module, script, lab experiment, or compatibility shim, search for
the existing command/API and read the owning domain's tests. If the work is
exploratory or plan-specific, start in `tools/lab/<plan-topic>/` and keep it
outside the production package until it has an explicit disposition.

## Root-level source versus generated output

Root source follows its maintained owner:

- `engines/llamacpp/dispositions/` holds narrow revision-and-digest-bound known-broken patch records. It is tracked and reviewed with the llama.cpp engine, not stored under ignored artifacts.
- `tools/bigcherry/tuning/sql/` holds the cross-engine (HIP and Vulkan) tuning database schema and migrations, owned by the tuning platform.
- `experiments/host-kernel-gfxoff-7.0.13/` catalogs a reproducible host-kernel reconstruction and its evidence; it is neither an installed module nor production engine code. Generated modules and installed kernel files must not be removed as scratch.

The root `out/` directory is ignored scratch, never a durable evidence store; promote retained reports to `docs/evidence/`. Synthetic C++ test models belong in isolated, automatically cleaned build scratch, not a tracked root-level fixture directory.

## Start with the right owner

| Concern | Maintained owner | Use it for |
| --- | --- | --- |
| CLI assembly | `tools/bigcherry/cli/main.py` | Parser registration, command routing, and top-level help |
| CLI handlers | `tools/bigcherry/cli/` | Argument validation and presentation for command families |
| Release | `bigcherry.release` | Pin, release, source/build consistency, and release validation |
| Source | `bigcherry.source` | Upstream identity, checkout, source audit, and workspace lifecycle |
| Build | `bigcherry.build` | Recipes, generated trees, toolchains, compile checks, and builds |
| Patches | `bigcherry.patch` | Package discovery, lifecycle, application, validation, and patch evidence |
| Patch-validation campaign | `bigcherry.patch.validation_campaign` (CLI/`run()` orchestrator) over `bigcherry.patch.campaign.*` | Infrastructure lives in the owning submodule: `build` (HIP env, CMake configure, tree builds, shared primitives), `trace` (trace probes, real-GPU guard), `contract` (correctness gate, parity, eligibility), `scaffold` (standard-campaign builds), `benchmark` (paired llama-bench, wiring, device pool), `qualification` (evidence manifests), `producer` (validation-producer runtime). `campaign/*` never imports the CLI module; patch-local producers import neither |
| Campaigns | `bigcherry.campaign` | Build/smoke/comparison/benchmark orchestration and campaign lanes |
| Experiments | `bigcherry.experiment` | Contracts, identities, bundles, and experiment state transitions |
| Tuning/replay | `bigcherry.tuning` | Catalog, journal, measurement, correctness, promotion, ranking, and replay |
| Core | `bigcherry.core` | Paths, context, configuration, artifacts, provenance, platform run-record v1 store/query, and pipeline foundations |
| Analysis | `bigcherry.analysis` | Reusable offline reports and derived views; not production selection logic |
| GPU power qualification | [AI power screen](AI_POWER_SCREEN.md), `bigcherry.hardware.ai_probe` | Independent leased matrix screening of observed named profile controls; locally built HIP diagnostic, no engine runtime dependency or power writes |
| Profiling/diagnostics | `bigcherry.profiling` | Real-hardware diagnostic harnesses such as rocprofv3 capture and RCCL qualification |
| Permanent tests | `tools/tests/<domain>/` | Deterministic offline tests organized with the implementation owner |

The observational `bigcherry --engine radiance patch-footprint` generator is owned
by `bigcherry.analysis.patch_footprint` and `cli.patch_footprint`. It derives
lexical anchor/edit/byte counts from the registry and every composed source
in the current engine recipes, without applying edits or reading a GPU.
`--check` compares the committed `docs/evidence/radiance-patch-footprint/`
JSON and Markdown reports without writing; regenerate on recipe/patch updates.
Patch-created targets are not counted as upstream-owned files, even if later
patches anchor into them. This is an impact inventory, not a rebase proof.

The normal call chain is:

```text
bigcherry CLI -> cli/main.py parser -> cli/<domain>.py handler
              -> <domain>/workflow.py or <domain> implementation
              -> tests, receipts, evidence, or artifacts owned by that workflow
```

### JSON persistence

Runtime matrix, patch-build and smoke campaign JSON publications use
`bigcherry.hardware.power_receipt.write_json_atomic`, the existing receipt
writer. It preserves sorted, two-space-indented JSON and a trailing newline;
campaign callers explicitly retain their ASCII/Unicode encoding, newline and
file-only fsync behavior. Power receipts retain directory fsync by default.
`bigcherry.core.json_io.read_jsonl` owns the shared execution-audit and rollup
reader: it skips blank lines and propagates malformed JSON and file errors.

### Calibration records (MRC04 slice 1)

The engine-agnostic JSON contract is
`tools/bigcherry/core/schemas/calibration-record-v1.schema.json`;
the **per-kind key declaration** is
`tools/bigcherry/core/schemas/calibration-kinds-v1.json`.
`bigcherry.core.calibration_records` validates strict records, computes the
declared SHA-256 key, and offers `CalibrationStore.get/put/list` for platform
JSON records under `records/<kind>/<key>.json`. `put` serialises writes with
a persistent `.lock`, fsyncs temp file and directory and atomically publishes
with rename; a duplicate refuses unless `replace=True` is explicitly
requested. Provenance is not in the key. The store does not activate or
generate a policy. MRC01 v1 now supports typed selected-calibration and quantity blob references.

Radiance **continues to read and write TOML natively**, without Python at
runtime: `bc-serve calibration-ingest` writes `[[live_policy_record]]` entries
into `live-policy-calibration.toml` under its native flock/atomic-rename
contract. New rows declare `record_schema_version = 1`,
`calibration_kind = "radiance.live_policy"`, `calibration_key_fields`,
and `calibration_inputs` (canonical JSON) along with source build identity,
plan key, run ID, host-facts digest, source-slice identity (or explicitly
`unavailable`), date, evidence, and measured/absent quantities. A native
reader enforces the narrow key before activation. Earlier post-#1169 rows
with `calibration_inputs` and source plan/build but without the extra MRC04
v1 fields stay **current** and activatable through a read-only, in-memory
compatibility mapping; missing run/source-revision/host digests are not made up.
Only genuinely hash-only records lack activation provenance and are rejected
with `legacy_record_schema`, leaving serving available with the policy off.
The production key itself is unchanged from #1169: `sha256` over
Radiance's sorted `policy_calibration_inputs` JSON.

The operator commands are read-only (no hardware lease, no activation):

```sh
bigcherry calibration list --store /path/to/live-policy-calibration.toml
bigcherry calibration show --store /path/to/live-policy-calibration.toml --kind radiance.live_policy --key <sha256>
bigcherry calibration validate --store /path/to/live-policy-calibration.toml
```

An omitted `--store` defaults to `$BC_WORK_ROOT/live-policy-calibration.toml`
when `BC_WORK_ROOT` is set. A JSON-store directory can be supplied instead.
`list` and `show` mark both native v1 and verified post-#1169 rows as
`current`; genuinely hash-only entries are `legacy` with their explicit refusal
reason. `validate` accepts current rows, and returns nonzero when any legacy or
malformed row is unusable. Source facts for newly written v1 rows are hashed
from the **verified source manifest's
`spec.facts`**, not guessed from the machine at query time; source revision
comes from installed build metadata's `source_slice_id` when present,
otherwise is explicitly `unavailable`. A rebuild changes provenance, not
the key. Changes to model, card identities, layout, objective, power
profile, histogram cadence, helper execution or remote service do change it.

### Immutable selected calibration in run records (MRC04)

An MRC01 run record may carry optional `calibration_record`:
`status=selected` requires an exact SHA-256-verified calibration JSON
`record_blob`, its narrow `key_sha256` and `kind`, and named
`quantity_blobs` for large measurements. The platform checks selected
key/identity and every referenced blob at write, read, and retention scans.
`status=none` requires a nonselection `reason_code` and `reason_detail`.
Old immutable v1 records without the optional field remain readable but do
not assert which calibration was selected.

Native `bc-serve status --json` reports the selected kind/key or a
nonselection reason; the native runtime does not run Python. The serving CPU test activates a record
from build A on build B when the same narrow scope matches; separate leased
device activation proof must not be inferred from CPU-only testing.

### Platform measurement records (MRC01–MRC03)

`bigcherry.core.run_records.RunRecordStore` owns v1 immutable run records at
`$BC_WORK_ROOT/run-records/v1/`, with SHA-256 blobs under
`blobs/<first-two-hex>/<sha256>` and canonical JSON records under
`records/<first-two-hex>/<sha256>.json`. The platform JSON Schema lives in
`tools/bigcherry/core/schemas/run-record-v1.schema.json`.
`put_blob`, `put_blob_file`, `put_record`, `get_blob`, `get_record` and
`query` are the supported platform APIs; all reads verify content hashes and
required blobs. Query supports engine, layout, workload, phase-qualified metric
(e.g. `decode.tokens_per_second`) and UTC time. Metrics for decode, prefill
and TTFT are separate fields; an empty phase means it was not measured.

An input path is only an optional hint: effective configuration, exact argv,
environment, power settings and result-affecting inputs are stored as blob
references. `host_facts` is populated by the MRC02 platform collector, and
`feature_states` by MRC03's native snapshot/reader. Engines supply adapters,
not schema or store logic; a native runtime remains Python-free. The WSL01
run manifest continues to own lifecycle. The platform's
`bigcherry.core.host_facts` collector returns per-fact
`{value, readable, reason?}` structures over injectable sysfs and procfs
roots. Its read-only probes include kernel, ROCm, amdgpu/firmware, VRAM,
PCIe card/bridge links, clock state and OD max, power cap and bounds,
host memory/DIMMs, environment-file SHA-256 and GPU lease status. A reported
zero current power cap is flagged `zero_cap_suspect`, never interpreted as
a valid cap. The native Radiance owner seals host facts before child
admission and binds the exact start bytes by SHA-256 in its session receipt.
Parity, workload, calibration, native lab, probe and RSC02 B-arm producers use
that verified receipt without a second sample; publication and retrieval
reject empty, unwrapped, late, or mismatched Radiance host facts. Unsealed
Radiance records must be deleted or re-measured, never reconstructed from
the current host. Experiment bundles seal their own pre-execution facts.
Other engines' producer migrations belong to MRC07. The platform collector
does not import any engine. Non-measurement lifecycle manifests may collect
host state separately without claiming it is a measured start.
The workloads tool imports its card/power sampling readers from this core
module. WSL01 manifests carry `references: [{type: measurement_record, id: SHA256,
role: measurement}]` linked with `run_manifest.link_measurement_record`.
`resolve_measurement_records(run_dir, store)` and
`find_run_for_record(runs_root, store, SHA256)` verify both identities and
content; links never change lifecycle/harvest/retention state. The SHA store
query accepts `run_id` as well as engine/layout/workload/metric/time.
Operators use `bigcherry results put --record FILE --blob SHA256=PATH
--run-dir RUN_DIR` to copy/validate all supplied content and link WSL01;
`bigcherry results show SHA256` and `bigcherry results query --run-id RUN_ID
--metric prefill.tokens_per_second --json` read exclusively through the store.
`results list` retains the explicit read-only legacy-row bridge. Never
promote missing historical provenance by guessing file bytes.
MRC06 owns remaining producer migrations.

MRC03's feature-state v1 JSON contract lives at
`tools/bigcherry/core/schemas/feature-state-v1.schema.json`: a list of
`{feature_id, state, reason_code, reason_detail, engine, observed_at}` entries,
with state `active`, `inactive` or `degraded`. The platform-only API
`bigcherry.core.feature_states.read_feature_states(path)` validates a native
`feature-states.v1.json` snapshot (no symlinks, duplicate feature IDs,
or invalid states), and `with_feature_states(record, path)` copies it into
a run record after checking the engine identity. The native Radiance
`bc-serve` publishes this file from its admitted manifest before starting
serving children; its durable owner session record and `status --json` expose
the same entries. A helper-backed production run warns in the owner log
when live placement is inactive or degraded, including the reason code and
the preflight/provenance reason; other default-off options are diagnostic
rather than automatically required. MRC06 still owns producer migration.

The platform registry at `tools/bigcherry/core/feature_registry.py` declares
Radiance feature IDs and their production expectations. The native owner reports
all seven declared features before child admission. Missing, duplicate, and
wrong-engine states refuse new Radiance measurement publication. Other engine
producers are scoped to MRC07.

Inspect the authenticated native snapshot with
`bigcherry status --run-dir RUN`. A required inactive or degraded feature
prints its native reason, and `bigcherry status --run-dir RUN
--strict-production-features` returns status 2. For production admission,
`bc-serve serve CONFIG --run-dir RUN --strict-production-features`
refuses a missing required feature **before child spawn**, while writing its
snapshot and diagnostic. A run without selected helpers does not require live
placement. No Python is invoked by the native serving runtime.

### Portable results: baseline, compare, history (MRC05 slice 1)

`bigcherry.core.results` owns offline baseline gates, comparisons and history
over verified MRC01 run records. Records may include `session_id` for
same-session A/B and `metric_specs` for each phase-qualified metric:
`direction: higher|lower`, `gated: true|false`, optional `repeat_groups`
(samples of the **same prompt**, not distinct prompts) and optional `covariate` (another metric of the same record that the
gated metric moves with; with four or more baseline launches the baseline fits the metric on it by least squares and a
comparison reads each measured launch at the baseline's mean covariate, so a metric that follows how many sessions decode
together is gated at equal concurrency, with the tolerance taken from the spread of the adjusted values). Existing records without
these optional fields remain readable but cannot create a gated baseline.

A baseline requires at least two distinct launch `run_id` values, identical
engine/model/layout/workload definitions, and intact provenance blobs. It stores
per-launch medians and a one-sided tolerance from the larger of across-launch
relative spread and true-repeat relative spread. Missing gated metrics fail
comparison; informational metrics are never gated. Baseline JSON embeds
canonical record contents and all referenced input blobs by value (base64
with SHA-256) and validates them without original run folders or source files.
**Treat baseline files as sensitive**: embedded environment/configuration
blobs may contain secrets; do not commit them without inspection.

`results compare` requires at least two candidate launches; host-facts and
feature-state differences appear before verdicts, alongside config, heat,
policy, kernel, power, argv, environment and build differences. Different or
unknown sessions warn to prefer same-session A/B. A provenance difference
does not itself fail a one-sided metric gate. `results history` requires an
engine, layout, workload and phase-qualified metric; baseline and trailing
drift flags use two launches, not a single-launch verdict.

```bash
bigcherry results baseline --engine radiance --layout flashnext-4gpu --workload flashnext-parity --name reference --to reference.json
bigcherry results compare --baseline reference.json --since 2026-11-20T00:00:00Z
bigcherry results history --engine radiance --layout flashnext-4gpu --workload flashnext-parity --metric decode.decode1_ms_per_step --baseline reference.json --json
```

For controlled cohorts, repeat `--record SHA256` on baseline/compare, or
filter with `--since`/`--until`. `--work-root` selects the record store.
Compare exits 1 for regression/missing gated metrics, 2 for invalid input,
0 for a clean gate. MRC06 continues to migrate the remaining producers.

**MRC06 slice 1: Flash-Next parity.** `tools/lab/radiance/flashnext-parity.sh
measure` publishes one immutable run-record v1 per completed native launch under
`$BC_WORK_ROOT/run-records/v1/`; the record is the result (there is no summary
file). Each `OUT/launch-N/record.sha256` contains a digest verified against the
immutable MRC01 store. `flashnext-parity.sh baseline --out OUT --name NAME --to
FILE` and `compare --out OUT --baseline FILE` are `bigcherry results baseline`
and `bigcherry results compare` over those receipts, with the host, feature,
config and policy provenance differences printed before the gates; both refuse
missing launch receipts and path-only data. The record carries
`engine=radiance`, build ID and binary hash from the build metadata, Git
revision, measured model/layout, the exact harness workload definition,
phase-separated cold/sess8 TTFT, document/long-context prefill, decode step
times and ungated draft acceptance. `metric_specs` keeps the harness's
one-sided gate and real same-prompt repeats (distinct prompts are not noise
samples). No GPU tests are needed for the adapter's fixture/compare parity.

The native strict manifest, source heat bytes, the selected calibration row
when the policy is configured, full `live_policy_state`, owner argv/sets,
a curated environment snapshot, effective power state and actual
warmup are SHA-256 content blobs, not mutable paths. Heat and calibration
inputs are snapshotted before launch and checked for mutation; feature states
are read from the native run directory using MRC03. Passive MRC02 host facts
are captured immediately before measurement. Incomplete state, metadata,
or mismatched key aborts **record publication**, rather than fabricate
provenance. There is no parity summary writer or reader. The shell baseline and
compare commands delegate to `bigcherry results` over SHA-verified
launch receipts; the platform store is the sole result view.

**MRC06 slice 2 — producer migration and legacy listing.** The parity producer now reads the selected current calibration with MRC04's shared native-TOML reader, without duplicating its key verification.

"bigcherry workload run" publishes one immutable record per run using manifest bytes captured before measurement, build metadata, a native feature-state snapshot, host facts, original measured values, power and argv. There is no results index and no reader of pre-record rows; "bigcherry results list" shows verified run records only.

The external llama.cpp "bench/run_bench.py" is intentionally not changed in a different repository. The platform-managed entrypoint is:

    module bigcherry.bench.llamacpp_runner (repository tools on PYTHONPATH): --runner-root "$BC_LLAMACPP_ROOT" --expected-identity admitted-va25.json --binary-path /path/to/llama-bench -- <run_bench arguments>

It drives the original benchmark and intercepts the existing result append boundary to write exactly one content-addressed record per measured row; duplicate per-family JSON and legacy SQLite sync are suppressed. Direct external invocations are outside the managed migration. Numeric "pp*_tps" and "tg*_tps" metrics map to prefill/decode without changing values. Unmapped quantities such as PPL or standard deviation remain byte-for-byte as the provenance source row, not falsely relabeled as prefill or decode.

Typed HI47 measurement sidecars can be published with "python -m bigcherry.experiment.bundle record BUNDLE --measurement-file BUNDLE/measurement.json" or with the managed "run --measurement-file measurement.json" option. The sidecar declares "identity", "workload", phase-separated "metrics", and VA25 "expected"/"observed". Its bytes, bundle, stdout/stderr, and process observation are stored by SHA-256 and a one-time "record.sha256" pointer is created. Process-only bundles without measurements remain unchanged.

The optional MRC01 "validity" field contains state valid/invalid/unverified and reason codes. VA25's canonical comparator supplies validity, not intent. MRC05 refuses explicitly invalid or unverified measurements for baseline, compare and history; records that predate the validity field remain readable. `metric_specs` (direction and gate for every metric) is required in every record; a record without it is refused when published, and publishing code must pass it. Unattested Radiance workload runs are recorded as unverified and cannot activate new gates; missing execution evidence is not invented from the manifest or rig label.

**MRC06 native-lab sidecar ingestion.** Radiance operators can publish a typed
non-calibration measurement from an explicitly *stopped* native owner with
`bigcherry results ingest-native-lab --run-dir RUN --measurement-file RUN/lab-measurement.json --work-root ROOT`.
The sidecar is within the stopped run directory and contains schema version
1, matching run ID and manifest digest, UTC recorded time, typed workload and
phase-qualified decode/prefill/TTFT metrics. Exact source evidence is pinned to
one immutable MRC01 record; absent VA25 proof is marked unverified. No metrics
are inferred from logs, no independent historical results reader is exposed.

**MRC06 slice 3 — Radiance native lab results.** The native calibration start and `bigcherry results ingest-calibration` were retired together; see LIVE_PLACEMENT.md for how records are produced now.

`needle_probe.py` and `image_probe.py` now require a matching local native
`--run-dir` and optional `--work-root`. Their `--out` is a digest-pointer
JSON object; full results, exact corpus/image bytes and controls are MRC01
content blobs. The S15 A/B driver `rsc02_s15_ab.py` is archived (it needed the retired
Bash launcher); its comparator JSON remains read-only lifecycle/comparison evidence, not a measurement index.

Today's parity records in **UTC** and one verified record by digest:

```sh
bigcherry results list --today --engine radiance --workload parity --json
bigcherry results show RECORD_SHA256 --json
```

These are read-only SHA-verified store views; missing historic manifest
paths are never rebuilt. MRC06 global writer retirement remains open.


`bigcherry.core.retention.plan_cleanup` reports record/blob roots separately
in `content_decisions` and blocks cleanup if store reachability cannot be
verified. Unreferenced blobs are informational, not deletion candidates; the
current run quarantine never deletes content-store objects.

`tools/bigcherry/__main__.py` remains a supported compatibility entrypoint.
It delegates the public command surface to `bigcherry.cli`, but it still has
some residual helper logic; treat it as a migration boundary, not as a
second place to add new command behavior.

## Rationalisation state

The TR00–TR18 tooling-rationalisation program is complete. The completed
planning records under `docs/planning/completed/rationalisation/` describe the
history and acceptance evidence; they are not a live status dashboard. The
stable facts that agents need are:

- Canonical domain packages are established under `tools/bigcherry/` and
  permanent tests are organized under `tools/tests/`.
- The current disposition control plane is the 401-row
  [`TOOL_DISPOSITION.md`](TOOL_DISPOSITION.md). The immutable 383-row TR00
  implementation-start snapshot is preserved under
  [`docs/evidence/tooling-rationalisation/TR00/`](../../evidence/tooling-rationalisation/TR00/).
- Three compatibility shims remain deliberately: `patcher`, `inventory`, and
  `replay_cache`. Their removal requires consumer scans, parity/identity
  checks, applicable CLI checks, and an explicit retirement decision.
- Deterministic hygiene diagnostics are available through
  `bigcherry check --quick`. A diagnostic is a finding to investigate, not
  permission to delete or silently reclassify a file.

## Supported no-hardware commands

Run these from the repository root. The `PYTHONPATH=tools` prefix makes the
commands independent of the caller's current Python installation state.

```bash
PYTHONPATH=tools python -m bigcherry --help
PYTHONPATH=tools python -m bigcherry doctor
PYTHONPATH=tools python -m bigcherry check --quick
PYTHONPATH=tools python -m bigcherry check --default
PYTHONPATH=tools python -m bigcherry check --full
PYTHONPATH=tools python -m bigcherry tune-campaign --help
PYTHONPATH=tools python -m bigcherry profile-campaign --help
PYTHONPATH=tools python -m bigcherry experiment-contract --help
PYTHONPATH=tools python -m pytest tools/tests -q
```

`doctor` and `check` are inspection/validation commands. They do not prove
GPU correctness or performance. Use the appropriate patch, campaign, or
experiment procedure for those claims and retain the resulting evidence.

## Hardware campaign entrypoints

The two repeatable campaign commands use the same explicit identity inputs:
platform, model, device set, runtime profile, and run ID. A representative
tuning invocation is:

```bash
PYTHONPATH=tools python -m bigcherry tune-campaign \
  --platform linux-multi \
  --model /path/to/model.gguf \
  --devices 0,1 \
  --runtime-profile production-dual-xtx \
  --run-id example \
  --json
```

`linux-multi` and `production-dual-xtx` are named configuration entries in
`config/recipes.toml`; use a configured name rather than inventing a profile
on the command line. See [`TUNE_CAMPAIGN.md`](TUNE_CAMPAIGN.md) for the
record → tune → correctness → tuning-promotion → replay contract and
measurement requirements. See [`PROFILING.md`](PROFILING.md) for real
rocprofv3 profiling and control-block stability requirements.

The command handlers live in `cli/tuning.py` and `cli/profiling.py`; the
orchestration and receipts live in the corresponding `tuning/` and
`profiling/` workflow modules. Keep parser changes, handler changes, and
workflow changes together with focused tests.

## Compatibility-shim policy

A root module or wrapper is allowed only when it is an intentional
compatibility boundary. It is not a second implementation. New internal
consumers must use the canonical domain package. Before retiring a shim:

1. scan static and dynamic consumers;
2. verify import identity and behavior parity;
3. run the applicable CLI and focused compatibility tests;
4. update the disposition registry and record the retirement evidence; and
5. confirm no historical evidence or supported fixture contract requires it.

Do not delete or recreate a facade merely to make the tree look clean. The
remaining shim tests include
`tools/tests/core/test_compatibility_facades.py`; the retirement conditions
are owned by the rationalisation records and current disposition entry.

## Lab, environment, and test boundaries

`tools/lab/<plan-topic>/` is temporary, plan-owned work. Lab code is not a
Python package: do not add `tools/lab/__init__.py`, do not import lab or tests
from production code, and do not treat lab output as evidence authority. Each
lab topic README records its question, inputs, outputs, runtime requirements,
mutation/safety notes, owner, and disposition. Generated output belongs under
`artifacts/lab/<experiment>/`.

Environment shell setup belongs in `tools/env/`; it does not own product
state, evidence, or validation. Permanent tests belong under
`tools/tests/<domain>/` and are discovered by the repository test command.
Hardware and benchmark results must be explicitly observed on the target
hardware and workload; offline tests only establish local tooling behavior.

## Disposition and hygiene contract

[`TOOL_DISPOSITION.md`](TOOL_DISPOSITION.md) is the current control-plane
registry. It is parsed by `tools/bigcherry/check.py` and records one
disposition for each in-scope row. It is not a raw filesystem snapshot: a
`TRANSITIONAL`, `ARCHIVE`, or other retained row may name an ignored,
historical, or not-present-on-this-checkout path while its ownership decision
is still being carried forward. Do not infer that a row's path is executable
or current merely because it is listed.

The TR00 files under `docs/evidence/tooling-rationalisation/TR00/` are frozen
evidence and must not be edited as a way to change current ownership. Update
the current registry and its rationale when a disposition changes. Record
lab files before retaining them, and do not use an unchecked disposition to
justify moving or deleting tooling.

`bigcherry check --quick` is deterministic, non-mutating, and hardware-free.
Its stable hygiene findings include:

`TR14.TOP_LEVEL_SCRIPT`, `TR14.ENVIRONMENT_SCRIPT`, `TR14.PYTHON_PARSE`,
`TR14.PRODUCTION_LAB_IMPORT`, `TR14.PRODUCTION_TEST_IMPORT`,
`TR14.DOMAIN_CLI_IMPORT`, `TR14.DOMAIN_ANALYSIS_IMPORT`,
`TR14.FIXED_PARENT_DEPTH`, `TR14.ROOT_FACADE`, `TR14.LAB_PACKAGE`,
`TR14.LAB_METADATA`, `TR14.LAB_UNCLASSIFIED`, and
`TR14.DISPOSITION_DELETE_PENDING`.

Warnings and errors require diagnosis against the current tree and relevant
plan item. They are not a license to alter another actor's in-progress work.

## Evidence and acceptance boundaries

Keep ownership distinct:

- `docs/planning/` owns current scope, decisions, status, and plan-level
  acceptance criteria;
- `config/experiment-contracts.toml` owns scientific Experiment Contract
  obligations, thresholds, and acceptance policy;
- `docs/evidence/<run-id>/` owns compact, tracked proof of a particular run;
- `artifacts/<run-id>/` owns large or machine-local outputs and raw traces;
  `run-id` must be the owning plan-item ID, optionally with a short suffix
  (`HI65`, `HI65-pass2`, `2026-08-21-HI35-HI36-27b-r9700`) — never a free-text
  name with no traceable plan-item link. Create these directories through
  `bigcherry.core.paths.evidence_dir(run_id)` rather than inventing a new
  ad hoc top-level name, so the convention lives in one place. `bigcherry
  check --quick`'s `TR14.ARTIFACT_UNTRACEABLE_RUN` finding flags any
  `artifacts/` top-level directory that is neither a recognised structural
  path (`logs`, `lab`, `pin-bump`, `patch-validation`, `release-runs`,
  `release-validation`), a 12-hex revision directory (`artifact_dir()`), nor
  matched by a `docs/evidence/<run-id>/` counterpart (exact-name, textually
  cited as `artifacts/<name>` anywhere under `docs/`/`engines/llamacpp/patches/`, or the same
  slug once a `docs/evidence/YYYY-MM-DD-<slug>/` write-up's date prefix is
  stripped);
- `engines/llamacpp/patches/<patch-id>/` owns patch contracts, fixtures, and patch evidence; and
- `docs/reference/` owns reusable guidance, not live hardware verdicts.

`profile-campaign` is diagnostic. Even a complete, stable profile does not
accept, promote, or validate a patch. The analysis commands `bigcherry impact`
and `bigcherry kernel-fraction` are valid artifact-consuming reports, but
their output also requires controlled inputs and does not replace an A/B,
patch-validation, or plan acceptance gate.

`tune-campaign` promotion means promotion of tuning winners inside that
campaign only. It does not promote a patch, experiment contract, plan item,
release, or production policy. Those decisions require their own identity,
correctness, provenance, and acceptance records.

## Agent/process rules

Use `ag-planning` MCP tools for plan items under `docs/planning/`: create,
update, review, and state transitions. Use `ag-ledger` MCP
`record_change_event` after substantive work, with `status: unreleased` and
`plan-item-ids` when applicable. The ledger is the release record; do not edit
generated release outputs as a substitute. Use MCP source-control tools and
preserve unrelated shared-worktree edits; do not stash, reset, rebase, or
stage another actor's work.

For generated agent surfaces, edit the owning component/configuration and
re-apply it. Do not edit managed blocks in `AGENTS.md` or `CLAUDE.md`
directly.


The Flash-Next parity reference `docs/reference/engines/radiance/baselines/flashnext-4gpu-parity.json`
is a portable `bigcherry results baseline` record over four independent runs of
release 2020.11.2 (one launch each, so distinct run ids): cold prefill, decode at 1
and 6 streams, 8 sessions and two long contexts, each embedded with its SHA-verified
run record and provenance blobs. The rule is spread-from-N-launches with no floor:
each row's tolerance is the measured spread across the launches, and N is four (the
tool's minimum stays two, but two launches that happen to agree give a gate that
flaps). `compare` also needs at least two launches on the measured side; two unseen
launches of the same build passed every row against this file. Re-record it (four
fresh runs, `flashnext-parity.sh baseline`) when the reference release changes.
`flashnext-parity.sh compare --baseline` reads it. The earlier v1.17.0 and 2026-11-19
summary files were not run-record baselines (their manifest and heat bytes were never
captured), could not be read by `compare`, and were removed rather than kept behind a
reader.
