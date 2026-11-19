# BigCherry tooling

This is the normative map for maintained tooling. Before adding, moving, or
retiring a module, script, lab experiment, or compatibility shim, search for
the existing command/API and read the owning domain's tests. If the work is
exploratory or plan-specific, start in `tools/lab/<plan-topic>/` and keep it
outside the production package until it has an explicit disposition.

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
| Profiling/diagnostics | `bigcherry.profiling` | Real-hardware diagnostic harnesses such as rocprofv3 capture and RCCL qualification |
| Permanent tests | `tools/tests/<domain>/` | Deterministic offline tests organized with the implementation owner |

The normal call chain is:

```text
bigcherry CLI -> cli/main.py parser -> cli/<domain>.py handler
              -> <domain>/workflow.py or <domain> implementation
              -> tests, receipts, evidence, or artifacts owned by that workflow
```

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
a valid cap. Use `run_records.with_host_facts(record)` in Python platform
adapters and `run_manifest.new_manifest(host=...)` or
`run_manifest.new_manifest(collect_host_facts=True)` for an explicit
snapshot; callers opt in to collection at run creation rather than
performing extra sysfs/tool queries on unrelated lifecycle operations.
The workloads tool imports its card/power sampling readers from this core
module. MRC01's manifest edges and results CLI, and MRC06's producer
migrations, are outside these slices.

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

### Portable results: baseline, compare, history (MRC05 slice 1)

`bigcherry.core.results` owns offline baseline gates, comparisons and history
over verified MRC01 run records. Records may include `session_id` for
same-session A/B and `metric_specs` for each phase-qualified metric:
`direction: higher|lower`, `gated: true|false`, optional `repeat_groups`
(samples of the **same prompt**, not distinct prompts). Existing records without
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
PYTHONPATH=tools python -m bigcherry results baseline --engine radiance --layout flashnext-4gpu --workload parity --name reference --to reference.json
PYTHONPATH=tools python -m bigcherry results compare --baseline reference.json --since 2026-11-20T00:00:00Z
PYTHONPATH=tools python -m bigcherry results history --engine radiance --layout flashnext-4gpu --workload parity --metric decode.tokens_per_second --baseline reference.json --json
```

For controlled cohorts, repeat `--record SHA256` on baseline/compare, or
filter with `--since`/`--until`. `--work-root` selects the record store.
Compare exits 1 for regression/missing gated metrics, 2 for invalid input,
0 for a clean gate. MRC06 will migrate producers; the Radiance lab CLI stays.

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

