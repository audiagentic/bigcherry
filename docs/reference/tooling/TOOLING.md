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
| Campaigns | `bigcherry.campaign` | Build/smoke/comparison/benchmark orchestration and campaign lanes |
| Experiments | `bigcherry.experiment` | Contracts, identities, bundles, and experiment state transitions |
| Tuning/replay | `bigcherry.tuning` | Catalog, journal, measurement, correctness, promotion, ranking, and replay |
| Core | `bigcherry.core` | Paths, context, configuration, artifacts, provenance, and pipeline foundations |
| Analysis | `bigcherry.analysis` | Reusable offline reports and derived views; not production selection logic |
| Profiling/diagnostics | `bigcherry.profiling` | Real-hardware diagnostic harnesses such as rocprofv3 capture and RCCL qualification |
| Permanent tests | `tools/tests/<domain>/` | Deterministic offline tests organized with the implementation owner |

The normal call chain is:

```text
bigcherry CLI -> cli/main.py parser -> cli/<domain>.py handler
              -> <domain>/workflow.py or <domain> implementation
              -> tests, receipts, evidence, or artifacts owned by that workflow
```

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
- The current disposition control plane is the 397-row
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
- `patches/<patch-id>/` owns patch contracts, fixtures, and patch evidence; and
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

