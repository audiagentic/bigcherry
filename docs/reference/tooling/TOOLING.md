# BigCherry tooling

This is the normative map for maintained tooling. Search the existing command,
API, and owning domain before adding a tool. If the work is exploratory or
plan-specific, start in `tools/lab/<plan-topic>/`.

## Current migration state (TR00–TR14)

| Phase | Truthful state |
| --- | --- |
| TR00 | Inventory and disposition map captured in [`TOOL_DISPOSITION.md`](../../planning/active/rationalisation/TOOL_DISPOSITION.md). It is a baseline, not proof that every listed move or delete is complete. |
| TR01–TR10 | The CLI split and canonical release, source, build, experiment, patch, campaign, tuning, core, and analysis domains are established. Root import paths still exist where compatibility or unresolved ownership requires them. |
| TR11 / RA35–RA36 | All in-scope test modules are classified under domain packages (`build`, `patch`, `campaign`, `tuning`, `core`, `analysis`, `integration`, `hardware`, `fixtures`, `source`, and `release`); discovery remains the acceptance gate. |
| TR12 / RA37 | Analysis implementations are under `tools/bigcherry/analysis/`; six obsolete facades plus the candidate-report root facade were removed, with three analysis consumers/facades retained or migrated as documented. RA37 remains in progress pending full check-tier closure. |
| TR13 / RA38 | Verified consumers were migrated and 55 root facades retired. Three compatibility shims remain (`patcher`, `inventory`, `replay_cache`) with owners and retirement conditions; RA38 remains in progress. |
| TR14 / RA39 | Deterministic hygiene diagnostics are registered in `bigcherry check --quick`; RA39 remains in progress because current findings and unrelated check blockers are intentionally visible. |
| TR15 / RA40 | This documentation reconciliation. It does not close RA37–RA39 or claim unresolved phases complete. |

## Product workflows and owning domains

- **Release:** `bigcherry repin`, `pin-status`, `pull`, `audit`, and release validation/records; implementation owner `bigcherry.release`.
- **Source:** source audit, identity, upstream, and workspace lifecycle; owner `bigcherry.source`.
- **Build:** generated trees, toolchain, compile checks, and builds; owner `bigcherry.build`.
- **Patches:** patch discovery, lifecycle, application, validation, and evidence; owner `bigcherry.patch`. Production patches are package directories; legacy flat discovery remains only for compatibility fixtures. `patch.toml` is metadata authority and `validation.toml` configures validation.
- **Campaigns:** planner, lanes, build/smoke/comparison/benchmark orchestration; owner `bigcherry.campaign`. Hardware campaigns are explicit.
- **Experiments:** contracts and bundles; owner `bigcherry.experiment`. Contract identity, inputs, outputs, and state transitions are scientific authority.
- **Tuning/replay:** catalog, journal, promotion, correctness, ranking, and replay; owner `bigcherry.tuning`. A measured result is not evidence without provenance and identity bindings.
- **Core:** shared paths, context, configuration, artifacts, provenance, and pipeline foundations; owner `bigcherry.core`.
- **Analysis:** reusable offline reports; owner `bigcherry.analysis`. Product workflows must not depend on analysis by default.

`tools/bigcherry/__main__.py` remains the supported compatibility entrypoint;
the parser and command presentation live in `bigcherry.cli`.

## Compatibility-shim policy

Root modules such as `tools/bigcherry/releases.py` may be retained as
compatibility facades. A facade is not a second implementation: new internal
consumers use the canonical domain, and a facade may be removed only after
static/dynamic consumer scans, import-identity/parity checks, CLI checks where
applicable, and an explicit disposition prove it is safe. Do not delete or
recreate a facade merely to make the tree look clean. RA38 tracks the remaining
inventory and its retirement evidence.

## Diagnostics and hygiene

`bigcherry check` is deterministic, non-mutating, hardware-free, and does not
launch ROCm builds, models, or campaigns. `doctor` is also inspection-only.
TR14 findings use stable codes, including:

`TR14.TOP_LEVEL_SCRIPT`, `TR14.ENVIRONMENT_SCRIPT`, `TR14.PYTHON_PARSE`,
`TR14.PRODUCTION_LAB_IMPORT`, `TR14.PRODUCTION_TEST_IMPORT`,
`TR14.DOMAIN_CLI_IMPORT`, `TR14.DOMAIN_ANALYSIS_IMPORT`,
`TR14.FIXED_PARENT_DEPTH`, `TR14.ROOT_FACADE`, `TR14.LAB_PACKAGE`,
`TR14.LAB_METADATA`, `TR14.LAB_UNCLASSIFIED`, and
`TR14.DISPOSITION_DELETE_PENDING`.

Warnings and errors are findings, not permission to silently delete or
reclassify files. Current findings are deliberately documented in RA39:
TR14 hygiene findings, 7 shared `overlay.vendor_sync` files, and any remaining
platform/toolchain limitations from the latest recorded validation run.

## Lab, environment, and tests

`tools/lab/<plan-topic>/` is temporary plan-owned work. Lab code is not a
Python package, must not be imported by production, and is never evidence
authority. Each lab topic records its question, inputs, outputs, requirements,
mutation/safety notes, and disposition. Generated lab outputs belong under
`artifacts/lab/<experiment>/`. Environment shell setup belongs in `tools/env/`
and does not own product state, evidence, or validation.

Permanent tests live under domain packages beneath `tools/tests/`; each
package has a discovery marker and nested path assumptions are validated.
Hardware and benchmark results must be explicitly observed.

## Evidence and acceptance boundaries

PA04 software implementation is complete, but acceptance remains open pending
isolated campaign execution and synchronization of shared `overlay.vendor_sync`
state; do not overwrite or stage those external files. PA05 remains in progress:
RD19 is demoted from `validated` to `untested` pending future qualifying HI83
evidence, but the plan/acceptance record is not closed. No hardware evidence
may be inferred from either status.

## Agent/process rules

Use `ag-planning` MCP tools for plan items under `docs/planning/`: create,
update, review, and state transitions. Use `ag-ledger` MCP
`record_change_event` after substantive work, with `status: unreleased` and
`plan-item-ids` when applicable. The ledger is the release record; do not edit
generated release outputs as a substitute. Use MCP source-control tools and
preserve unrelated shared-worktree edits; do not stash, reset, rebase, or stage
another actor's work.

For generated agent surfaces, edit the owning component/configuration and
re-apply it. Do not edit managed blocks in `AGENTS.md` or `CLAUDE.md` directly.
