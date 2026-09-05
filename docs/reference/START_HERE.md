# Start here — new agents

This is the shortest route from a fresh context to useful, safe work in
bigcherry. It is an orientation guide, not a status report: current work and
acceptance state live in the planning records under `docs/planning/`.

## What this project is

bigcherry is a measured-dispatch autotuner for llama.cpp, focused primarily on
AMD GPU backends. It is maintained as a release-tolerant overlay instead of a
long-lived llama.cpp fork:

- `src/` mirrors the llama.cpp tree and contains whole files owned by
  bigcherry.
- `patches/<patch-id>/` contains anchored edits to files owned by upstream,
  together with the patch's metadata, validation, fixtures, and evidence.
- `vendor/llama.cpp/` is the real upstream checkout to which the overlay is
  applied and from which builds run.

The normal lifecycle is pull an upstream revision, audit it, check that patch
anchors still apply, apply the overlay, generate the candidate catalogue, and
build a named lane. Each stage is idempotent and later stages fail closed when
their prerequisites have not passed.

The autotuner observes the matrix-operation signatures exercised by a real
workload, measures eligible implementations on real hardware, promotes proven
winners, and produces replay builds. Production replay does not benchmark and
does not depend on SQLite; it uses a compact cache and falls back to native
dispatch when necessary.

## Your first ten minutes

1. Read the repository-level [`AGENTS.md`](../../AGENTS.md). Its shared-tree,
   planning, ledger, and execution-profile rules are mandatory.
2. Read the project [`README.md`](../../README.md) for the overlay and candidate
   model, then [`GETTING_STARTED.md`](../../GETTING_STARTED.md) for commands and
   operational gotchas.
3. Read [`docs/README.md`](../README.md) before creating or moving documentation.
   It defines which material belongs in reference, planning, evidence,
   artifacts, patches, and fixtures.
4. Use the `ag-planning` MCP tools to list active plans and open the plan item
   you are working on. Do not infer current status from architecture snapshots,
   archived handoffs, or filenames.
5. Inspect the owning code and its tests before editing. If the work touches
   `tools/bigcherry`, read [`tooling/TOOLING.md`](tooling/TOOLING.md) before
   adding, moving, or retiring a module, script, lab, or compatibility shim.
6. Run a local baseline appropriate to the task. The usual no-GPU checks are:

   ```bash
   PYTHONPATH=tools python -m bigcherry doctor
   PYTHONPATH=tools python -m bigcherry check --default
   cd tools && python -m pytest ../tools/tests -q
   ```

   Use a narrower targeted test while iterating, then expand validation in
   proportion to the change.

## Sources of truth

When two documents appear to disagree, use this precedence:

1. `AGENTS.md` for agent process and source-control rules.
2. `docs/standards/` for normative engineering standards.
3. The relevant active plan item for current scope, decisions, state, and
   acceptance criteria.
4. `docs/reference/` for maintained cross-cutting architecture and operations.
5. Patch-owned files under `patches/<patch-id>/` for a patch's contract and
   evidence.
6. Tracked `docs/evidence/` bundles and transient `artifacts/` outputs for what
   a particular run actually demonstrated.
7. `docs/archive/` and completed plan records for history only.

The release ledger is the authoritative release record. Check it before
editing release artifacts, and record an unreleased change event after
substantive implementation work.

## Find the right owner

| If you are working on... | Start with... |
| --- | --- |
| Project architecture or dispatch concepts | [`architecture/OVERVIEW.md`](architecture/OVERVIEW.md), then the relevant architecture document |
| Building or bumping llama.cpp | [`build/BUILD.md`](build/BUILD.md) and [`build/PIN_BUMP.md`](build/PIN_BUMP.md) |
| Tests, tuning, coverage, or replay | [`testing/TEST.md`](testing/TEST.md) |
| A patch | [`patches/PATCH_SYSTEM.md`](patches/PATCH_SYSTEM.md), then the owning `patches/<patch-id>/` package |
| A new or changed patch | [`patches/PATCH_AUTHORING.md`](patches/PATCH_AUTHORING.md) and [`patches/PATCH_VALIDATION.md`](patches/PATCH_VALIDATION.md) |
| Python tooling or a tooling move/retirement | [`tooling/TOOLING.md`](tooling/TOOLING.md), then [`tooling/TOOL_DISPOSITION.md`](tooling/TOOL_DISPOSITION.md) |
| A full tune or profiling run | [`tooling/TUNE_CAMPAIGN.md`](tooling/TUNE_CAMPAIGN.md) or [`tooling/PROFILING.md`](tooling/PROFILING.md) |
| An experiment contract | [`experiments/EXPERIMENT_CONTRACT.md`](experiments/EXPERIMENT_CONTRACT.md) |
| Current or proposed work | The matching item returned by the `ag-planning` MCP tools |

The complete maintained-reference index is
[`docs/reference/README.md`](README.md).

## Repository map

| Path | Role |
| --- | --- |
| `tools/bigcherry/` | Python CLI and workflow/domain implementations |
| `tools/tests/` | Offline tooling tests and permanent deterministic fixtures |
| `src/` | New files overlaid into llama.cpp at matching paths |
| `patches/` | Packaged anchored edits to upstream-owned files |
| `config/recipes.toml` | Source, build, platform, and campaign recipes |
| `sql/dispatch-db.sql` | Record/tune persistence schema |
| `vendor/llama.cpp/` | Mutable upstream working checkout; not a scratch directory |
| `docs/planning/` | Current and completed plan items, decisions, and reviews |
| `docs/reference/` | Maintained cross-cutting guidance |
| `docs/evidence/` | Compact tracked evidence for reproducible validation |
| `artifacts/` | Large, transient, or machine-local run outputs |

## Working safely in the shared tree

- Do not use raw `git` or GitHub APIs; use the repository's MCP source-control
  tools.
- Never use `git stash`, `git reset`, or `git rebase`. Another agent's live
  changes may share this checkout.
- Inspect the exact files you intend to change and preserve unrelated work.
- Do not casually clean or replace `vendor/llama.cpp/`; builds use it in place.
- Use the planning MCP tools for multi-step work and state transitions. Close
  incorporated reviews, but complete a plan item only after implementation and
  validation are genuinely done.
- Keep generated release outputs synchronized through the ledger process; do
  not edit only the generated artifact.

## Hardware and evidence discipline

Local tests establish Python/tooling behavior; they do not prove GPU kernel
correctness or performance. Hardware claims require observed evidence from the
target machine and workload.

Before using a shared ROCm host, check GPU memory and active processes with:

```bash
rocm-smi --showmeminfo vram --showpids
```

Do not kill an existing process without explicit authorization. Tune mode needs
more VRAM than native or replay, and record/tune data is flushed only on clean
server shutdown. Use the opt-in shutdown endpoint described in
[`GETTING_STARTED.md`](../../GETTING_STARTED.md).

Keep evidence with its owner:

- patch-specific evidence in `patches/<patch-id>/`;
- compact, reproducible run evidence in `docs/evidence/<run-id>/`;
- large raw traces and machine-local outputs in `artifacts/<run-id>/`;
- reusable cross-cutting conclusions in `docs/reference/`;
- current progress and decisions in the relevant plan item.

## Before handing work back

1. Re-read the diff through the MCP source-control tools and ensure it contains
   only your intended files.
2. Run targeted validation and the appropriate broader check; report exactly
   what ran and any hardware limitations.
3. Update the relevant plan item and close handled reviews when the task is
   plan-backed.
4. Record the substantive change in the release ledger with its change class,
   files, technical summary, user-facing summary candidate, `unreleased`
   status, and plan-item IDs when applicable.
5. Leave the next agent a concrete result: what changed, what was verified,
   what remains, and where the evidence lives.
