# BigCherry patch system

This page is the operational index for patches. Use it to decide which
representation, command, evidence record, and lifecycle update an agent
needs. The detailed validation policy lives in
[`../testing/PATCH_VALIDATION.md`](../testing/PATCH_VALIDATION.md); this page
keeps the whole patch lifecycle visible in one place.

## The production rule

Every new production patch is a package directory:

```text
patches/<patch-id>/
    patch.toml       # machine-readable identity and composition metadata
    patch.py         # anchored implementation
    SUMMARY.md       # short release-facing description; lint checks its header
```

Add these files when the patch is validation-ready:

```text
    README.md        # validation instructions, scope, controls, limitations
    validation.toml  # execution adapter; not a second experiment contract
    validation/      # optional custom checks and immutable fixtures
    evidence/
        validation.json
```

`patches/_template/` and `patches/_shared/` are reserved support directories;
they are not patches. The registry's flat-module reader exists for legacy
compatibility fixtures only. Do not create a new `patches/*.py` production
patch, and do not add a packaged patch to `patches/catalog.toml`.

## Fast path for a new or changed patch

Use this sequence. Stop at the first failing gate and fix or record the
failure; do not skip ahead to hardware or promotion.

1. Read the owning plan item, the bound Experiment Contract (if the patch
   makes an experimental claim), this page, and
   [`PATCH_AUTHORING.md`](PATCH_AUTHORING.md).
2. Inspect the intended source composition:

   ```text
   PYTHONPATH=tools python -m bigcherry patches --source <source-name>
   PYTHONPATH=tools python -m bigcherry patch-explain <patch-id>
   PYTHONPATH=tools python -m bigcherry patch-graph --roots <patch-id>
   ```

3. Create or edit the package. Keep `patch.toml` authoritative for packaged
   metadata and `patch.py` limited to the transformation and provenance.
4. Run the static gates:

   ```text
   PYTHONPATH=tools python -m bigcherry patch-lint --json
   PYTHONPATH=tools python -m bigcherry check --quick
   PYTHONPATH=tools python -m unittest discover -s tools/tests
   ```

5. Check applicability against the exact pinned upstream revision, then apply
   only the named source composition:

   ```text
   PYTHONPATH=tools python -m bigcherry patch-rebase-check \
       --source <source-name> --json <rebase-report.json>
   PYTHONPATH=tools python -m bigcherry apply --source <source-name> --dry-run
   PYTHONPATH=tools python -m bigcherry apply --source <source-name>
   ```

   `apply` requires a passing audit unless an explicitly authorised operator
   uses `--force`. `--force` is not a validation method and must not be used to
   hide a failed audit.

6. For a hardware or performance claim, run the explicit,
   content-addressed validation campaign described in
   [`PATCH_VALIDATION.md`](PATCH_VALIDATION.md). `bigcherry check` and
   `patch-validate` do not run a GPU campaign.
7. Inspect every named check, the control/subject composition, the resolved
   pin SHA, hardware identity, and artifact hashes. Then run:

   ```text
   PYTHONPATH=tools python -m bigcherry patch-verify-evidence <patch-id>
   ```

   Use `--no-legacy-grandfather` when the decision must rely only on current
   evidence. A successful verifier means the evidence qualifies the current
   tracked status; it does not edit `patch.toml` or `external-sources.toml`.
8. Make any lifecycle promotion, demotion, rejection, or supersession a
   separate deliberate metadata change. Update the matching `SUMMARY.md`
   header and rerun `patch-lint`.

## Command map

| Command | What it proves or reports | What it does not do |
| --- | --- | --- |
| `patches [--source NAME]` | Lists packages, metadata, state, and source selection | Does not apply or validate a patch |
| `patch-explain ID` | Shows identity, provenance, dependencies, conflicts, and source/experiment references | Does not prove runtime behaviour |
| `patch-graph [--roots ID]` | Shows `requires`/`conflicts` topology and dependency closure | Does not choose a scientific baseline |
| `patch-lint` | Static package, metadata, summary, adapter, contract-producer, and path checks | Does not inspect current evidence freshness or run patch code |
| `check --quick/--default/--full` | Deterministic local CI; hardware-free and non-mutating | Does not build ROCm targets, launch models, or prove GPU behaviour |
| `patch-rebase-check --source/--all` | Observes whether patches apply to the current upstream revision in isolated worktrees | Does not mutate the vendor checkout or advance release state |
| `apply --source NAME` | Applies the exact canonical source selection after audit | Does not validate a performance claim |
| `apply --rebase-report PATH --known-good` | Applies only the fresh, report-authorised known-good subset | Does not make a stale report current; it refuses identity mismatches |
| `patch-validate [ID]` | Alias for existing-evidence verification | Does not execute a new campaign |
| `patch-verify-evidence [ID]` | Checks current-pin, identity, named-result, artifact, and status qualification | Does not promote or demote metadata |
| `python -m bigcherry.patch.validation_campaign` | Runs the explicit patch validation campaign and writes evidence | Does not change patch lifecycle state automatically |
| `patch-disposition` | Manages a revision- and digest-bound known-broken rebase disposition | Is not a permanent waiver and is not patch rejection |
| `patch-status [--item ID]` | Computes plan/source/package/contract signals | Does not fabricate trigger, correctness, performance, or promotion evidence |
| `patch-doc --source/--all` | Merges `SUMMARY.md` files with the exact selection and pin metadata | Does not validate the summaries' scientific claims |

On Windows PowerShell, set the equivalent environment variable before the
commands, for example `$env:PYTHONPATH = 'tools'`. The repository's hardware
runbooks may use a different launcher (`python3`) on Linux hosts.

## Authorities and identity

Keep each fact in the system that owns it:

| Fact | Authority |
| --- | --- |
| Package identity, state, order, group, dependencies, conflicts, backend, hardware and contract binding | `patch.toml` |
| Anchored source transformation and external-source provenance | `patch.py` |
| Which source/build selection includes the patch | `config/recipes.toml` |
| External fork/upstream tracking and tracked-status history | `external-sources.toml` |
| Hypothesis, workload, controls, architectures, thresholds, and acceptance criteria | Experiment Contract in `config/experiment-contracts.toml` |
| How a contract's evidence is produced | `validation.toml` and patch-local `validation/` |
| What actually ran and was proven | `evidence/validation.json` plus bound artifacts |
| Current work, decisions, reviews, and acceptance state | `docs/planning/` via `ag-planning` |
| Release change record | `ag-ledger` |

For a packaged patch, do not maintain a second metadata authority in
`patches/catalog.toml` or duplicate `STATE`, `GROUP`, or equivalent catalog
fields in `patch.py`. `SUMMARY.md` is human-facing, but its `Status`, `Group`,
and `Plan item` header must agree with `patch.toml`; `patch-lint` checks this.

The registry hashes the implementation and validation identity. Validation
identity includes the validation files, framework semantic version, and bound
contract identity. A change to patch code, validation code/configuration, the
bound contract, or validation framework can invalidate old evidence even when
the raw benchmark output is unchanged.

## Composition and attribution

Patch selection is exact and source-driven. `--source NAME` resolves the
canonical `[source.NAME]` entry from `config/recipes.toml`; it is not a loose
filter over group or state. Dependencies are closed and ordered by the
registry. Unknown, rejected, superseded, conflicting, or dependency-incomplete
compositions fail closed.

For a focal patch `X`, validation must use explicit source compositions:

```text
BASELINE = the named source composition
CONTROL  = BASELINE + X's prerequisites, without X
SUBJECT  = BASELINE + the same prerequisites + X
STOCK    = pristine pinned upstream, contextual only
```

The causal attribution is `SUBJECT` versus `CONTROL`. `SUBJECT` versus
`STOCK` can show total overlay impact but cannot attribute a change to `X` if
other patches are present. If `X` is already in the baseline, conflicts with
it, or a baseline patch depends on it, the comparison is `BLOCKED`; do not
silently remove the dependent patch or substitute a different control.

Do not confuse validation build roles (`control`, `subject`, optionally
`stock`) with tuning-campaign build roles (`tune`, `replay`, `stock`). They
have different provenance meanings even when a campaign reuses a physical
build.

## Two lifecycle axes that agents must keep separate

The package `state` and the external-source `status` answer different
questions:

| Axis | Values | Meaning |
| --- | --- | --- |
| `patch.toml state` | `untested`, `validated`, `rejected`, `superseded` | Whether the patch implementation is accepted for composition; recipes normally require `validated` |
| `external-sources.toml` tracked status | `planned`, `ported-untested`, `ported-benched`, `ported-validated`, `deferred-hardware`, `superseded`, `excluded`, `evidence-only` | Historical/working progress of a tracked logical change and its proof level |
| Campaign/evidence result | `PASS`, `FAIL`, `BLOCKED`, `ERROR` plus named results | What this run actually established; not a lifecycle edit |

An experimental patch can have a package `state = "untested"` while its
tracked logical change is `ported-benched`; that is not automatically a
contradiction. Read both records and the evidence verifier.

### Promotion and demotion rules

- A campaign may print `eligible_for_validated_state = true`, but it never
  edits `patch.toml`, `SUMMARY.md`, or the external-source registry.
- Promote a patch implementation to `state = "validated"` only after the
  applicable current-pin evidence verifier passes and the review accepts the
  actual claim. Update the matching summary header and source-tracking status
  in the same deliberate change when applicable.
- `ported-benched` requires a real current-pin control/subject benchmark with
  build and hardware identities. A required correctness failure forbids this
  qualification; performance alone cannot override it.
- `ported-validated` requires every named required check to pass and a
  non-empty, meaningful `validation-architectures` declaration. One generic
  `correctness` result cannot stand in for multiple named contract checks.
- `deferred-hardware` means the methodology is complete and a structured
  `BLOCKED` result records the unavailable prerequisite. It is not a pass and
  must not be converted to `validated` without the missing execution.
- `rejected` means the candidate was tested and the implementation was not
  accepted. A missing GPU, harness error, stale pin, or inability to reproduce
  a fault is not by itself evidence for rejection.
- `superseded` means the same change is independently present upstream. It is
  different from rejecting a patch as incorrect.
- A pin bump makes prior evidence historical; it does not authorize rewriting
  or deleting the old record. Re-run the current-pin gates and update metadata
  only if the current policy decision changes.

### Transition record and dependency effects

There is no separate automatic patch-state transaction or cascade engine today.
For every manual transition, record a review/plan note and an unreleased
ledger event containing at least:

```text
patch_id
old patch.toml state -> new patch.toml state
old external-source status -> new status (if tracked)
reason and decision owner
upstream pin tag + resolved SHA
implementation/validation digests
qualifying campaign/evidence IDs and verifier result
dependency/conflict impact and affected source recipes
re-promotion criteria, if demoting or deferring
```

If a prerequisite `Y` is demoted, rejected, or superseded, a dependent patch
`X` is not silently demoted by the registry, but `X` is no longer safe to
compose until the dependency is restored or `X` is reworked. Re-run
`patch-graph`, resolve the source again, and make an explicit decision for
`X`; do not leave a production recipe carrying an unresolvable dependency.
Installed/composed state must be treated as invalid until the source is
reconciled. Re-promoting `X` requires the same current-pin evidence and review
as any other promotion, plus a fresh dependency/composition check.

When demoting because of a measured regression or failed required check,
retain the evidence and point to the failing campaign. When demoting only
because evidence is stale, a pin moved, or a harness errored, retain the old
record as historical and state the revalidation condition; do not relabel an
unknown result as a rejection. A previously `rejected` patch must not be
silently revived: either create a corrected patch identity or record an
explicit review that resets it to `untested` before new validation.

## Fail-closed invariants

Never weaken these rules to get a green report:

- `BLOCKED`, `ERROR`, and missing results are not `PASS`.
- A validation-ready experimental package must bind a resolvable Experiment
  Contract. Local non-RD framework packages without external-source bindings
  may execute a package-local adapter with zero contracts; that is not current
  qualification or a throughput claim. Every package must provide producers
  for all universal and bound-contract-required capabilities.
- Required checks must pass individually. `not_applicable` never satisfies a
  required check.
- A benchmark artifact existing is not proof that the target path executed;
  activation and a causal control/subject comparison are required for a
  performance claim.
- A generic tune/replay result, a README statement, or caller-supplied JSON is
  not a substitute for bound evidence and artifact digests.
- Old evidence is append-only. New evidence gets a new campaign identity.
- Hardware validation is explicit, workload-specific, and environment-bound.
  `bigcherry check` must remain safe to run on a laptop or in local CI.

## Read next

- [`PATCH_AUTHORING.md`](PATCH_AUTHORING.md) — package creation, anchored edit
  design, metadata, summaries, tests, and migration checks.
- [`PATCH_VALIDATION.md`](PATCH_VALIDATION.md) — operational validation,
  campaign conditions, evidence interpretation, and lifecycle handoff.
- [`PATCH_REFACTOR_RUNBOOK.md`](PATCH_REFACTOR_RUNBOOK.md) — pin-bump rebase,
  known-good quarantine, dispositions, and flat-to-package migration.
- [`../testing/PATCH_VALIDATION.md`](../testing/PATCH_VALIDATION.md) — complete
  validation-package policy, contract binding, validator semantics, and
  evidence provenance.
- [`../testing/TEST.md`](../testing/TEST.md) — concrete hardware and benchmark
  procedures.
- [`../tooling/TOOLING.md`](../tooling/TOOLING.md) — domain ownership and
  tooling hygiene.
