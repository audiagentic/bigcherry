# Patch refactor and rebase runbook

Use this page when an upstream pin moves, an existing patch must be
reconciled, or a legacy flat fixture is being migrated into the package-only
layout. This is an operational runbook, not permission to run `git rebase`:
the shared-tree policy forbids `git stash`, `git reset`, and `git rebase`.

The locked design and historical acceptance run sheets remain in
[`../../planning/active/patch-system/PATCH_REFACTOR_RUNBOOK.md`](../../planning/active/patch-system/PATCH_REFACTOR_RUNBOOK.md).
Use the current registry/rebase implementation and the active plan item for
live behavior and status; do not treat an old run-sheet criterion as evidence
that has already been completed.

## When this runbook applies

Use it for:

- an upstream pin bump or a changed `vendor/llama.cpp` revision;
- an anchored patch whose implementation or target upstream shape changed;
- a conflict, missing anchor, or unexpected no-op in a source selection;
- isolating one known-broken patch while applying the rest of a source;
- migrating a legacy flat compatibility fixture to a package directory.

It is not needed for merely reading existing evidence. Use
[`PATCH_VALIDATION.md`](PATCH_VALIDATION.md) for new hardware validation and
[`PATCH_AUTHORING.md`](PATCH_AUTHORING.md) for a new package.

## 1. Freeze the identities you are changing

Before editing, record the old and target upstream revisions, the BigCherry
revision, the canonical source name, overlay identity, patch IDs, and each
implementation digest. Inspect the exact composition:

```text
PYTHONPATH=tools python -m bigcherry patches --source <source-name>
PYTHONPATH=tools python -m bigcherry patch-explain <patch-id>
PYTHONPATH=tools python -m bigcherry patch-graph --roots <patch-id>
```

Do not reuse a report, build directory, or evidence record from a different
pin, source selection, patch digest, or validation identity. A symbolic tag is
not enough for evidence; the resolved commit SHA is part of the identity.

## 2. Probe applicability without mutating the vendor checkout

For the exact source that will be applied:

```text
PYTHONPATH=tools python -m bigcherry patch-rebase-check \
    --source <source-name> --json <rebase-report.json>
```

For release-wide coverage, probe every non-retired registry patch:

```text
PYTHONPATH=tools python -m bigcherry patch-rebase-check \
    --all --json <all-patches-rebase-report.json>
```

The command uses isolated detached worktrees and is observational. It does
not change `vendor/llama.cpp`, release stage, patch metadata, or evidence.
Use the report's exact statuses:

```text
CLEAN
CLEAN_NOOP
NOT_APPLICABLE_BY_DESIGN
FAILED_NEEDS_RECONCILIATION
BLOCKED_BY_DEPENDENCY
QUARANTINED
```

The first three are apply-safe outcomes. Any other status requires a
reconciliation decision. Do not interpret a failed patch as `rejected` merely
because its anchor no longer matches; applicability to one pin is a
revision-specific fact.

## 3. Reconcile a failed patch

For each failure:

1. Read the bounded report context and the patch's `rationale`, `guard`, and
   expected match count.
2. Inspect the new upstream shape at the exact target revision in an isolated
   worktree. Do not edit the shared vendor checkout as a scratch copy.
3. Update the anchored edit or add a genuinely shape-gated alternative with
   `applies_if`. Never turn an unexpected missing anchor into a silent
   success.
4. Test both the old/new intended shapes, the guard, match count, path
   containment, and second-application no-op.
5. Rerun `patch-lint`, `check --quick`, and the same `patch-rebase-check`.
6. Re-run the applicable validation campaign if `patch.py` or any validation
   identity changed. Old evidence is then historical for the changed identity.

If the patch should no longer be carried because upstream independently
contains it, make an explicit `state = "superseded"` decision and update
`SUMMARY.md`. If the implementation was tested and found incorrect, make an
explicit `state = "rejected"` decision with the evidence/review. Neither
decision should be inferred from a transient rebase failure.

## 4. Apply a fresh known-good subset

When a source has a mixture of clean and failed patches, a fresh report may be
used to apply only the report-authorised subset:

```text
PYTHONPATH=tools python -m bigcherry apply \
    --rebase-report <rebase-report.json> --known-good
```

Rules:

- `--rebase-report` and `--known-good` are required together.
- Do not combine `--rebase-report` with `--source`; the report owns the exact
  logical selection.
- The report must match the current upstream revision, BigCherry revision,
  source/patch-set identity, overlay digest, and patch implementation
  digests. A stale report fails closed.
- A partial reconciliation apply does not advance the release stage.
- Run `audit`, `apply --dry-run`, and the post-apply checks appropriate to the
  release workflow. Never use `--force` to bypass a failed audit just to make
  the subset apply.

If the patch is selected by the recipe, it cannot be excused by a disposition;
the source must be reconciled or the recipe changed through the owning review.

## 5. Use a revision-bound disposition only when eligible

`patch-disposition` is a narrow coverage record for a non-recipe patch that is
known broken at one exact upstream revision. It is not patch rejection, a
validation result, or a permanent waiver. It expires immediately when the
target revision or implementation digest changes.

Record the digest printed in the `--all` rebase report:

```text
PYTHONPATH=tools python -m bigcherry patch-disposition set \
    --patch-id <patch-id> \
    --revision <full-upstream-commit-sha> \
    --digest <implementation-digest> \
    --failure-status FAILED_NEEDS_RECONCILIATION \
    --reason "<specific reproducible reason>" \
    --owner "<responsible owner>" \
    --tracking-item "<plan-or-issue-id>"
```

Inspect or remove records with:

```text
PYTHONPATH=tools python -m bigcherry patch-disposition list
PYTHONPATH=tools python -m bigcherry patch-disposition clear --patch-id <patch-id>
```

The disposition is valid only for the recorded `(patch ID, target revision,
implementation digest)` triple. A recipe-selected patch must still be clean;
do not use a disposition to excuse a required production patch. Include the
owner and tracking item so another agent can resolve the failure.

## 6. Flat-to-package migration gate

New patches must be packaged. This procedure is only for an existing flat
compatibility fixture:

1. Create `patches/<patch-id>/patch.toml`, `patch.py`, and `SUMMARY.md` with
   the same canonical patch ID and declared composition.
2. Freeze one immutable upstream base and identical overlay/source conditions.
3. Apply the flat representation to one isolated tree and the package
   representation to another. Compare complete source-tree identities after
   both succeed; matching selected files or a visually similar diff is not
   enough.
4. Check that implementation digests, patch application results, dependency
   closure, and provenance agree. Reject duplicate IDs.
5. If validation is being migrated, also compare the validation identity and
   bound contract. Any changed validation/framework/contract/source/hardware
   identity requires new evidence.
6. Run `patch-lint`, the package tests, and `patch-rebase-check --all` before
   retiring the old fixture representation. Record the equivalence evidence.

The package's `patch.toml` becomes the metadata authority. Do not keep a
second `catalog.toml` entry for the packaged ID or rely on duplicated
`STATE`/`GROUP` constants in its `patch.py`.

## 7. Evidence invalidation and revalidation

Use this matrix when deciding what must be rerun:

| Change | Old apply/build evidence | Old validation evidence | Required follow-up |
| --- | --- | --- | --- |
| Upstream pin/tag or resolved SHA | Historical | Historical for current qualification | Rebase/apply against the new SHA; rerun current-pin validation as needed |
| `patch.py` bytes or anchored semantics | Invalid for the changed implementation | Invalid | Re-run apply/build and the full affected claim |
| `patch.toml` composition, contract, architecture, or state | Re-check composition | Usually invalid or no longer sufficient | Lint, re-resolve plan, rerun evidence gate; new execution if identity changed |
| `validation.toml` or `validation/` | Reusable only if source/build identity still matches | Invalid | Re-run the validation plan and write a new record |
| Bound Experiment Contract semantics | Reusable only for implementation mechanics | Invalid | Re-run contract-required checks against the new contract hash |
| Validation framework semantic version | Reusable for production source identity | Invalid | Re-run validation under the new framework |
| `SUMMARY.md` prose only | Unchanged | Unchanged | Run summary/lint checks; do not present prose as evidence |

Evidence records are append-only. Do not rewrite an old record's meaning,
change its pin to the new SHA, or delete a failed run to make the catalog
green. Use `patch-verify-evidence --no-legacy-grandfather` to confirm the
current claim.

## 8. Final reconciliation gate

Before handing the patch or pin bump back:

```text
PYTHONPATH=tools python -m bigcherry patch-lint --json
PYTHONPATH=tools python -m bigcherry check --quick
PYTHONPATH=tools python -m bigcherry check --default
PYTHONPATH=tools python -m unittest discover -s tools/tests
PYTHONPATH=tools python -m bigcherry patch-rebase-check --all --json <final-report.json>
PYTHONPATH=tools python -m bigcherry patch-verify-evidence --no-legacy-grandfather
```

Use `check --full` when the active plan/acceptance gate requires it. Before a
hardware run, resolve any `overlay.vendor_sync` or other shared-tree finding
through its owner; do not overwrite, stage, reset, or copy another actor's
files. For changed or newly qualified patches, complete the explicit
campaign and handoff requirements in
[`PATCH_VALIDATION.md`](PATCH_VALIDATION.md).

The final record must name the exact upstream SHA, source selection, report
path, clean/failed/quarantined set, dispositions, changed patch digests,
validation records, and lifecycle decisions. Record the substantive change in
the release ledger and link the applicable planning item.
