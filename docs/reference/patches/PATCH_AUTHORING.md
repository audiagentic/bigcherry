# Patch authoring

This is the creation checklist for a new patch or a patch whose anchored
implementation is changing. Production patches are package directories. The
package is the unit of identity, review, validation, and evidence ownership.

For the system-wide lifecycle and command map, start with
[`PATCH_SYSTEM.md`](PATCH_SYSTEM.md). For the evidence and hardware gates,
continue with [`PATCH_VALIDATION.md`](PATCH_VALIDATION.md).

## Before writing files

1. Read the matching plan item with `ag-planning`. The plan item defines the
   work boundary and current decision; it is not a substitute for evidence.
2. Identify the upstream revision and the canonical source composition that
   will carry the patch. Check existing topology and conflicts:

   ```text
   PYTHONPATH=tools python -m bigcherry patches --source <source-name>
   PYTHONPATH=tools python -m bigcherry patch-explain <patch-id>
   PYTHONPATH=tools python -m bigcherry patch-graph --roots <patch-id>
   ```

3. If the patch is experimental or makes a correctness/performance claim,
   locate or author its Experiment Contract before writing a validation
   adapter. The contract is the scientific authority for the hypothesis,
   workload, controls, architectures, thresholds, and acceptance criteria.
4. Decide the intended initial state. New candidates normally start with
   `state = "untested"`; a patch must not enter a production recipe that
   requires `validated` until the required evidence and review exist.

Do not make a patch look validated just because it applies cleanly. Apply/build
proof and scientific qualification are separate gates.

## Package layout

Use this shape:

```text
patches/<patch-id>/
    patch.toml
    patch.py
    SUMMARY.md

    README.md              # required for validation-ready status
    validation.toml        # required for a new validation execution
    validation/             # optional custom validators/fixtures
        checks.py
        fixtures/
    evidence/
        validation.json    # written by the evidence writer/campaign
```

Large logs, build trees, binaries, models, and raw measurements belong under
`artifacts/patch-validation/<patch-id>/<campaign-identity>/`, not in the
package. Do not use `docs/evidence/` for a single patch's validation authority;
that directory is for cross-cutting evidence narratives.

`patches/_template/` is a documentation template, not a registered patch.
Directories or files below a reserved component beginning with `_` are not
discoverable. Do not add a second patch implementation, campaign engine, or
evidence framework; extend the owning `bigcherry.patch` domain after reading
[`../tooling/TOOLING.md`](../tooling/TOOLING.md).

## `patch.toml`: machine metadata

The minimum packaged manifest is:

```toml
schema = 1
id = "1204_example_patch"
order = 1204
group = "rdna-boosts"
state = "untested"

plan-ids = ["RD08"]
requires = []
conflicts = []

kind = "enhancement"
origin = "external-fork"
backend = "hip"
external-source = "stew675-rdna-boosts"
validation-architectures = ["gfx1100", "gfx1201"]
experiment-contract = "RD08-Q6K-MMVQ-VDR2"
```

Rules that the registry enforces:

- The directory basename and `id` must match exactly.
- The ID is repository-global and begins with a numeric order prefix, such as
  `1204_...`; duplicate IDs fail closed.
- `order` must match the numeric ID prefix. It controls stable application
  order; dependency order still comes from `requires`.
- `state` is one of `untested`, `validated`, `rejected`, or `superseded`.
- `requires` and `conflicts` contain patch IDs, not directory paths. A
  dependency is included in the resolved composition; a conflict blocks it.
- Use `plan-ids` for one or more plan bindings. Existing patches may also use
  the compatibility `plan-item` field; do not create conflicting singular and
  plural values.
- `experiment-contract` is the compatibility form for one contract;
  `experiment-contracts = ["..."]` is the plural form. Never declare both,
  and never use an empty plural list.
- `validation-architectures` states the architectures required for the
  validation claim. For `ported-validated`, it must be non-empty and match
  the contract-required set where the contract declares one.
- `kind`, `origin`, and `backend` use the registry vocabulary. Do not encode
  scientific thresholds, hypotheses, or workload acceptance in this file.

For a packaged patch, `patch.toml` is the metadata authority. Do not duplicate
its state, group, order, or dependency fields in `patches/catalog.toml` or in
`patch.py`.

## `patch.py`: anchored implementation

`patch.py` should define the patch's anchored `FilePatch`/`Edit` transformation
and, where applicable, a literal `PROVENANCE` record that can be checked
against `external-sources.toml`. Keep the implementation deterministic and
package-local.

Every edit should make its safety assumptions explicit:

```python
FilePatch(
    path="src/target.cu",
    description="Add the guarded fast path",
    edits=(
        Edit(
            id="add-fast-path",
            anchor=r"^static __device__.*target_kernel",
            text="...",
            mode="insert_after",
            guard=r"BIGCHERRY fast path",
            expect_matches=1,
            rationale="Attach to the unique target kernel declaration",
        ),
    ),
)
```

Authoring rules:

- Use an anchor that identifies the intended construct, not a line number.
- Set an explicit `guard` for the output and an explicit
  `expect_matches=1` unless multiple matches are intentional and handled by
  `replace_all`/`occurrence`.
- Explain why the anchor is stable in `rationale`.
- Anchors are matched against comment/string-noise-stripped source where the
  file dialect requires it; do not anchor on comments or string literals.
- Keep edit target paths relative and contained. Never use an absolute path,
  `..`, or a symlink escape.
- Use `applies_if` only for a genuine alternate upstream shape. It must not
  turn a missing anchor into a silent success. Provide separate shape-gated
  edits and test each one.
- A second application must be an explicit `already-applied` no-op. An
  ambiguous anchor, missing guard, unexpected match count, or partial write is
  a failure.
- Do not make the patch implementation reach outside the target checkout or
  silently mutate unrelated files.

For an external backport, preserve the source commit, snapshot/base identity,
adaptations, and plan binding in `PROVENANCE`. The branch name is a locator,
not the identity; use the commit SHA and run the external-source checks before
promotion or a pin bump.

## `SUMMARY.md`: release-facing description

`SUMMARY.md` is required in the real repository because the lint gate checks
it and `patch-doc` merges it into release documentation. Its first three
metadata lines must be contiguous and exactly match `patch.toml`:

```markdown
# 1204_example_patch

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD08

## What it does

Short description of the source change.

## Why

The engineering reason, without claiming unmeasured benefit.

## Upstream

Origin commit or local ownership and any adaptation note.
```

The summary is not evidence and must not claim a benchmark result that the
bound contract/evidence record does not support. After changing `state`,
`group`, or plan bindings, update this header in the same change and rerun
`patch-lint`.

## Validation package files

When a tracked logical change is `ported-benched`, `ported-validated`, or
`deferred-hardware`, the package policy requires `README.md`, `validation.toml`,
and a resolvable Experiment Contract. A grandfathered legacy shape may be
lint-tolerated once, but grandfathering never authorizes a new validation run.

`README.md` should contain:

- patch ID, plan item(s), and bound contract ID(s);
- target hardware/architectures, backend, prerequisites, and workload scope;
- the exact validation command or a pointer to the patch-specific command;
- what any custom check measures;
- control versus subject composition, including any patch-specific A/B;
- known limitations and blockers;
- artifact and compact-evidence locations.

Keep the scientific hypothesis and acceptance thresholds in the Experiment
Contract. The README can explain implementation scope and point to the
contract; it must not become a second threshold authority.

`validation.toml` is an execution adapter. Use the actual schema:

```toml
schema = 1

[[check]]
id = "apply"
capability = "apply"
validator = "apply"
required = true

[[check]]
id = "build"
capability = "build"
validator = "build"
required = true

# Add only producers required by the bound contract or this adapter.
[[check]]
id = "activation"
capability = "activation"
validator = "trace-marker"
required = true
marker-regex = "BIGCHERRY_PATCH_HIT patch=1204_example_patch path=fast-path"
```

`apply` and `build` are universal. Contract-required correctness, activation,
performance, controls, architecture, and other obligations must each have a
capable required producer. An adapter cannot remove a contract obligation.
Unknown validators and missing producers fail closed.

Built-in validators currently include `apply`, `build`, `compile-option`,
`runtime-smoke`, `architecture`, `benchmark`, `autotune-campaign`,
`backend-ops`, and `trace-marker`. A custom validator uses:

```toml
validator = "custom"
callable = "validation/checks.py:check_example"
```

The callable is a package-contained file path plus function name, not a dotted
import path. It must be a synchronous function with exactly `check(ctx)` and
must return a framework `ValidationResult` for the declared check. Prefer a
built-in validator; custom code is rare and must have focused positive,
negative, and tamper tests.

## Authoring tests

Before hardware, test the transformation in an isolated immutable fixture or
temporary checkout. Cover at least:

- the expected upstream shape applies once;
- a second application is `already-applied` and does not change bytes;
- a missing anchor, ambiguous anchor, wrong guard, and unexpected match count
  fail closed;
- alternate shapes are independently shape-gated and do not weaken failure;
- path traversal and symlink targets are rejected;
- dependencies and conflicts resolve as declared;
- external provenance and summary metadata agree;
- custom validators return the correct check/capability and cannot escape the
  package;
- tampered evidence, stale pin identity, missing artifacts, and missing named
  checks do not qualify a patch.

Run the repository gates while iterating:

```text
PYTHONPATH=tools python -m bigcherry patch-lint --json
PYTHONPATH=tools python -m bigcherry check --quick
PYTHONPATH=tools python -m unittest discover -s tools/tests
```

Then use `patch-rebase-check` and the workflow in
[`PATCH_VALIDATION.md`](PATCH_VALIDATION.md). Do not use a hand-run benchmark
or a successful import as a substitute for the real apply/build/contract
checks.

## Legacy flat-to-package migration

Do this only when migrating an existing compatibility fixture; new patches
start packaged.

1. Freeze the same immutable upstream base and the same patch ID.
2. Apply the legacy representation to one isolated copy and the package
   representation to another. Include the same source composition and overlay
   conditions.
3. Require both applications to succeed with the same expected edit results,
   then compare the complete source-tree identity, not just a hand-picked
   file or a similar diff.
4. Preserve or migrate validation evidence only when its implementation,
   validation, contract, framework, source, and hardware identities still
   match. Otherwise record it as historical and revalidate.
5. Remove the old representation only after the equivalence evidence is
   recorded and the registry/lint/test gates pass. Do not leave duplicate IDs.

Use the repository's source-control MCP and preserve other shared-worktree
changes. Never use `git stash`, `git reset`, `git rebase`, or a destructive
cleanup to make the migration appear clean.
