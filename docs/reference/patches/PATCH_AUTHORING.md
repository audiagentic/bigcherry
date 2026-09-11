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
tags = ["optimization", "gfx1201"]
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
- `tags` is optional and sparse -- see `## Tags` below. Never required to be
  non-empty; a patch with nothing genuinely specific to assert carries none.

For a packaged patch, `patch.toml` is the metadata authority. Do not duplicate
its state, order, or dependency fields in `patches/catalog.toml` or in
`patch.py`.

## Tags

`kind` (`framework` / `diagnostic` / `upstream-backport` / `enhancement`) is
what patch-set composition actually selects on -- small, fixed, required in
practice. `tags` is a separate, optional, sparse, multi-valued field for
BigCherry's own cross-cutting classification and analysis, never a build
input. Apply a tag only where it is genuinely true and specific: an
architecture-independent optimization gets no architecture tag; something
genuinely RDNA4-only gets `rdna4` and nothing else. Most patches carry 0-3
tags, never one from every category out of habit.

**This is the authoritative list.** `tools/bigcherry/patch/registry.py`'s
`PATCH_TAGS` constant is validated against this exact table by
`tools/tests/patch/test_patch_tags_registry.py` -- the two cannot silently
drift. Check here before inventing a new tag name; if nothing fits, add the
new value to both `PATCH_TAGS` and this table in the same change, and
consider a review of the whole list before it grows much further.

### `optimization` carries a real validation obligation (2026-09-11)

Unlike every other tag, `optimization` is not purely a classification aid.
A patch tagged `optimization` makes a performance claim, and a performance
claim can only be trusted against a real comparison to unmodified upstream
llama.cpp -- BigCherry's own internal control-vs-subject A/B (the pattern
`tools/bigcherry/experiment/perplexity.py`'s correctness producers use, and
most RD-series correctness/A-B evidence in this repo) proves the patch
doesn't regress *BigCherry's own baseline*. It does **not** show whether
that baseline itself already gained or lost ground against upstream --
which is the actual question "is this patch worth carrying" depends on.

**Rule**: before tagging a patch `optimization` AND setting
`state = "validated"`, its README.md must document a real 3-arm comparison:
**native (unmodified) llama.cpp**, **BigCherry baseline** (patch excluded),
and **BigCherry + this patch**. `tools/bigcherry/patch/validation_policy.py`'s
`check_performance_evidence()` enforces this structurally in `patch-lint`
(a literal-substring presence check for a phrase like "native llama.cpp" in
the README -- it cannot verify the comparison is real or current, only that
some evidence was written down; a human/GPT reviewer still owns whether
that evidence is honest and sufficient). A patch that is `kind=enhancement`
but makes no performance claim (a correctness fix, a safety guard, a
functional addition) should NOT carry the `optimization` tag and is exempt
from this rule -- see `kind`'s own framework/diagnostic/upstream-backport/
enhancement split for the functional-vs-performance distinction that
doesn't need a tag to express.

| tag | meaning |
| --- | --- |
| `optimization` | Makes an existing path faster. Only add when `kind=enhancement` doesn't already say enough. |
| `tuning` | Improves the autotune/dispatch-selection mechanism itself, not kernel logic. |
| `gfx1100` / `gfx1101` / `gfx1151` / `gfx1201` | Chip-specific. Use only when the patch is genuinely scoped to that one die, not a whole architecture family. |
| `rdna3` / `rdna3.5` / `rdna4` | Architecture-family-wide. Never combine with a specific `gfx****` tag for the same patch -- tag the narrowest true scope. |
| `allreduce` | Touches the internal HIP AllReduce collective path. |
| `p2p` | Touches direct peer-to-peer GPU access/transport. |
| `tensor-parallel` | Touches tensor-parallel dispatch. Orthogonal to `split-tensor` below -- `LLAMA_SPLIT_MODE_ROW` can also use tensor parallelism per llama.cpp's own enum comment, so these are not the same fact. |
| `meta-backend` | Touches the Meta/virtual backend path. |
| `graph-fusion` | Touches CUDA-graph op fusion. |
| `gated-delta-net` | Touches the Gated DeltaNet recurrent path. |
| `mtp` | Touches Multi-Token-Prediction speculative decoding specifically (a specific mechanism within `speculative-decoding`). |
| `state-snapshots` | Touches recurrent/MTP state capture or restore. |
| `wmma` | Touches WMMA matrix-core kernel paths. |
| `prefill` | Touches the prompt-processing (prefill) phase specifically. |
| `speculative-decoding` | Touches speculative decoding generally (the family `mtp` is one mechanism within). |
| `top-k` | Touches top-k selection/reduction. |
| `moe-routing` | Touches MoE expert routing. |
| `wave32` | Touches wavefront-width-specific (32-lane) execution behavior. |
| `flash-attention` | Touches flash-attention kernels. |
| `mmvq` / `mmq` / `mmvf` | Touches the named matrix-vector-quantized / matrix-matrix-quantized / matrix-vector-float kernel path specifically. |
| `quantization` | Touches quantization format/conversion logic. |
| `kv-cache` | Touches the KV cache mechanism. |
| `dispatch` | Touches BigCherry's own dispatch-selection plumbing. |
| `split-none` / `split-layer` / `split-row` / `split-tensor` | The patch's behavior genuinely differs by `-sm` split mode (llama.cpp's real `enum llama_split_mode`, `include/llama.h`). Only tag the mode(s) that actually matter to this patch, not the one(s) it happens to have been tested under. |

Do not add a tag that only restates another field already present on the
patch: `dual-gpu`/`peer-access`-shaped assertions belong as the relevant
subsystem tag (`allreduce`/`tensor-parallel`/`p2p`) since multi-GPU-ness is
already implied by touching one of those; a tag that only restates `kind`
(e.g. a `fix` tag on an `upstream-backport` patch) does not belong here
either.

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
it and `patch-doc` merges it into release documentation. Its first two
metadata lines must be contiguous and exactly match `patch.toml`:

```markdown
# 1204_example_patch

**Status:** untested
**Plan item:** RD08

## What it does

Short description of the source change.

## Why

The engineering reason, without claiming unmeasured benefit.

## Upstream

Origin commit or local ownership and any adaptation note.
```

The summary is not evidence and must not claim a benchmark result that the
bound contract/evidence record does not support. After changing `state`
or plan bindings, update this header in the same change and rerun
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
