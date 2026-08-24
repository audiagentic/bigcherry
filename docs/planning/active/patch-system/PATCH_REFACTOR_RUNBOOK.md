# BigCherry Packaged Patch + Validation Refactor

## Locked implementation specification and staged agent runbook

## 0. Purpose

The refactor has four goals:

1. Make a BigCherry patch capable of owning its implementation, validation adapter, supporting validation code, notes and documentation without turning every support Python file into another patch.
2. Give every new patch a clearly documented path from implementation through isolated source, build, correctness, activation, benchmarking, evidence and explicit lifecycle promotion.
3. Replace scattered patch-specific validation knowledge with reusable built-in validators plus narrowly scoped package-local validation adapters.
4. Make the canonical development gate run locally with **no new third-party dependencies**, through `python -m bigcherry check`.

This is not a rewrite of BigCherry's campaign, source identity, benchmarking or evidence systems.

The core principle is:

```text
Reuse mechanisms.
Centralize invariants.
Move patch-specific knowledge next to the patch.
Fail closed.
```

---

## 0.1 Review record (2026-08-23)

> See §0.2 for the 2026-08-25 identity amendments (RV80).

Reviewed against the live `tuning-code-rebase` codebase: every factual premise
verified (recursive `discover_modules` with `_`-filter, `_patch_path` /
`patch_implementation_digest` / `importlib.import_module` in source isolation,
byte-compile `_load_module`, `_TRACE_PROBE_SPECS` holding exactly 1205_rd12 +
1206_rd13, fully flat `patches/`, and `check`/`patch-lint`/`patch-validate` being
genuinely new commands). External GPT clarification (gateway
`req_cf1dc98ecf4b4ef0`) confirmed the architecture and mandated 8 must-fix
corrections + 6 pins, all applied in this revision (list continued in §0.3):

## 0.2 Amendment record (2026-08-25, RV80)

GPT adversarial identity review of the implemented RS01–RS06 (plan review RV80)
found six spec-compliance blockers in the PA02 code. This runbook was already
normative on all six (§15 invalidation table: overlay → new source, state-only
→ no new source, dependency composition → new source; §58: "Replace implicit
state-scan baseline with explicitly resolved composition"); the implementation
had drifted from the spec. Amendments:

- A1 (amends §11 DAG pin): `patch_registry` MAY import `experiment_contract`
  (one-way; `experiment_contract` must never import `patch_registry`) so the
  linked Experiment Contract hash is the canonical `ExperimentContract.
  contract_hash` (EC01), not an ad-hoc SHA-256 of a raw TOML table.
- A2 (confirms §14.2): `VALIDATION_FRAMEWORK_VERSION` lives in `patch_registry`
  (lower layer) and is re-exported from `patch_validation`.
- A3 (confirms §9): descriptor paths (`implementation_path`, `validation_path`,
  `metadata_path`) are REGISTRY-ROOT-relative.
- §12 addition: the implementation loader MUST re-hash the bytes it is about
  to execute against `descriptor.implementation_digest` and fail closed on
  mismatch (descriptor and executed bytes must be the same content).
- §12 addition: packaged `patch.py` imports are statically validated BEFORE
  execution (AST walk): Python stdlib + `bigcherry.patcher` only; relative
  imports rejected. (Flat-patch import scope is a v1 open question — the live
  flat tree is 100% conformant; ruled separately.)
- §14.3 addition: materialised source identity is normative payload
  `bigcherry-patch-source-v2` (see §14.3). State scans are forbidden in the
  identity path; the requested base ref is recorded informationally, the
  RESOLVED commit SHA is the semantic one.
- §11/§59 addition: composition ORDER is a true topological order —
  `patchset.topological_order()`; the canonical `(order, patch_id)` key is a
  tie-breaker among ready nodes ONLY, never a global re-sort (a global sort
  destroys dependency order when numbering and REQUIRES disagree).
- §59 (RS06) addition: both the control AND the subject composition must be
  run through the authoritative exact-composition validator
  (`patchset.resolve_exact`): unknown IDs, missing requires, forward AND
  reverse conflicts, and rejected members all BLOCK the comparison.

### Required PA02 acceptance tests (RV80; all landed with the fix commits)

1. `0100_child` REQUIRES `0200_parent` → resolve/expand order is parent-first (temp registry).
2. Overlay file change → different source key (temp overlay).
3. Patch implementation change → different implementation digest → different key.
4. Lifecycle state flips (packaged: `patch.toml` state; flat: unrelated patch promoted) → same key — no state in identity.
5. Base ref name vs its SHA → same resolved identity; ref moves → new identity.
6. `materialize_composition` applies EXACTLY the explicitly given composition (manifest composition list), even when other validated patches exist in the registry.
7. Focal in baseline → BLOCKED; focal REQUIRES target rejected → BLOCKED; reverse conflict → BLOCKED.
8. Patch bytes differ from descriptor digest → loader raises.
9. Packaged `patch.py` imports a non-stdlib, non-`bigcherry.patcher` module → loader raises.
10. `validation.toml` symlink escaping the package root → not accepted as that patch's manifest (raises at discovery).

## 0.3 Original 8 must-fixes (applied in the 0.1 revision)
- B1: removed the undefined `fallback` validator from the v1 built-in set (fail-closed).
- B2: locked the custom-callable API (§31).
- B3: `not_applicable` never satisfies a required check — v1 lock (§19).
- B4: pinned `VALIDATION_FRAMEWORK_VERSION` (§14.2).
- M1: staged commit boundaries per RS; dropped "one-shot" (§52.1).
- M2: plan items + ledger events before/around implementation (§52.2).
- M3: flaky-test policy — no retry-to-green (§44).
- M4: non-mutating audit gate in the DEFAULT tier (§44).
- N1–N6: provenance, path safety, repo-relative paths, idempotency, evidence
  schema v2, and the module dependency DAG (§4, §9, §11, §22, §38, §64).

Disposition: all eight B/M items MUST-FIX before implementation; none deferred.

---

# 1. Locked terminology

Do **not** introduce a new first-class concept named `Change`.

BigCherry already uses “change event” for the authoritative release ledger, so adding a lifecycle entity also named Change would create needless ambiguity.

The canonical term remains:

```text
Patch
```

There are two representations:

```text
simple patch
packaged patch
```

The low-level implementation primitives remain:

```text
FilePatch
Edit
```

Experiment remains:

```text
Experiment Contract
```

Validation outputs remain:

```text
evidence
```

This means the vocabulary becomes:

```text
Patch               source transformation + lifecycle identity
PatchDescriptor     normalized repository representation
Experiment Contract scientific claim / evaluation requirements
Validation adapter  how this patch satisfies those requirements
Campaign            execution
Evidence            immutable proof of what ran
Patch state         explicit lifecycle decision
Ledger change event repository/release history
```

---

# 2. Non-negotiable architecture

The target data flow is:

```text
                      PATCH REGISTRY
                           │
             ┌─────────────┴─────────────┐
             │                           │
      simple flat patch            packaged patch
       patches/X.py            patches/.../X/patch.toml
                                          │
                                       patch.py
                                          │
                                validation.toml
                                     optional
                                          │
                               validation/*.py
                                     optional
             │                           │
             └─────────────┬─────────────┘
                           │
                    PatchDescriptor
                           │
                 Experiment Contract
                    when applicable
                           │
                    ValidationPlan
                           │
                  exact source plans
                    ┌──────┴──────┐
                    │             │
                 CONTROL       SUBJECT
                    │             │
                    └──────┬──────┘
                           │
                 existing build/campaign
                           │
                  built-in validators
                     + custom adapter
                           │
                     CheckResults
                           │
                    evidence record
                           │
               eligibility determination
                           │
                explicit state promotion
```

Nothing about that architecture may:

- create another benchmark engine;
- create another source identity system;
- create another build identity system;
- create another Experiment Contract schema;
- infer patch paths from patch IDs outside the registry;
- allow custom validation code to bypass evidence requirements;
- automatically edit `STATE` after a successful run.

---

# 3. Filesystem layout

## 3.1 Simple patch

Existing form stays valid:

```text
patches/
    0100_cmake_options.py
    1002_hip_unsafe_math_opt_in.py
    ...
```

A simple patch stays one file.

That must remain a first-class representation indefinitely.

## 3.2 Packaged patch

Target:

```text
patches/
    rd/
        1204_rd08_q6k_mmvq_vdr2/
            patch.toml
            patch.py

            validation.toml

            validation/
                checks.py
                corpus.py
                reference.py
                fixtures/
                    ...

            README.md

            notes/
                investigation.md
                provenance.md
```

Only these are mandatory:

```text
patch.toml
patch.py
```

Everything else is optional.

## 3.3 Reserved directories

Any relative path component beginning `_` is reserved for non-discoverable repository support:

```text
patches/_template/
patches/_shared/
```

The initial refactor should add:

```text
patches/_template/
    patch.toml
    patch.py
    validation.toml
    validation/
        checks.py
    README.md
```

The template is documentation, not a registered patch.

---

# 4. Discovery rules

Current `patchset.discover_modules()` recursively discovers every non-private `.py` under `patches/`. That would accidentally turn `validation/checks.py`, `corpus.py`, etc. into patch modules.

Replace discovery semantics with exactly:

```text
Simple patches:
    patches/*.py
    root level only
    filename must not start "_"

Packaged patches:
    patches/**/patch.toml
    any depth
    no relative path component may start "_"

Never discover:
    arbitrary nested *.py
```

Examples:

```text
patches/1002_foo.py
    DISCOVER simple patch

patches/rd/1204_x/patch.toml
    DISCOVER packaged patch

patches/rd/1204_x/patch.py
    NOT independently discovered

patches/rd/1204_x/validation/checks.py
    NOT discovered

patches/_template/patch.toml
    NOT discovered
```

Path safety (normative): resolve candidate paths/symlinks and reject anything
whose resolved path escapes `patches/` — or, for package-local files, the package
root. (See the §73 adversarial sheet for the concrete cases.)

---

# 5. Duplicate identity rules

Patch ID is repository-global.

This must fail:

```text
patches/1204_rd08_q6k_mmvq_vdr2.py

patches/rd/1204_rd08_q6k_mmvq_vdr2/patch.toml
```

even though the physical representations differ.

Similarly, these must fail:

```text
patches/rd/foo/patch.toml
    id = "1204_x"

patches/core/bar/patch.toml
    id = "1204_x"
```

Migration therefore means:

```text
same canonical patch ID
different physical representation
```

not creation of a second patch.

---

# 6. Package path rules

For a packaged patch:

```text
directory basename == patch ID
```

should be mandatory.

Therefore:

```text
patches/rd/1204_rd08_q6k_mmvq_vdr2/patch.toml
```

must declare:

```toml
id = "1204_rd08_q6k_mmvq_vdr2"
```

This should fail:

```text
directory: 1204_rd08_q6k_mmvq_vdr2
id:        1204_rd08
```

The parent grouping directories such as `rd/`, `core/`, `vulkan/`, etc. are organizational only and **never become semantic identity**.

Moving:

```text
patches/rd/X/
```

to:

```text
patches/experimental/X/
```

must not change X's patch ID or production implementation digest.

---

# 7. patch.toml

`patch.toml` is the canonical patch metadata authority for a packaged patch.

Recommended v1:

```toml
schema = 1

id = "1204_rd08_q6k_mmvq_vdr2"
order = 1204

group = "experimental-rdna-boosts"
state = "untested"

kind = "enhancement"
origin = "external-fork"
backend = "hip"

external-source = "stew675-rdna-boosts"
plan-ids = ["RD08"]

requires = ["1100_hi70_direct_op_evidence"]
conflicts = []

requires-options = []
forbids-options = []

subsystems = ["mmvq"]
hardware = ["rdna3", "rdna4"]

validation-architectures = [
    "gfx1100",
    "gfx1201",
]

experiment-contract = "RD08-Q6K-MMVQ-VDR2"
```

The fields should reuse existing vocabulary wherever possible.

---

# 8. Metadata authority rules

## Legacy simple patch

Current authorities remain:

```text
patch module constants
+
patches/catalog.toml
```

No forced migration.

## Packaged patch

Authority becomes:

```text
patch.toml
```

Do not duplicate packaged metadata in:

```text
patch.py
patches/catalog.toml
```

Specifically, `patch.py` should not need:

```python
STATE =
GROUP =
REQUIRES =
CONFLICTS =
```

for packaged patches.

The registry reads those values from `patch.toml`.

This gives us one source of truth after a patch is packaged.

---

# 9. PatchDescriptor

Create:

```text
tools/bigcherry/patch_registry.py
```

with a normalized immutable representation approximately:

```python
@dataclass(frozen=True)
class PatchDescriptor:
    patch_id: str
    order: int

    representation: str
    implementation_path: Path
    package_root: Path | None
    metadata_path: Path | None

    group: str
    state: str

    kind: str | None
    origin: str | None
    backend: str | None

    upstream: str | None
    external_source: str | None
    plan_ids: tuple[str, ...]

    requires: tuple[str, ...]
    conflicts: tuple[str, ...]

    requires_options: tuple[str, ...]
    forbids_options: tuple[str, ...]

    subsystems: tuple[str, ...]
    hardware: tuple[str, ...]
    validation_architectures: tuple[str, ...]

    experiment_contract: str | None

    implementation_digest: str

    validation_path: Path | None
    validation_digest: str | None
```

Field pins:
- `upstream` is explicit optional exact-upstream-commit provenance (equivalent to
  legacy `UPSTREAM`). Never derive it from `origin`; external-fork, local, or
  PR-only patches normally have `upstream = None`.
- Path fields (`implementation_path`, `package_root`, `metadata_path`,
  `validation_path`) are canonical repo-relative paths; convert to absolute only
  at I/O boundaries. Evidence must never depend on checkout location.

Do not call this `ChangeDescriptor`.

---

# 10. Patch registry responsibilities

`patch_registry.py` owns:

```text
discovery
metadata parsing
schema validation
duplicate detection
canonical ID
canonical order
implementation path
package root
package validation path
implementation loading
implementation digest
validation digest
dependency/conflict resolution inputs
```

Everything else uses it.

The critical API principle:

```text
patch ID -> resolve once -> PatchDescriptor
```

No downstream code decides whether a patch is flat or packaged.

---

# 11. patchset.py migration strategy

Do not rewrite every caller in one step.

Instead:

```text
patch_registry.py
        ↑
patchset.py compatibility façade
```

Existing APIs such as:

```text
catalog()
resolve_exact()
expand_composition()
load_patches()
```

should continue working.

Internally, migrate them toward the new registry.

Preserve deterministic ordering, canonical content hashes, duplicate rejection and fail-closed state handling.

Module dependency DAG (normative, no import cycles):
    patch_registry -> {paths, patcher, experiment_contract}
    patchset       -> patch_registry
    patch_source_isolation / patch_validation -> registry APIs
    check          -> higher-level / public validation APIs
`patch_registry` must never import `patchset`; no lower layer may import `check`.
A1 (RV80): `experiment_contract` is a leaf config module (imports stdlib +
`autotune_schema` only) and must never import `patch_registry`; the registry
imports it ONE-WAY for the canonical linked-contract hash (EC01).
Add an import-cycle / architecture test on this boundary.

---

# 12. Implementation loader

Current `_load_module()` deliberately compiles source bytes directly instead of relying on normal import machinery because Python bytecode caching previously allowed source bytes and executed patch implementation to diverge.

For both representations:

```text
read current implementation bytes
compile directly
execute in synthetic module
extract PATCH / PATCHES
```

No normal `importlib.import_module()` loading of patch implementation.

RV80 addition: after reading the bytes, the loader MUST re-hash them and
compare against `descriptor.implementation_digest`, failing closed on mismatch
(the descriptor and the executed bytes must be the same content — a file
edited between registry load and execution is a tree error, not a silent
re-hash).

## v1 package restriction

Keep packaged `patch.py` **self-contained** except for imports from:

```text
Python stdlib
bigcherry public patch APIs
```

Do not initially allow production implementation helpers such as:

```text
implementation/helpers.py
```

That would immediately complicate implementation identity.

If a future patch genuinely requires production helper modules, extend the implementation identity deliberately later.

Validation helpers are allowed because they belong to a different identity domain.

---

# 13. Production implementation contract

A packaged `patch.py` exposes exactly what simple patches already expose:

```python
PATCH = FilePatch(...)
```

or:

```python
PATCHES = [...]
```

No new edit DSL.

No change to `FilePatch`.

No change to anchored edit behaviour.

No change to expected-match-count semantics.

---

# 14. Identity model

This needs to be treated as normative.

There are three independent identity domains.

## 14.1 Production implementation identity

For v1:

```text
implementation_digest = sha256(patch.py bytes)
```

For a legacy patch:

```text
implementation_digest = sha256(patches/<id>.py bytes)
```

Do not hash README, notes or validator code.

## 14.2 Validation identity

For packaged patches:

```text
validation_digest =
    hash(
        validation.toml
        + validation/** deterministic path+bytes
        + linked Experiment Contract hash
        + validation framework semantic version
    )
```

The hash must use sorted relative paths plus bytes, not filesystem traversal order.

The "validation framework semantic version" is pinned as a constant in
`patch_registry.py` (the lower identity layer) and RE-EXPORTED from
`patch_validation.py` for validator/consumer convenience (A2 — the constant
must live in the lower layer so both identity domains read the same value):

    VALIDATION_FRAMEWORK_VERSION = "1"

Include the exact value in the hash. Bump it on any semantic change to
requirement aggregation, validator semantics, or result interpretation. Bumping
invalidates validation evidence while leaving production source/build identity
reusable (per §15).

## 14.3 Materialised source identity

Continue to use actual content-addressed source/tree identity.

Do not derive source identity merely from patch ID.

RV80 normative payload (`bigcherry-patch-source-v2`; variants use
`bigcherry-patch-source-variant-v2` adding `variant_name` + `variant_digest`):

```text
source_key = sha256({
  schema,
  resolved_revision,          # base ref resolved to an IMMUTABLE commit SHA
                              # (branch/tag/HEAD requested refs are recorded
                              # in the manifest informationally only)
  overlay_digest,             # sha256 over sorted (relpath, sha256) of every
                              # file under the source overlay (src/ additions)
  composition,                # ORDERED [(patch_id, implementation_digest)]
                              # in the exact application order returned by
                              # patchset.resolve_exact() — NEVER re-sorted:
                              # a lexicographic sort could give two DIFFERENT
                              # application orders the same key when a
                              # packaged patch.toml changes requires/order
                              # while patch.py digests stay fixed —
                              # stock = empty list
})
```

State scans are FORBIDDEN in the identity path: the composition is supplied
explicitly by the caller (campaign layer resolves the named
`[patch-set.*]` sets for the source under test); promoting, rejecting, or
adding ANY patch in the registry that is not in the composition does not
change the key. `git worktree add --detach` receives the resolved SHA, so a
moved ref yields a new identity, never a reused stale worktree.

---

# 15. Invalidation contract

This table must appear in both implementation docs and tests.

| Changed input | New source? | New production build? | New validation? |
| --- | ---: | ---: | ---: |
| `patch.py` | yes | yes | yes |
| legacy patch `.py` | yes | yes | yes |
| pinned llama.cpp commit | yes | yes | yes |
| dependency composition | yes | yes | yes |
| source overlay | yes | yes | yes |
| `validation.toml` | no | no | yes |
| `validation/checks.py` | no | no | yes |
| validation fixture/reference | no | no | yes |
| linked Experiment Contract semantic content | no | no | yes |
| README | no | no | no |
| notes | no | no | no |
| package parent folder move | no | no | no |
| `state = untested → validated` | no | no | no |

A state-only transition must not invalidate the evidence that justified it.

---

# 16. Experiment Contract remains the scientific authority

Do not add `bugfix`, `feature`, and `performance` validation profiles that duplicate Experiment Contract information.

The Experiment Contract owns:

```text
source
hypothesis
expected effect
scope
architectures
positive evaluation
controls
boundaries
correctness
acceptance thresholds
```

Therefore:

```text
patch.toml
    what is the patch?

Experiment Contract
    what must be demonstrated?

validation.toml
    which implementation produces each piece of evidence?
```

That ownership must remain strict.

---

# 17. validation.toml

`validation.toml` is an **execution adapter**, not another hypothesis language.

Example RD08:

```toml
schema = 1

[[check]]
id = "direct-op-correctness"
capability = "correctness"
validator = "backend-ops"
required = true
ops = ["MUL_MAT"]

[[check]]
id = "activation"
capability = "activation"
validator = "custom"
required = true
callable = "validation/checks.py:activation_check"

[[check]]
id = "runtime-smoke"
capability = "smoke"
validator = "runtime-smoke"
required = true

[[check]]
id = "performance"
capability = "performance"
validator = "benchmark"
required = true
```

RD12:

```toml
schema = 1

[[check]]
id = "fusion-activation"
capability = "activation"
validator = "trace-marker"
required = true
marker-regex = "BIGCHERRY_PATCH_HIT patch=1205_rd12 path=dual_output_mmvq_fusion"

[check.negative-control.environment]
GGML_CUDA_DISABLE_FUSION = "1"
```

Do not place performance acceptance percentages in this file.

Those belong to the Experiment Contract.

---

# 18. Validation requirement resolution

Create:

```text
tools/bigcherry/patch_validation.py
```

The validation plan is built from three sources:

```text
universal BigCherry requirements
+
linked Experiment Contract requirements
+
patch validation adapter
```

The adapter may add requirements.

It may **not remove** requirements imposed by the framework or Experiment Contract.

For example:

```text
Experiment Contract expected_effect = performance
```

implies that a performance result is required.

A contract with declared correctness checks implies correctness is required.

A contract containing controls/boundaries requires evidence covering those controls/boundaries.

A runtime performance claim should also require causal attribution/activation.

If the adapter provides no validator capable of satisfying a required capability:

```text
CONFIGURATION ERROR
```

not skip.

---

# 19. Validation statuses

Every check returns exactly one of:

```text
pass
fail
blocked
error
not_applicable
```

Meaning:

`pass`: Check ran and requirement was proven.

`fail`: Check ran and disproved requirement.

`blocked`: Required external prerequisite unavailable, such as required architecture or model.

`error`: Validation infrastructure or adapter malfunction.

`not_applicable`: v1 lock — NEVER satisfies a required check; valid only for non-required/advisory checks. Conditional applicability, if ever needed, belongs in the Experiment Contract schema as scientific policy, not the validation adapter.

A required check is satisfied only by:

```text
pass
```

`blocked` is never success.

`error` is never success.

---

# 20. Check result structure

Use an immutable structured result:

```python
@dataclass(frozen=True)
class ValidationResult:
    check_id: str
    capability: str
    status: str
    summary: str
    details: tuple[str, ...]
    artifacts: tuple[ArtifactRef, ...]
```

Do not let custom code directly set:

```text
eligible_for_validated_state = true
```

Only the central aggregator computes that.

---

# 21. Built-in validator registry

Initial built-ins should be closed and documented.

Recommended:

```text
apply
build
backend-ops
trace-marker
compile-option
runtime-smoke
autotune-campaign
benchmark
architecture
custom
```

Do not introduce a general plugin framework.

Note: there is deliberately NO undefined catch-all `fallback` validator in the v1
set. A future concrete need adds a specifically defined validator with one
specifically named capability — never a validator that satisfies arbitrary
missing capabilities (that would undercut §18 fail-closed requirement resolution).

---

# 22. `apply` validator

Proves:

```text
patch descriptor resolves
dependencies resolve
conflicts clear
patch transformation loads
patch applies
all expected anchors/counts match
source tree identity is known
second application is a verified no-op with an explicit already-applied result
    (guard-detected), no second mutation; shape-gated genuinely non-applicable
    edits may remain not-applicable
```

Use existing `patcher.py` rather than recreate edit logic.

---

# 23. `build` validator

Proves:

```text
correct source tree built
requested architecture(s) configured
requested CMake options present
compile-command identity verified
runtime bundle identity verified
```

Preserve existing completed-build evidence and configure-request identity behaviour.

---

# 24. `backend-ops` validator

Generalize the useful permanent portion of the current RD correctness workflow.

Typical operation:

```text
test-backend-ops test -o MUL_MAT
```

with at least:

```text
native mode
tune mode with correctness screening
```

The check must capture:

```text
command
exit code
target architecture
mode
candidate status
correctness rejects
log artifact
```

Patch-specific mapping such as:

```text
RD08 -> MUL_MAT
```

must live in `validation.toml`, not generic Python.

---

# 25. `trace-marker` validator

Extract the generic mechanics currently embedded in `_TRACE_PROBE_SPECS`.

Generic validator behaviour:

```text
run positive workload
search exact configured marker

run negative control
search same marker
```

Classification:

```text
positive=yes, negative=no  -> PASS
positive=no                -> FAIL / not activated
negative=yes               -> FAIL / probe not selective
process unavailable        -> BLOCKED
process error              -> ERROR
```

---

# 26. `compile-option` validator

Used for compile/configuration changes.

Proves a concrete flag or configuration exists in the actual verified compile/build inputs.

Can optionally check a control where the flag must be absent.

This is preferable to pretending every compile-only patch needs runtime marker instrumentation.

---

# 27. `runtime-smoke` validator

Reuse current runtime smoke machinery.

Do not build another llama invocation wrapper.

---

# 28. `autotune-campaign` validator

Wrap existing:

```text
record
inventory
tune
promote
correctness gate
export
replay
bench/report
```

as required.

The current patch campaign already drives the mature e2e smoke campaign rather than inventing a second tuning flow.

---

# 29. `benchmark` validator

Generalize existing A/B benchmarking rather than build another statistics engine.

Preserve:

- alternating arm ordering;
- reproducible balanced scheduling;
- log preservation;
- build-parity checks;
- protection against thermal/clock drift becoming a false performance signal.

Experiment Contract remains the source of acceptance thresholds.

---

# 30. `architecture` validator

Separate:

```text
compile coverage
```

from:

```text
runtime coverage
```

A patch may compile for:

```text
gfx1100
gfx1201
```

while the current host can only runtime-test one.

Required unavailable runtime hardware produces:

```text
BLOCKED
```

not PASS.

Evidence can accumulate across machines.

---

# 31. Custom validator

Custom validator declaration:

```toml
validator = "custom"
callable = "validation/checks.py:check_vdr_reference"
```

No pluggy.

No package entry points.

No environment-wide plugin discovery.

No pip-installed validator.

Loader rules:

```text
path must be inside package root
file must exist
callable must exist
callable must match the locked API (below)
result must be a ValidationResult
exceptions become ERROR
```

Locked custom-callable API (v1):

    def check(ctx: ValidationContext) -> ValidationResult

- Exactly one positional argument (`ctx: ValidationContext`).
- No additional required/optional parameters, no `*args`, no `**kwargs`.
- Synchronous only in v1.
- The framework verifies the returned `ValidationResult.check_id` and
  `.capability` match the declaring `[[check]]` — custom code cannot claim a
  different requirement.

---

# 32. ValidationContext

Custom checks receive a structured context, not raw global state.

Include:

```text
PatchDescriptor

base revision

control source
subject source
optional stock source

control source-tree identity
subject source-tree identity

build identities by role

architecture
device identity

model/workload info

Experiment Contract + hash

run directory

helpers:
    run binary
    register artifact
    create validation-only source variant
```

Do not expose a helper allowing mutation of the canonical control/subject source in place.

---

# 33. Validation-only source transformations

If a validator needs temporary instrumentation or corpus expansion:

```text
DO NOT mutate subject source.
```

Use a content-identified validation source variant.

Preserve content-addressed isolated worktrees and fail-closed manifest/tree verification.

---

# 34. Refactor `patch_source_isolation.py`

This is mandatory.

Current problems:

```text
_patch_path() accepts only flat patches/<id>.py
patch_implementation_digest() hashes that file
_apply_baseline_and_stack() imports patches.<module>
framework baseline inferred from every STATE=validated patch
```

Target:

```text
resolve PatchDescriptor
use descriptor.implementation_digest
load implementation through registry
apply resolved patches through registry/patchset
```

Delete `_patch_path()`.

Delete `importlib.import_module(f"patches.{patch_module}")`.

Preserve:

```text
content-addressed worktrees
manifest verification
HEAD verification
git working-tree hash verification
tamper refusal
atomic manifest writing
moving base revision fail-closed behaviour
```

---

# 35. Baseline composition

Do not continue implicitly defining:

```text
baseline = every patch whose STATE == validated
```

as the scientific baseline for focal-patch attribution.

Instead use explicit resolved source compositions from the canonical campaign/source configuration.

For focal patch X:

```text
BASELINE
    named BigCherry source composition

CONTROL
    baseline + X prerequisites, without X

SUBJECT
    baseline + same prerequisites + X
```

---

# 36. Subject/control/stock semantics

Authoritative attribution:

```text
SUBJECT vs CONTROL
```

Contextual comparison:

```text
SUBJECT vs STOCK
```

Stock is pristine pinned llama.cpp.

Do not attribute X's performance to:

```text
SUBJECT vs STOCK
```

if SUBJECT also contains many unrelated BigCherry patches.

---

# 37. Dependency handling in control construction

If:

```text
X requires A and B
```

then:

```text
CONTROL = baseline + A + B
SUBJECT = baseline + A + B + X
```

Use normal dependency closure.

If an existing baseline patch Y requires X, removing X makes the proposed control invalid.

Default result:

```text
BLOCKED: focal patch not independently isolatable
```

Do not silently remove Y too.

---

# 38. Evidence evolution

Do not create a completely separate “Change evidence” system.

Evolve current patch validation evidence to schema v2 (the live writer is
SCHEMA_VERSION = 1 / CONTRACT_VERSION = "hi83-v1").

New record adds:

```text
patch representation
production implementation digest
validation digest
Experiment Contract ID/hash

baseline composition
control composition
subject composition

control source tree
subject source tree
stock tree if relevant

individual ValidationResults
hardware identities
artifact hashes

eligibility result
blockers
```

Old evidence remains readable.

New campaigns write the new schema.

---

# 39. Patch state promotion

Validation never edits:

```text
state = "validated"
```

automatically.

Campaign result may say:

```text
eligible_for_validated_state = true
```

Promotion remains deliberate.

---

# 40. Artifact storage

Do not put builds under packaged patch directories.

Keep generated material under ignored runtime storage.

Recommended:

```text
artifacts/
    patch-validation/
        <patch-id>/
            <campaign-identity>/
                run.json
                verdict.json
                report.md

                checks/
                    apply.json
                    build.json
                    activation.json
                    correctness.json
                    benchmark.json

                logs/
                    ...

                campaign/
                    ...
```

Large:

```text
build trees
models
binaries
runtime logs
raw measurements
```

remain untracked.

Compact evidence stays tracked.

---

# 41. Path constants

Extend `tools/bigcherry/paths.py`.

Add constants conceptually like:

```text
PATCH_VALIDATION_EVIDENCE
PATCH_VALIDATION_ARTIFACTS
```

Do not scatter new repository-relative paths across modules.

---

# 42. Local CI is authoritative

Add:

```text
tools/bigcherry/check.py
```

and expose:

```text
python -m bigcherry check
```

This becomes BigCherry's canonical repository validation orchestrator.

GitHub Actions, if later added, simply invokes this command.

GitHub has no independent validation semantics.

---

# 43. No new third-party CI framework

Do not add:

```text
tox
nox
pre-commit framework
Pydantic
pluggy
jsonschema
marshmallow
```

for this refactor.

Use:

```text
argparse
dataclasses
subprocess
unittest
tomllib
json
hashlib
pathlib
time
```

---

# 44. `bigcherry check` tiers

## QUICK

```text
python -m bigcherry check --quick
```

Runs:

```text
patch registry/package lint
patch metadata validation
Experiment Contract loading/validation
dependency/conflict graph validation
validation.toml schema validation
custom validator import checks
fast identity invariant tests
source composition static checks
```

Target: seconds.

No build.

No GPU.

No model.

## DEFAULT

```text
python -m bigcherry check
```

Runs QUICK plus:

```text
full unittest suite
strict source-audit gate (non-mutating)
apply --dry-run
source identity tests
evidence schema tests
campaign planner tests
A/B benchmark unit tests
```

This is the authoritative normal local gate.

The audit gate must NOT subprocess the existing `audit` subcommand — that command
writes source-audit.json and updates release state. It must reuse the
`source_audit.audit()` / strict `passed()` semantics through a non-mutating check
runner (or an explicit no-write audit mode).

## FULL

```text
python -m bigcherry check --full
```

Runs DEFAULT plus slower hardware-free integration/reproducibility/cache tests.

It still does not unexpectedly launch large GPU model campaigns.

## Flaky test policy (locked)

No retry-to-green, no silently tolerated flakes. Fix the flaky test, or quarantine
it with `unittest.skip` carrying a tracked plan-item/reason — `check` reports it
as `skipped`. Pre-existing failures may be documented at RS00 while work begins,
but Definition of Done requires zero unquarantined DEFAULT failures.

---

# 45. Hardware remains explicit

Run:

```text
python -m bigcherry patch-validate <patch-id> ...
```

or, if nested CLI restructuring is later chosen:

```text
python -m bigcherry patch validate <patch-id> ...
```

Do not make:

```text
bigcherry check
```

implicitly compile ROCm and execute models.

Hardware validation is too expensive/environment-specific for that.

---

# 46. Local CI result model

Define:

```python
@dataclass(frozen=True)
class CheckSpec:
    check_id: str
    description: str
    tier: str
    runner: Callable
    timeout_seconds: float | None
```

and:

```python
@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: str
    duration_seconds: float
    summary: str
    output: str
```

Statuses:

```text
pass
fail
error
skipped
```

---

# 47. Local CI output

Example:

```text
BigCherry local check

PASS patch-registry              0.14s
PASS patch-metadata              0.08s
PASS experiment-contracts        0.11s
PASS validation-adapters         0.03s
PASS dependency-graph            0.05s
PASS identity-invariants         0.27s
PASS unit-tests                 42.18s
PASS audit                       1.61s
PASS apply-dry-run               3.22s

9 passed, 0 failed, 0 errors
RESULT: PASS
```

Default should run all requested checks and report all failures.

Add:

```text
--fail-fast
```

for debugging.

---

# 48. Machine-readable local CI output

Support:

```text
python -m bigcherry check --json artifacts/check.json
```

Schema should include:

```json
{
  "schema": "bigcherry-check-v1",
  "bigcherry_revision": "...",
  "python_version": "...",
  "tier": "default",
  "started_at": "...",
  "results": [],
  "status": "pass"
}
```

Do not write artifacts by default.

Explicit `--json PATH` avoids unnecessary filesystem mutation.

---

# 49. Local CI exit codes

Lock these:

```text
0 = all required checks passed
1 = one or more checks failed
2 = usage/configuration error
```

Do not invent complicated exit-code semantics.

Machine readers use JSON for details.

---

# 50. Optional Git hook

Later convenience:

```text
python -m bigcherry hooks install
```

can install a pre-push hook executing:

```text
python -m bigcherry check --quick
```

But this is optional and bypassable.

Never consider the Git hook authoritative.

---

# 51. CLI integration

Initial compatibility-friendly command set:

```text
bigcherry patches
bigcherry patch-explain <id>
bigcherry patch-graph
bigcherry patch-verify-evidence

bigcherry patch-lint [id]
bigcherry patch-validate <id>
bigcherry check
```

Do not rename every existing patch command during this refactor.

A future nested `patch ...` CLI can be cosmetic follow-up work.

---

# 52. Agent safety rules during implementation

The implementation agent must obey existing shared-working-tree doctrine.

Do not use:

```text
git stash
git reset
git rebase
```

Do not change release outputs directly instead of using the project's release/ledger workflow.

## 52.1 Commit boundaries (execution doctrine)

RS00–RS18 are staged, green commit boundaries. Each RS is independently tested and
committed before proceeding, or, where separation is impossible, an explicitly
documented tightly coupled RS pair. This is NOT a one-shot: do not collapse
intermediate steps into one large unreviewable diff. In this shared working tree,
never branch-switch the common checkout merely to satisfy this runbook; a
separately allocated worktree/branch is fine. This matches the repository's
no-stash / no-reset / no-rebase and deliberate-check-in-group doctrine.

## 52.2 Process integration (plan items + ledger)

Before implementation: create/assign the plan items and map the RS groups to them.
After each substantive implementation/check-in group, record an ag-ledger change
event with `plan-item-ids`; the associated plan item is completed only after that.
Plan grouping (see the `patch-system` plan group) — do not create one plan item
per RS:
- PA02 = RS00–RS06 (registry + source isolation)
- PA03 = RS07–RS11 (validation framework + evidence)
- PA04 = RS12–RS18 (local CI + docs + pilot migrations + acceptance)

---

# 53. Implementation run sheet RS00 — baseline freeze

Before modifying code:

Record:

```text
branch
HEAD SHA
Python version
current patch count
current patch IDs
current resolved recipe compositions
```

Run the current full unit suite from repository root:

```text
python -m unittest discover -s tools/tests
```

Also run current repository audit/dry-run commands appropriate to the checked-out environment.

Save baseline output outside tracked source.

### Exit criterion

No unexplained existing failures.

If there are pre-existing failures, document them before refactoring so they cannot be confused with regressions.

---

# 54. RS01 — add patch registry

Create:

```text
tools/bigcherry/patch_registry.py
tools/tests/test_patch_registry.py
```

Implement:

```text
PatchDescriptor
legacy discovery
package discovery
patch.toml parsing
schema validation
duplicate detection
deterministic ordering
implementation digest
validation path detection
validation digest
```

### Required tests

```text
root legacy patch discovered
nested patch.toml discovered
nested patch.py not separately discovered
nested validation/checks.py not discovered
_private legacy patch ignored
_template package ignored
duplicate ID rejected
directory/ID mismatch rejected
missing patch.py rejected
bad schema rejected
bad state rejected
bad dependency metadata rejected
deterministic ordering
```

### Exit criterion

New registry tests green.

No caller migrated yet.

---

# 55. RS02 — compatibility bridge in patchset

Modify:

```text
tools/bigcherry/patchset.py
```

Make existing catalog/resolution APIs consume normalized registry descriptors or a compatibility representation derived from them.

Do not change public behaviour yet.

### Regression suite

At minimum:

```text
test_patch_resolution.py
test_patch_selection.py
test_patch_catalog.py
test_recipes.py
test_campaign_resolution.py
test_source_plan_patch_contract_links.py
```

Then full unit suite.

### Exit criterion

All existing flat patches resolve exactly as before.

Order identical.

Composition identical.

Content IDs for legacy implementation remain equivalent.

---

# 56. RS03 — patch_catalog integration

Modify:

```text
tools/bigcherry/patch_catalog.py
```

Rules:

```text
legacy patch:
    metadata may come from existing catalog.toml

packaged patch:
    metadata comes from patch.toml
```

A packaged patch must not require duplicate `catalog.toml` metadata.

Update:

```text
patches command
patch-explain
patch-status
patch-graph
```

as required.

Remove/update historical comments stating `patches/` remains flat indefinitely.

### Exit criterion

Legacy catalog output unchanged.

Package metadata renders correctly.

Mixed catalog behaves deterministically.

---

# 57. RS04 — implementation loader cutover

Move all implementation loading behind:

```text
patch_registry.load_implementation(descriptor)
```

Use direct byte compilation.

Remove path guessing from callers.

### Search-based acceptance criterion

There should be no generic code constructing:

```text
PATCHES / f"{patch_id}.py"
```

outside migration/legacy-specific code inside the registry.

There should be no generic:

```text
importlib.import_module(f"patches.{patch_id}")
```

---

# 58. RS05 — source isolation cutover

Modify:

```text
patch_source_isolation.py
```

Delete flat-only `_patch_path()`.

Replace focal patch digest calculation with descriptor identity.

Replace `patches.<module>` imports with registry loading.

Replace implicit state-scan baseline with explicitly resolved composition.

Preserve:

```text
content-addressed worktrees
manifest verification
HEAD verification
git working-tree hash verification
tamper refusal
atomic manifest writing
moving base revision fail-closed behaviour
```

### Tests

```text
legacy materialization
package materialization
tampered tree rejected
missing manifest not trusted
wrong manifest identity rejected
base pin movement changes identity
legacy/package migration tree equivalence
```

---

# 59. RS06 — subject/control source plans

Implement explicit focal comparison construction.

Inputs:

```text
baseline source composition
focal patch
dependency closure
```

Outputs:

```text
control composition
subject composition
optional stock
```

Assert:

```text
subject composition - control composition == focal patch
```

after dependency closure normalization.

If not:

```text
BLOCKED
```

### Tests

```text
no dependency
single dependency
transitive dependency
baseline already contains dependency
dependency ordering deterministic
focal patch already in baseline
another baseline patch depends on focal
conflict introduced by focal
rejected dependency
```

---

# 60. RS07 — validation framework

Create:

```text
tools/bigcherry/patch_validation.py
tools/tests/test_patch_validation.py
```

Implement:

```text
CheckSpec
ValidationPlan
ValidationContext
ValidationResult
validation.toml parser
built-in validator registry
requirement aggregation
verdict computation
```

Do not execute hardware yet.

### Required negative tests

Unknown validator:

```text
ERROR
```

Missing required capability producer:

```text
configuration failure
```

Custom callable escaping package:

```text
configuration failure
```

Duplicate check ID:

```text
configuration failure
```

Required check `blocked`:

```text
not eligible
```

Required check `error`:

```text
not eligible
```

---

# 61. RS08 — Experiment Contract binding

Link packaged patch:

```text
patch.toml experiment-contract
```

to existing Experiment Contract registry.

Verify:

```text
contract exists
contract hash captured
scope/backend compatible with patch metadata
declared hardware/architecture not contradictory
correctness requirements get required producers
performance acceptance requires performance evidence
controls/boundaries appear in validation plan
```

Do not duplicate Experiment Contract fields in `validation.toml`.

### Exit criterion

Changing only an Experiment Contract changes validation identity but not implementation/source identity.

---

# 62. RS09 — built-in validators

Extract/generalize one at a time.

Recommended order:

```text
apply
build
trace-marker
backend-ops
compile-option
runtime-smoke
architecture
benchmark
autotune-campaign
custom
```

Each validator requires:

```text
unit tests
documentation
stable configuration schema
structured artifact output
clear pass/fail/blocked/error semantics
```

Do not migrate patch-specific use cases until the generic validator is independently tested.

---

# 63. RS10 — patch_validation_campaign refactor

Refactor current:

```text
patch_validation_campaign.py
```

into orchestrator over:

```text
PatchDescriptor
ValidationPlan
control/subject source plans
existing build/campaign machinery
```

Remove `_TRACE_PROBE_SPECS`.

Generic campaign code must contain zero:

```text
1205_rd12
1206_rd13
RD08 -> MUL_MAT
```

knowledge.

Keep current reliable configure/build identity work.

---

# 64. RS11 — evidence schema upgrade

Modify:

```text
patch_validation_evidence.py
```

Support:

```text
schema v1 read
schema v2 write
reader accepts v1 + v2
```

Do not delete legacy evidence.

New evidence records:

```text
implementation digest
validation digest
contract hash
control composition/tree
subject composition/tree
build identities
check results
artifact hashes
hardware
final eligibility
```

### Negative tests

Tamper:

```text
patch implementation
validation implementation
contract
subject tree
control tree
build identity
artifact
campaign identity
```

Each must invalidate the applicable evidence.

README edit must not.

---

# 65. RS12 — local CI implementation

Create:

```text
tools/bigcherry/check.py
tools/tests/test_check.py
```

Wire:

```text
bigcherry check --quick
bigcherry check
bigcherry check --full
bigcherry check --fail-fast
bigcherry check --json PATH
```

Tests must mock child commands where appropriate rather than recursively running the whole repository suite inside individual unit tests.

Test:

```text
successful aggregate
single failure
multiple failures reported
fail-fast
timeout
exception
JSON output
exit codes
tier selection
deterministic check ordering
```

---

# 66. RS13 — CLI integration

Modify:

```text
tools/bigcherry/__main__.py
```

Add:

```text
check
patch-lint
patch-validate
```

Do not break:

```text
patches
patch-status
patch-explain
patch-graph
patch-verify-evidence
```

---

# 67. RS14 — template

Add:

```text
patches/_template/
```

Template README should say:

```text
Delete validation.toml if built-in/default validation is sufficient.

Delete validation/ entirely unless custom code is necessary.

Do not create package-local production helpers in v1.

Do not put build output here.

Do not put large raw benchmark output here.
```

---

# 68. RS15 — pilot migration RD12

Migrate:

```text
1205_rd12_paired_mmvq_dual_output
```

from flat patch to package.

Move its exact implementation bytes into:

```text
patch.py
```

Move activation marker knowledge from generic campaign into:

```text
validation.toml
```

### Before deleting old patch

Apply old representation to same base.

Record resulting git working-tree hash.

Apply package representation to same base.

Require:

```text
OLD_TREE_HASH == NEW_TREE_HASH
```

If not equal:

```text
STOP
```

Do not rationalize differences during migration.

---

# 69. RS16 — pilot migration RD13

Repeat exact equivalence procedure for:

```text
1206_rd13_mul_mat_add_view_fusion
```

This proves a second trace-marker package.

After both migrate:

```text
_TRACE_PROBE_SPECS
```

should no longer exist.

---

# 70. RS17 — pilot migration RD08

Migrate:

```text
1204_rd08_q6k_mmvq_vdr2
```

Package owns:

```text
patch implementation
MUL_MAT validation mapping
custom VDR/direct-op logic if required
architecture requirements
documentation
```

The linked Experiment Contract owns:

```text
hypothesis
workloads
controls
boundaries
correctness requirements
performance acceptance
```

RD08 is the decisive complex test because it exercises both correctness and performance validation.

---

# 71. RS18 — retain a flat simple patch

Do **not** migrate everything.

Keep something such as:

```text
1002_hip_unsafe_math_opt_in.py
```

flat.

Run it through the new registry/campaign where appropriate.

This proves simple patches remain first-class and the architecture has not turned every one-line edit into a package bureaucracy.

---

# 72. Identity acceptance run sheet

Perform these exact mutations in a test fixture or disposable branch/worktree.

## A

Edit only:

```text
patch.py
```

Expected:

```text
implementation digest changes
subject source identity changes
build identity changes
validation identity/evidence changes
```

## B

Revert A. Edit only:

```text
validation.toml
```

Expected:

```text
implementation digest unchanged
source identity unchanged
existing production build reusable
validation digest changes
validation reruns
```

## C

Edit only:

```text
validation/checks.py
```

Same expectation as B.

## D

Edit only:

```text
README.md
```

Expected:

```text
implementation unchanged
validation unchanged
source unchanged
build unchanged
evidence remains current
```

## E

Change patch state only:

```text
untested -> validated
```

Expected:

```text
source/build identity unchanged
qualifying evidence remains applicable
```

## F

Change pinned llama.cpp revision.

Expected:

```text
source identity changes
build changes
evidence stale
```

---

# 73. Discovery adversarial run sheet

Create temporary repository trees testing:

```text
patches/foo.py
patches/_foo.py
patches/foo/bar.py
patches/rd/X/patch.toml
patches/rd/X/patch.py
patches/rd/X/validation/checks.py
patches/_template/patch.toml
patches/_template/patch.py
```

Expected discovered patches:

```text
foo
X
```

Nothing else.

Also test:

```text
malformed patch.toml
missing patch.py
duplicate ID
package directory mismatch
nested path escape
symlink escape if symlinks are supported
```

All fail closed.

---

# 74. Local CI acceptance run sheet

Run:

```text
python -m bigcherry check --quick
```

Require PASS.

Run twice.

Require deterministic set/order of checks.

Then:

```text
python -m bigcherry check
```

Require PASS.

Then:

```text
python -m bigcherry check --full
```

Require PASS on an environment satisfying its offline prerequisites.

Then deliberately break a package schema.

Require:

```text
check fails
error names exact package/file/problem
other checks continue by default
```

Then test:

```text
--fail-fast
```

Require stop at first failure.

Then:

```text
--json artifacts/check-test.json
```

Parse JSON independently and verify status/result count.

---

# 75. Hardware acceptance: simple patch

Run a flat patch through the new resolver/build path.

Goals:

```text
legacy representation works
isolated source works
build evidence works
no package assumptions leak
```

This is primarily compatibility proof.

---

# 76. Hardware acceptance: RD12/RD13

Run migrated package on correct hardware.

Require:

```text
subject source materialized
control materialized
builds verified

positive workload:
    trace marker present

negative fusion-disabled workload:
    trace marker absent

correctness requirement:
    passes

smoke:
    passes

evidence:
    current
```

Second identical run:

```text
reuse isolated source
reuse unchanged build
reuse completed stages where identities permit
```

---

# 77. Hardware acceptance: RD08

On declared primary hardware:

```text
build control
build subject

run native backend-op reference
run tune backend-op correctness

verify exact RD08 workload coverage
verify no correctness rejects

prove VDR path activation/attribution
run matched subject/control benchmark
run Experiment Contract positive workloads
run controls
run boundaries
```

If gfx1100 is declared non-regression coverage and gfx1201 primary qualification, keep those distinctions explicit.

Do not report unavailable required hardware as PASS.

---

# 78. Benchmark discipline

Subject/control executions must have:

```text
same model
same workload
same relevant runtime arguments
same architecture
same toolchain
same CMake parity
balanced/interleaved run ordering
```

Only intended variable:

```text
focal patch implementation
```

Use existing A/B scheduling mechanics rather than sequential:

```text
control x N
then subject x N
```

which is vulnerable to time/temperature drift.

---

# 79. Cross-machine qualification

Evidence should support cumulative qualification.

Example:

```text
Machine A / gfx1100
    correctness PASS
    non-regression PASS

Machine B / gfx1201
    correctness PASS
    performance PASS
```

Repository can then establish that declared architecture coverage is satisfied by multiple evidence records.

Do not require all architectures to exist in one machine.

---

# 80. Pin-bump acceptance

Before bump:

```text
pin A
patch X
subject/control/build/evidence A
```

After bump:

```text
pin B
same patch implementation
```

Require:

```text
new base revision
new subject/control source identities
new build identities
old validation evidence considered stale
new validation required
```

No manual cache deletion should be required.

---

# 81. Migration equivalence gate

For every migrated flat → packaged patch:

```text
same base commit
same overlay
same dependencies
old patch implementation
new patch implementation
```

Require exact:

```text
git_worktree_tree(old) == git_worktree_tree(new)
```

This is stronger than diff inspection.

Only after equivalence is proven should the old flat module be removed.

---

# 82. Full regression suite before sign-off

At minimum the refactor must leave green all tests around:

```text
patch resolution
patch catalog
patch selection
patch lifecycle
patch governance
recipes
campaign resolution
campaign source
campaign build
campaign lane
source identity
source materialization
patch validation campaign
patch validation evidence
correctness evidence
promotion correctness
A/B benchmarking
Experiment Contracts
experiment bundles
multi-GPU validation
```

The authoritative local runner should use stdlib unittest so a fresh Python environment does not require pytest merely to execute the canonical check.

---

# 83. Documentation deliverables

Do not consider this refactor complete without these four documents.

## `docs/reference/PATCH_SYSTEM.md`

Normative system architecture.

Must cover:

```text
terminology
simple vs packaged
PatchDescriptor
discovery
metadata authority
dependencies/conflicts
source composition
subject/control/stock
identity/invalidation
evidence
lifecycle
```

## `docs/reference/PATCH_AUTHORING.md`

The first document an agent reads before adding a patch.

Must include:

```text
when to stay flat
when to create package
template walkthrough
patch.toml schema
implementation rules
Experiment Contract linking
validation adapter rules
commands
definition of done
```

## `docs/reference/PATCH_VALIDATION.md`

Catalogue every built-in validator.

For each:

```text
purpose
config syntax
inputs
capability produced
artifacts
PASS semantics
FAIL semantics
BLOCKED semantics
ERROR semantics
examples
```

Also document:

```text
custom validator API
ValidationContext
identity rules
hardware evidence
```

## `docs/reference/PATCH_REFACTOR_RUNBOOK.md`

The implementation/migration document containing essentially RS00-RS18 and all acceptance sheets above.

State explicitly that it supersedes the older flat-only RE41 layout decision.

---

# 84. Agent authoring run sheet after refactor

A future agent implementing a new patch should execute exactly this flow.

### Step 1 — read

Read:

```text
PATCH_AUTHORING.md
PATCH_VALIDATION.md
relevant plan item
linked Experiment Contract if any
```

### Step 2 — choose representation

Use simple patch when:

```text
one implementation file is sufficient
no patch-specific validation code required
no package-local documentation required
```

Use package when:

```text
custom validation is required
complex provenance/notes useful
multiple validation fixtures required
patch-specific workload mapping exists
```

### Step 3 — implement

Create patch implementation.

Do not add custom validation until existing built-ins have been checked.

### Step 4 — lint immediately

Run:

```text
python -m bigcherry patch-lint <id>
```

Fix all structural issues first.

### Step 5 — local quick CI

Run:

```text
python -m bigcherry check --quick
```

### Step 6 — full offline CI

Run:

```text
python -m bigcherry check
```

### Step 7 — hardware validation

Run:

```text
python -m bigcherry patch-validate <id> ...
```

on appropriate hardware.

### Step 8 — inspect result

Require:

```text
all required checks pass
no required blocked checks
evidence identity current
Experiment Contract obligations satisfied
```

### Step 9 — rerun

Run identical validation again.

Confirm cache/build/source reuse.

### Step 10 — review

Review generated evidence/report.

Do not promote merely because command exited zero.

### Step 11 — explicit lifecycle promotion

Only after evidence review:

```text
untested -> validated
```

### Step 12 — final repository gate

Run:

```text
python -m bigcherry check
```

again after state change.

---

# 85. Rules future agents must not violate

Put these near the beginning of `PATCH_AUTHORING.md`:

```text
Never infer a patch's filesystem path from its ID.

Never add RD-specific knowledge to generic campaign code.

Never add a second benchmark implementation.

Never use stock as the sole causal control for a focal patch.

Never mutate an identity-bound source tree during validation.

Never let validator code declare itself validated.

Never silently skip a required validator.

Never convert unavailable required hardware into PASS.

Never make notes/docs part of production build identity.

Never put raw build trees in patch packages.

Never add a third-party dependency merely for configuration/schema/plugin convenience.

Never automatically promote patch lifecycle state.
```

---

# 86. Third-party tooling decision

Locked recommendation:

**Required third-party Python dependencies for this refactor: none.**

Do not add tox/nox/pluggy/Pydantic/etc.

Possible future optional test-only additions:

```text
Hypothesis
```

could strengthen property testing of graph/discovery/identity invariants later, but it is not required to ship this architecture.

Likewise Ruff/pyright could be valuable repository-wide initiatives later, but should not be coupled to this refactor.

---

# 87. GitHub decision

GitHub CI is optional secondary verification.

It is **not** the architecture.

If added later:

```text
checkout
Python
python -m bigcherry check
```

should be essentially the whole semantic workflow.

No GPU CI is required.

No GitHub self-hosted runner is required.

No external CI service is required.

The primary path is local.

---

# 88. Definition of done

The refactor is complete only when every statement below is true.

| Requirement | Required |
| --- | --- |
| Existing flat patches still work | yes |
| Packaged patches work | yes |
| Package discovery is `patch.toml` based | yes |
| Nested validator Python cannot become accidental patches | yes |
| One canonical PatchDescriptor exists | yes |
| Generic code does not reconstruct patch paths | yes |
| Packaged metadata has one authority | yes |
| Experiment Contract remains scientific authority | yes |
| Validation adapter does not duplicate experiment semantics | yes |
| Production/validation/doc identities are separate | yes |
| Source isolation accepts both representations | yes |
| Exact subject/control composition exists | yes |
| Non-isolatable control fails closed | yes |
| Built-in validators documented | yes |
| RD-specific mappings removed from generic campaign | yes |
| Evidence binds validation + source + build identities | yes |
| Existing evidence remains readable | yes |
| Validation cannot auto-promote state | yes |
| `bigcherry check` exists | yes |
| `bigcherry check` has no new Python dependencies | yes |
| QUICK/default/FULL tiers tested | yes |
| Legacy → package tree equivalence tested | yes |
| RD12/RD13 pilot validated | yes |
| RD08 complex pilot validated | yes |
| At least one flat patch validated through new infrastructure | yes |
| Validator-only change reuses production build | yes |
| Implementation change rebuilds | yes |
| Pin change rebuilds | yes |
| Documentation/template shipped | yes |
| Full offline suite green | yes |
| Required real-hardware acceptance green | yes |

## Final locked recommendation

The refactor should therefore add **only three genuinely new infrastructure concepts**:

```text
PatchDescriptor / patch registry
ValidationPlan / built-in validator adapter layer
bigcherry check local CI
```

Everything else should be adaptation of systems BigCherry already has.

That is the version to hand to an implementation agent as a staged major refactor: each RS lands as its own tested, committed, independently revertible increment (see §52.1), strict enough to prevent architectural drift, but it avoids turning BigCherry into a plugin framework, CI platform, second experiment system, or dependency-heavy Python application.
