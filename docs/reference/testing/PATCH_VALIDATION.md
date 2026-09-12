# Patch validation policy

This is the sole canonical policy for proving what a BigCherry patch does or
does not establish. It covers package requirements, Experiment Contract
binding, campaign capability, evidence identity, lifecycle decisions, and
promotion/demotion. Use [TEST.md](TEST.md) for commands and host procedures;
use [../patches/PATCH_AUTHORING.md](../patches/PATCH_AUTHORING.md) for creating
the package.

## Gate sequence at a glance

The gate sequence is operation-driven. A patch kind, tag, or origin can change
which authority is consulted inside a gate (especially G3 and G4), but does
not change the top-level `gate_applies()` matrix. The public `patch-gates`
intents are exactly `validate`, `promote`, `build`, and `rebase`; `AUTHOR` and
`LINT` are internal intents used by package authoring and repository lint.
Each result is `PASS`, `FAIL`, `BLOCKED`, or `NA`; an inapplicable gate is not
silently treated as a pass.

| Gate | Applies to | Requirement | Authoritative implementation |
| --- | --- | --- | --- |
| G0 Exact composition | author, validate, promote, rebase, build | Supplied ordered composition re-resolves exactly, including patch IDs and content hashes; rejected, conflicting, or unresolvable composition fails closed | `patch.gates.evaluate_composition_gate()` -> `patch.patchset.resolve_exact()` |
| G1 Documentation | author, lint, validate, promote, rebase, build | Focal `SUMMARY.md` exists and its Status/Plan item header agrees with the canonical descriptor; repository LINT applies the same summary authority per descriptor | focal: `patch.gates.evaluate_summary_gate()` -> `patch.docs.check_summary_for_patch()`; LINT: `patch.gates.evaluate_repository_lint_gates()` -> `patch.gates._evaluate_lint_summary()` |
| G2 Revision/apply | validate, promote, rebase, build | Rebase report is fresh against live source/BigCherry/overlay/selection/patch identities and the focal patch is known-good | `patch.gates.evaluate_rebase_gate()` -> `patch.rebase.require_fresh_report()` |
| G3 Validation definition | lint, validate, promote | Static package/ValidationPlan policy passes; focal operational paths reach execution-package and optimization checks according to the live evaluator; a not-required focal package returns NA before those checks; repository LINT still performs its per-descriptor performance check | focal: `patch.gates.evaluate_package_gate()` -> `patch.validation_policy.check_validation_packages()`, `patch.validation_policy.require_execution_package()`, `patch.validation_policy.check_performance_evidence_for_patch()`; LINT: `patch.gates.evaluate_repository_lint_gates()` -> `patch.gates._evaluate_lint_package()` |
| G4 Evidence qualification | promote, build | Current qualifying focal evidence satisfies pin/resolved-SHA, identity, architecture/status obligations; malformed or stale evidence fails closed | `patch.gates.evaluate_evidence_gate()` -> `patch.catalog.validation_evidence_statuses()` |
| G5 Lifecycle promotion | promote | For an untested focal patch, G0-G4 must all PASS; the gate only derives eligibility and never edits metadata | `patch.gates.evaluate_lifecycle_gate()` |
| G6 Disposition coverage | rebase, build | Fresh all-patches report plus exact revision/digest dispositions cover every non-retired patch; recipe-selected failures cannot be waived | `patch.gates.evaluate_disposition_gate()` -> `patch.disposition.compute_coverage()` and `patch.rebase.require_fresh_report()` |
| G7 Production admission | build | Complete ordered production composition passes production admission; inactive/not-ready is BLOCKED at the PA21 gate evaluator, active rejection is FAIL, and malformed/unknown output is BLOCKED | `patch.gates.evaluate_admission_gate()` -> `patch_admission.admit()` |

## Before G0: authorities and non-gate prerequisites

Keep these authorities separate:

| Authority | Owns |
| --- | --- |
| `patch.toml` | Package identity, lifecycle state, order, dependencies, conflicts, bindings, and other machine metadata |
| `SUMMARY.md` | Human-facing mirror checked by G1 for Status and Plan item; it is not the composition authority |
| `config/recipes.toml` | Named source and patch-set composition |
| `config/external-sources.toml` | External-source tracking status and porting history |
| Experiment Contract in `config/experiment-contracts.toml` | Scientific hypothesis, scope, required checks, thresholds, and acceptance policy |
| `validation.toml` and patch-local `validation/` | How a contract's declared evidence is produced |
| `evidence/validation.json` plus bound artifacts | What actually ran and was proven, with identity-bound results |

Every production patch is a package:

```text
patches/<patch-id>/
    patch.py
    patch.toml
    SUMMARY.md
    README.md                 # required when entering validation
    validation.toml           # required when entering validation
    validation/               # optional custom checks/fixtures
    evidence/validation.json  # compact tracked evidence
```

Raw logs and generated measurement dumps belong under
`artifacts/patch-validation/<patch-id>/<campaign-identity>/`, not in the
package. Patch-specific fixtures, validators, and evidence stay under the
package so its validation identity remains self-contained.

Do not write one lifecycle axis' values into another. Validation produces
evidence; metadata promotion or demotion remains a separate deliberate
change.

| Axis | Authority | Valid values | Meaning |
| --- | --- | --- | --- |
| Package state | `patch.toml` | `untested`, `validated`, `rejected`, `superseded` | Whether the package is accepted for composition |
| Source status | `config/external-sources.toml` | `planned`, `ported-untested`, `ported-benched`, `ported-validated`, `deferred-hardware`, `superseded`, `excluded`, `evidence-only` | Progress and proof level of a tracked external change |
| Run result | `evidence/validation.json` | `PASS`, `FAIL`, `BLOCKED`, `ERROR` plus named results | What this run established; it is not a metadata edit |

Validation readiness is determined by the evidence and contract, not by a
status string alone:

- `ported-benched` requires a package, current-pin evidence, a bound contract,
  and real hardware.
- `ported-validated` additionally requires every named contract check to pass.
- `deferred-hardware` records a fresh structured `BLOCKED` result and does not
  claim hardware validation.
- `eligible_for_validated_state` can support the package state `validated`;
  it does not mean `ported-validated` is a package state.

A historical source status may remain after a pin bump; only fresh current-pin
evidence is currently qualified. A stale pin, missing hardware, or harness
error removes qualification but is not by itself a rejection.

Do not collapse these layers:

| Layer | Values/meaning | Effect |
| --- | --- | --- |
| Package lifecycle | `untested`, `validated`, `rejected`, `superseded` | `patch.toml` state; controls composition |
| Source tracking | `planned`, `ported-untested`, `ported-benched`, `ported-validated`, `deferred-hardware`, etc. | `external-sources.toml` status; tracks porting/proof progress |
| Individual check | `pass`, `fail`, `blocked`, `error`, `not_applicable` | Result of that named check only |
| Contract gate | Contract-specific gate results and promotion verdict | Whether scientific obligations passed |
| Persisted eligibility | `eligible_for_validated_state` | Whether evidence supports a deliberate package-state `validated` transition |

Diagnostic PASS, adapter PASS, partial campaign, or single-contract evidence
does not authorize a lifecycle transition. `BLOCKED` means an external
prerequisite is unavailable; `ERROR` means the harness/identity/configuration
failed; `FAIL` means the required requirement was disproved.

Keep repository hygiene separate from G0-G7 results. The usual non-gate,
hardware-free checks are:

```bash
PYTHONPATH=tools python -m bigcherry check --quick
PYTHONPATH=tools python -m unittest discover -s tools/tests
PYTHONPATH=tools python -m bigcherry patch-lint --json
PYTHONPATH=tools python -m bigcherry patch-validate <patch-id>
PYTHONPATH=tools python -m bigcherry patch-explain <patch-id>
PYTHONPATH=tools python -m bigcherry patch-graph --roots <patch-id>
```

`patch-lint` is the internal LINT consumer of G1 and G3 only. It has no focal
composition, upstream checkout, current-pin evidence, disposition target, or
production-admission decision.

## G0 — Exact composition identity

G0 verifies identity, not scientific causality. A focal validate or promote
operation resolves its dependency closure; a source build or rebase resolves
the canonical named source and its exact ordered composition. The supplied
ordered list must re-resolve to the same patch IDs and content hashes. No
selected patch may be silently dropped or substituted, and rejected,
conflicting, or unresolvable composition fails closed.

See [PATCH_SYSTEM.md](../patches/PATCH_SYSTEM.md) for the detailed source
selection and composition mechanics. This page records the gate contract and
does not copy that entire authority.

## G1 — SUMMARY/documentation consistency

For a focal operation, `evaluate_summary_gate()` uses
`check_summary_for_patch()` to require `SUMMARY.md` and to check that its
canonical `Status` and `Plan item` header agree with `patch.toml`. `Group` is
not part of this live G1 comparison. The complete SUMMARY template remains in
[PATCH_AUTHORING.md](../patches/PATCH_AUTHORING.md).

Repository LINT evaluates the same rule per descriptor through
`evaluate_repository_lint_gates()` and `_evaluate_lint_summary()`. This is a
static projection of G1, not a fabricated focal `GateContext`; it does not
run G2-G7. The public `patch-gates` CLI intentionally does not expose the
internal `lint` intent.

## G2 — Revision freshness and apply readiness

G2 is observational and uses isolated worktrees. A passing gate requires a
fresh report bound to the live upstream revision, BigCherry revision, overlay
digest, selector/source identity, exact patch set, implementation digests,
dependency/conflict metadata, and a focal ID in `known_good_patch_ids`.
Report generation does not mutate lifecycle state or the shared vendor
checkout.

Use the canonical configured source name, not a filesystem placeholder:

```bash
PYTHONPATH=tools python -m bigcherry patch-rebase-check \
  --source <source-name> --json <fresh-report.json>
PYTHONPATH=tools python -m bigcherry apply --source <source-name> --dry-run
```

Do not use an unscoped apply as a generic test, and do not run a non-dry apply
as an offline gate. Never hand-build inside an identity-bound campaign build
directory.

The public gate examples are:

```bash
PYTHONPATH=tools python -m bigcherry patch-gates <patch-id> --intent validate --rebase-report <fresh-report>
PYTHONPATH=tools python -m bigcherry patch-gates <patch-id> --intent promote --rebase-report <fresh-report>
PYTHONPATH=tools python -m bigcherry patch-gates <patch-id> --intent build --source <source-name> \
  --rebase-report <fresh-report> --all-report <fresh-all-patches-report>
PYTHONPATH=tools python -m bigcherry patch-gates <patch-id> --intent rebase --source <source-name> \
  --rebase-report <fresh-report> --all-report <fresh-all-patches-report>
```

Build and rebase need the canonical `--source` selection and the fresh
all-patches report because they also evaluate G6. A stale or identity-mismatched
report is not made current by applying it.

## G3 — Validation definition and package

`patch.toml` must resolve every bound Experiment Contract before a
validation-ready patch is executed. Local, non-RD framework packages with no
external-source binding may execute a complete package-local adapter without
inventing an Experiment Contract. This exception does not waive current-pin or
architecture evidence, qualify a performance claim, or apply to experimental
enhancements. Both canonical `plan-ids` and the legacy `plan-item` identify RD
scope. `apply` and `build` are universal capabilities; all other obligations
come from the bound contract. `validation.toml` may add producers, but cannot
remove, replace, or change contract obligations or thresholds.

The canonical descriptor field is `experiment_contracts` (zero, one, or many
IDs). The singular `.experiment_contract` property is a compatibility helper
for zero/one-contract callers and fails closed for multiple IDs. A final claim
must have complete applicable evidence for every bound contract; validating
only one contract is incomplete.

Required named correctness checks are individually authoritative. One generic
`correctness` PASS cannot stand in for multiple named checks such as
`backend_reference` and `ppl_equality`.

For focal operational evaluation, `evaluate_package_gate()` first applies the
static package policy. A `not-required` result is `NA` before it reaches
`require_execution_package()` or performance-documentation checking. For a
real new validation campaign, lint grandfathering is not an execution waiver:
the campaign still calls `require_execution_package()` and must satisfy the
declared contract. Repository LINT uses `_evaluate_lint_package()` and calls
`check_performance_evidence_for_patch()` per descriptor, including the
structural optimization check.

An `optimization` tag carries a real validation obligation. Before tagging a
patch `optimization` and setting `state = "validated"`, its README must
document a real three-arm comparison: native (unmodified) llama.cpp, the
BigCherry baseline with the patch excluded, and BigCherry plus this patch.
The structural requirement and the reviewer/evidence distinction are detailed
in [PATCH_AUTHORING.md](../patches/PATCH_AUTHORING.md).

## Evidence production between G3 and G4 (not a gate)

The following is the methodology for producing qualification evidence, not an
additional PA21 gate. The live implementation is authoritative when this page
and code disagree:

- `tools/bigcherry/patch/validation_campaign.py` builds isolated
  control/validation-subject trees, runs campaign lanes, and writes evidence.
- `tools/bigcherry/experiment/` provides paired execution, contract
  aggregation, correctness/resource/trigger gates, and promotion evaluation.
- `tools/bigcherry/patch/validation.py` validates adapter packages and
  persisted evidence.
- `config/experiment-contracts.toml` provides the contract definitions.

Current campaign flags are module options for
`python -m bigcherry.patch.validation_campaign`, not top-level `bigcherry`
commands:

`--baseline-source NAME` selects the explicit named CONTROL composition from
`config/recipes.toml` (default `bigcherry`). SUBJECT adds the focal patch to
that same composition. Both resolve against the configured pin; the evidence
records the baseline name and exact patch/digest list. This does not subtract
a patch from a production source or relax dependencies: a focal already in
the selected baseline still blocks the comparison. Core-patch qualification
therefore needs a deliberately declared, reviewed baseline and its applicable
contract producers; merely selecting another source is not qualification.

| Flag | Role | Final promotion evidence? |
| --- | --- | --- |
| `--run-rd08-contract` | RD08 full contract path: paired lanes, correctness, trigger proof, promotion gate | Yes, RD08 |
| `--run-rd73-contract` | RD73 full path: activation, resource, correctness, control, paired performance, promotion gate | Yes, RD73 |
| `--run-rd08-lanes` | RD08 paired diagnostic lanes | No |
| `--run-rd04-benchmark` | RD04 paired benchmark diagnostic | No |
| `--run-rd58-state-restore` | RD58 state-restore correctness/activation diagnostic | No |

The final RD08/RD73 paths populate contract promotion evidence. The diagnostic
paths cannot make `eligible_for_validated_state` true. A generic or
contract-bound campaign without a contract-specific final producer is not
automatically a `ported-validated` proof.

### Local framework configuration (no runtime claim)

For packaged local framework patches without RD or Experiment Contract
bindings, use the explicit configuration mode. It builds the exact named
`bigcherry-native` composition once per role; it does not pretend the focal
framework patch can be removed to form a causal CONTROL.

```bash
source tools/env/bigcherry-env.sh
PYTHONPATH=tools python3 -m bigcherry.patch.validation_campaign \
  --patch <framework-patch-id> --framework-configuration \
  --hip-path "$BC_ROCM_SHIM" --amdgpu-targets <gfx-target> \
  --workdir <new-run-directory> --build-root <new-build-directory> \
  --worktree-root <isolated-source-directory>
```

This mode does not require a model or input tuning manifest and never runs
`llama-bench`. It compiles production and diagnostic `llama-server` targets,
verifies the actual generated-input copies at each build boundary, captures
completed-build identities, then runs every package adapter. Each attempt
requires fresh build directories; previous evidence is preserved.

Schema-5 `framework-configuration-v1` records cover **compiled targets**, not
observed GPU execution. They cannot qualify runtime performance, external
ported-benched status, or deferred-hardware evidence. Production end-to-end
testing remains a separate campaign/server-bench operation. Failed adapters
or stale implementation, validation, pin, or composition identities block
configuration admission.

For focal patch `X`, define and record:

1. **Apply/build identity:** anchors, match counts, already-applied behavior,
   source-tree identity, effective configuration, compiler/toolchain, runtime
   bundle, architecture, and produced binaries.
2. **Control/subject causality:** subject contains `X`; control has the same
   intended prerequisites without `X`; stock/tune/replay are separate campaign
   roles and are not aliases for validation control/subject.
3. **Correctness:** every named required check passes using the stated method.
   If the originating fault was not reproduced, record that limitation; lack
   of a crash is not proof of a fix.
4. **Activation:** the subject exercised the claimed path and the specific
   negative control did not. A generic disabled-fusion control is not a valid
   negative control for unrelated mechanisms.
5. **Performance/controls:** use the contract workload, metric, repetitions,
   and threshold; pair and alternate control/subject where the method calls
   for it. A 1/1 smoke run cannot establish a low-single-digit effect.
6. **Environment:** record model digest, target architectures, GPU/toolchain,
   visibility/topology, active processes, VRAM headroom, source pin, and exact
   commands/environment.

## G4 — Current evidence qualification

G4 consumes persisted evidence, not a README claim. The evidence authority is
`validation_evidence_statuses()`: it checks the active pin and resolved SHA,
patch/implementation and validation identities, bound contract identity,
applicable architecture and source-status obligations, named results, and
artifact bindings. Stale or missing qualifying evidence is a failure;
malformed or unknown authority output is `BLOCKED`.

| Result | Meaning | Action |
| --- | --- | --- |
| `PASS` | Requirement ran and passed with bound evidence | May contribute to qualification |
| `FAIL` | Requirement ran and was disproved | Do not promote; investigate or reject |
| `BLOCKED` | Required external prerequisite is unavailable, or authority output is incomplete/unknown | Preserve the blocker; do not call it a patch failure |
| `ERROR` | Harness, identity, or infrastructure malfunction | Fix and rerun; no claim established |
| Missing/invalid required result | Incomplete evidence | Fail closed; not qualified |

Build may treat `not-required` as `NA`. PROMOTE requires an evidence
obligation; `not-required` cannot qualify promotion. A local
framework-configuration PROMOTE is currently `BLOCKED` before evidence
resolution until the prospective canonical-composition seam exists.

The campaign normally writes:

```text
patches/<patch-id>/evidence/validation.json
artifacts/patch-validation/<patch-id>/<campaign-identity>/
```

The tracked record must bind the active pin tag and resolved SHA, patch and
validation identities, contract ID/hash, source compositions, build and
hardware identities, named check results, artifact hashes, blockers, and the
final supported claim. Evidence is append-only; a new run creates a new
campaign identity and never rewrites an old result.

The verifier commands are read-only observations:

```bash
PYTHONPATH=tools python -m bigcherry patch-verify-evidence <patch-id>
PYTHONPATH=tools python -m bigcherry patch-validate <patch-id>
```

That read-only CLI behavior does not mean evidence is absent from operational
admission. G4 and G7 consume the evidence authority as gates; the verifier
does not promote or demote metadata.

## G5 — Lifecycle promotion

G5 is PROMOTE-only. It applies when the focal package is still `untested` and
derives eligibility only after G0-G4 all PASS. A focal package in another
lifecycle state is `NA` for G5. G5 never edits `patch.toml`, `SUMMARY.md`,
external source status, or the release ledger.

There is no automatic patch-state promotion command. Before changing
`patch.toml`, `SUMMARY.md`, or an external tracked status:

1. Confirm the evidence verifier passes for the active resolved pin.
2. Confirm contract set, source composition, workload, hardware, correctness,
   activation, and named results match the requested claim.
3. Check the plan/review and record the decision through planning and ledger
   processes.
4. Update synchronized lifecycle metadata and rerun the relevant static and
   evidence checks.

Promotion requires complete current evidence for every bound contract and
`eligible_for_validated_state = true`. Demotion/rejection retains prior
evidence and records the owner, old/new state, reason, evidence identity, pin,
dependency impact, and re-promotion conditions. A stale pin, changed
contract/framework, missing hardware, or harness error calls for
revalidation/deferment, not silent rejection. Re-promotion requires fresh
current-pin evidence and a reviewed transition record.

## G6 — Revision-bound disposition coverage

G6 applies to REBASE and BUILD. It requires a fresh
`patch-rebase-check --all` report and exact revision/digest dispositions for
every non-retired patch. Rejected and superseded patches are excluded. Clean
coverage uses `CLEAN`, `CLEAN_NOOP`, or `NOT_APPLICABLE_BY_DESIGN`.

A recipe-selected patch must be clean; its failure cannot be excused by a
disposition. Only a non-selected failing patch may use a `known_broken`
disposition, and that record must bind exactly to the patch ID, target
revision, and implementation digest. Missing coverage is `FAIL`; incomplete
or malformed authority inputs are `BLOCKED`. Use
[PATCH_REFACTOR_RUNBOOK.md](../patches/PATCH_REFACTOR_RUNBOOK.md) for the
disposition and rebase commands rather than duplicating that procedure here.

## G7 — Complete production-composition admission

G7 applies to BUILD only. Admission receives every ID in the complete resolved
ordered production composition, not only the focal patch. Post-selection
rejection never removes or substitutes a patch, and admission reuses current
evidence verification. At the PA21 evaluator, inactive or not-ready is
explicitly `BLOCKED`, an active rejection is `FAIL`, and malformed or unknown
results are `BLOCKED`. An apply-only stale-evidence escape is not valid for
production G7.

The gate evaluator calls `patch_admission.admit(..., mode="production")` and
deliberately maps an inactive/not-ready result to `BLOCKED`. This is distinct
from the current production integration seam,
`patch.catalog.resolve_for_context()` ->
`patch_admission.require_admission()`: its bootstrap not-ready result is
currently `admissible=True` and does not itself raise. That existing behavior
must not be described as the PA21 G7 blocker; the gate evaluator's explicit
mapping is the documented BUILD admission contract.

## Final handoff

The handoff must identify the patch/plan/contract IDs, implementation and
validation digests, upstream tag/SHA, source compositions, model/workload/
metric, architecture/hardware/toolchain/topology, every named check and
artifact digest, activation/correctness method, measured effect and
uncertainty, blockers/errors, exact lifecycle decision, evidence/raw-artifact
paths, review, and ledger event.

For patch mechanics, use [PATCH_SYSTEM.md](../patches/PATCH_SYSTEM.md) and
[PATCH_AUTHORING.md](../patches/PATCH_AUTHORING.md). For host commands, use
[TEST.md](TEST.md). For refactor/rebase dispositions, use
[PATCH_REFACTOR_RUNBOOK.md](../patches/PATCH_REFACTOR_RUNBOOK.md). For the
contract schema, use
[EXPERIMENT_CONTRACT.md](../experiments/EXPERIMENT_CONTRACT.md).
