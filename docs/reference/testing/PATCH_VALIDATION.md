# RD-patch validation methodology

Reference for how an RD/experimental patch under `patches/` proves its
claim. This document defines the portable policy and package contract; for
concrete Brutus benchmark commands and dispatch-mode invocations, see
[TEST.md](TEST.md).

## Purpose and authority

Four separate things exist, and a validation-ready patch keeps them
separate rather than collapsing them into one file:

- **Experiment Contract** (`config/experiment-contracts.toml`) — the
  scientific authority. States the hypothesis, required architectures,
  correctness checks, performance thresholds, and acceptance criteria.
  Nothing else in this stack decides *what must be proven*.
- **`validation.toml`** — the execution adapter. States *how* evidence for
  that contract gets produced: which capabilities run, which validator
  implements each one, and any custom check/fixture wiring. It never
  restates or overrides a contract's thresholds or hypotheses.
- **Evidence record** (`evidence/validation.json`) — what was *actually*
  proven: real build/hardware identities, named check results, artifact
  digests, blockers. Compact and tracked in git; large raw logs/campaign
  output live outside the patch directory (see
  [Evidence storage and provenance](#evidence-storage-and-provenance)).
- **Tracked status** (`patch-status`'s `TRACKED-STATUS` column) — a
  historical lifecycle level for the plan item, not revision-bound. It can
  stay `ported-benched` forever even if the evidence behind it goes stale.
  Whether that level is **currently qualified for the active pin** is a
  separate question, answered by the evidence verifier (`patch-verify-evidence`),
  not by the status string itself.

**Never duplicate a contract's thresholds or hypothesis into `validation.toml`
or a patch's README.** If a number needs to change, it changes in exactly
one place: the Experiment Contract.

## Canonical package layout

```
patches/<id>/
    patch.py             # implementation
    patch.toml            # catalog identity, plan-item, experiment-contract binding
    README.md              # mandatory once a patch enters validation
    validation.toml         # mandatory once a patch enters validation
    validation/             # optional -- only when genuinely needed
        checks.py             # custom validators (rare; prefer built-ins)
        fixtures/              # immutable inputs/reference data
    evidence/
        validation.json         # compact, tracked, machine-readable evidence
```

Large or generated artifacts (raw campaign output, full logs, per-run
measurement dumps) do **not** live under `patches/<id>/`. They belong under
`artifacts/patch-validation/<patch-id>/<campaign-identity>/`, outside the
tracked repo tree. `evidence/validation.json` holds the compact, permanent
record that points at that identity, not the raw data itself.

`docs/evidence/` is a different thing and is **not** where per-patch
validation authority lives — that directory is for cross-cutting incident
or investigation narratives (HI141, HTR01, HTR03-style), which span
multiple patches/systems and tell a story. A single RD patch's proof of its
own claim belongs under its own `patches/<id>/`, because the registry
hashes `validation.toml` + `validation/**` + the bound contract's hash
together as one validation identity — moving the test code elsewhere would
break that.

## When a validation package is required

| tracked-status | package required? | current evidence required? | contract required? | hardware execution required? |
|---|---|---|---|---|
| `planned` | no | no | no | no |
| `ported-untested` | no (grandfathered until touched — see [Structural grandfathering](#structural-grandfathering)) | no | no | no |
| `ported-benched` | yes | yes, current-pin | yes | yes |
| `ported-validated` | yes | yes, current-pin, every required named check passing | yes | yes |
| `deferred-hardware` | yes (methodology only) | yes, structured `BLOCKED` entry | yes | no (blocked, recorded as such) |
| `superseded` | no | no | no | no |
| `excluded` | no (evidence/reason retained if the exclusion was itself empirical) | no | no | no |
| `evidence-only` | no, unless a real patch exists and is being tested | no | no | no |

A patch's historical tracked-status may legitimately remain
`ported-benched` or `ported-validated` from past work even after a pin
bump — that string is a lifecycle record, not a live claim. What requires
package + current-pin evidence is *current qualification* of that status:
`patch-verify-evidence` answers whether the status is qualified for the
active pin right now, not whether the historical work happened at all.
Reporting a stale-but-real `ported-benched` is honest; reporting it as
currently qualified without fresh evidence is not.

Spelled out exactly, current qualification requires:

- **`ported-benched`**: a real control/subject benchmark actually ran,
  with build and hardware identities recorded. A failing required
  correctness result **forbids** current qualification at this level —
  performance alone never overrides a correctness failure.
- **`ported-validated`**: every required named check (not just the generic
  capability) passes, AND `validation-architectures` is non-empty and
  meaningful (an empty list means the verifier has no architecture
  obligation at all, which is never sufficient for this status).

## Experiment Contract binding

`patch.toml` must bind an Experiment Contract before a patch can enter
validation. **Fail closed**: a validation-ready patch (any tracked-status
requiring a package, per the table above) with no resolved contract is a
lint error, not a "validate what you can" fallback.

The contract determines every non-universal obligation:

- `apply` and `build` are the only capabilities required of *every* patch
  regardless of contract content.
- Correctness, performance, activation, and control obligations are
  derived from the bound contract's own declared checks/thresholds/scope.
  A `validation.toml` MAY add supplementary checks/producers beyond what
  the contract requires (`compute_verdict()` treats any adapter-declared
  `required = true` check as required); it may NOT alter, remove, or
  replace an Experiment Contract's scientific obligations or acceptance
  thresholds.
- A contract's required correctness checks are each individually
  authoritative. **One generic `correctness` capability PASS does not
  satisfy multiple distinct named checks** (e.g. a contract requiring both
  `backend_reference` and `ppl_equality` needs both to genuinely pass, not
  one arbitrary correctness producer standing in for both).
- **Multi-contract patches are represented canonically by
  `PatchDescriptor.experiment_contracts`, a tuple that may contain zero, one,
  or many contract IDs.** The singular `.experiment_contract` property is a
  compatibility convenience for zero/one-contract callers and fails closed
  when more than one contract is bound. Final validation must account for
  every bound contract and must not validate one contract while calling a
  multi-contract patch complete. The generic executor does not imply that an
  arbitrary N-contract composition is automatically qualified; each bound
  contract still needs complete applicable evidence.

## Status and verdict vocabulary

Keep these four layers separate when recording or reviewing evidence:

| Layer | Values or meaning | What it can change |
| --- | --- | --- |
| Patch lifecycle state | `untested`, `ported-benched`, `ported-validated`, `rejected`, `superseded`, and other catalog states | The tracked status of the patch package |
| Individual check | `pass`, `fail`, `blocked`, `error`, `not_applicable` | Only that named check; a diagnostic PASS is not promotion |
| Experiment Contract gate | Contract-specific gate results and final promotion verdict | Whether the contract's scientific obligations are satisfied |
| Persisted eligibility | `eligible_for_validated_state` | Whether evidence is sufficient to support `ported-validated` |

The invariant is fail-closed: a diagnostic PASS, adapter PASS, partial
campaign, or single-contract result does not authorize a lifecycle transition.
Promotion requires complete current evidence for every bound contract,
including activation and correctness obligations. Demotion or rejection must
preserve the evidence and reason. Re-promotion requires fresh complete
evidence at the current source/contract pin.

## Authoring `README.md`

Minimum required contents:

- Patch identity: patch id, plan-item id, bound Experiment Contract id(s).
- Scope: target hardware/architectures, prerequisites, what workload this
  addresses.
- How to invoke validation for this patch (point at the real commands —
  `patch-validate <id>`, `bigcherry.patch.validation_campaign` — see
  [TEST.md](TEST.md) for the concrete forms).
- What each non-obvious or custom check actually measures, if
  `validation.toml` declares any custom validator.
- Control vs. subject definition specific to this patch, if not obvious
  from the contract.
- Known limitations/blockers (mandatory for `deferred-hardware`).
- Where runtime artifacts and tracked evidence land.

Do **not** put scientific thresholds or hypothesis text here — that stays
in the Experiment Contract, which remains the single source of truth.

## Authoring `validation.toml`

Schema 1, a list of `[[check]]` entries:

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
```

A realistic performance-patch example, adding correctness/activation/
performance/control checks derived from the bound contract:

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

[[check]]
id = "correctness"
capability = "correctness"
validator = "autotune-campaign"
required = true

[[check]]
id = "activation"
capability = "activation"
validator = "trace-marker"
required = true
marker-regex = "BIGCHERRY_PATCH_HIT patch=<id> path=<name>"

[[check]]
id = "performance"
capability = "performance"
validator = "autotune-campaign"
required = true

[[check]]
id = "controls"
capability = "controls"
validator = "autotune-campaign"
required = true
```

### Capability and validator rules

- `apply` and `build` are universal; every other required capability comes
  from the bound Experiment Contract, never from `validation.toml` alone.
- An adapter cannot *remove* an obligation the contract/framework declares
  required — `validation.toml` wires *how* a required capability gets
  checked, it does not get to decide a required capability isn't needed.
- Unknown or unimplemented validators fail closed (an error, not a skip).
- Custom validators use `validator = "custom"` with
  `callable = "validation/checks.py:function_name"` (a file path relative
  to the patch's own `package_root`, colon, function name — not a dotted
  Python import path). The function's signature is locked to `check(ctx)`:
  exactly one positional parameter named `ctx`, no `*args`/`**kwargs`, no
  extra parameters, not async. The resolved file must exist inside
  `package_root` — a custom check cannot reach outside its own patch
  directory.

The built-in validators registered as of this writing are: `apply`,
`build`, `compile-option`, `runtime-smoke`, `architecture`, `benchmark`,
`autotune-campaign`, `backend-ops`, `trace-marker`. `custom` is a
supported special validator path (see above), not a registered built-in
implementation. Treat
`tools/bigcherry/patch/validation.py` as the normative implementation —
this document explains the *policy*, not every internal mechanic; when the
two disagree, the code is authoritative and this doc is stale and should be
corrected.

## Controls and source compositions

Two distinct provenance domains exist and must not be conflated:

- **Campaign builds** — `tune`, `replay`, `stock`. These are genuine,
  distinct build variants produced by a tune campaign (see
  `tools/bigcherry/tuning/workflow.py`): `tune` is what gets measured
  during tuning, `replay` re-applies a promoted cache, `stock` is the
  unpatched-behaviour baseline (no dispatch layer at all — see
  `[build.stock]` in `config/recipes.toml`).
- **Validation builds** — `control`, `subject` (and optionally `stock`).
  These are the roles `ValidationContext` uses when checking one patch's
  claim: `subject` carries the patch under test, `control` is the same
  source composition without it.

Do not rename or alias one set into the other. A patch's `validation.toml`
and evidence record deal in `control`/`subject` terms; a tune campaign's
own artifacts deal in `tune`/`replay`/`stock` terms. "Baseline" is a
source-composition concept in this codebase, not currently a distinct
built binary — don't invent one.

## Correctness qualification

Named Experiment Contract checks are individually authoritative. The
generic `correctness` capability is routing/planning — it tells the
executor a correctness check must run — but a PASS on that generic
capability is **not by itself** sufficient proof if the bound contract
names more than one required check. Every named required check must have
its own recorded PASS; a missing required named result is a fail/block,
not an omission to overlook. See `experiment_contract.evaluate_correctness_gate()`
— the correctness-gate implementation that independently requires every
named contract check to pass, rather than accepting one generic PASS as a
stand-in for all of them.

## Activation and performance qualification

A performance claim requires the actual target code path to have executed
— not merely a benchmark artifact existing. Presence of a `campaign_id`
and `passed=true` flag is not itself proof; the contract's declared
workload/scope/thresholds must be evaluated against the real measured
result. Explicitly distinguish a liveness/smoke check (the patch runs
without crashing) from a statistically meaningful benchmark comparison
(the patch's claimed effect is actually measured against a control with
enough repetitions to mean something) — the two are not interchangeable,
and a validation.toml/README should be explicit about which one each check
provides.

## Evidence storage and provenance

`evidence/validation.json` records, at minimum:

- Active pin: both the human-readable tag (e.g. `b10692`) and the
  **resolved commit SHA** — a pin bump makes prior evidence historical, not
  false, and re-verification must check against the resolved SHA, not just
  the tag string.
- Patch implementation identity (hash of `patch.py`) and validation
  identity — this is NOT merely a hash of `validation.toml` + `validation/**`;
  it canonically also includes `VALIDATION_FRAMEWORK_VERSION` and the bound
  Experiment Contract's id/hash (`plan_digest()`'s actual composition).
  Changing contract semantics or framework semantics invalidates
  validation evidence even when the adapter files themselves are
  byte-identical.
- Bound Experiment Contract id and hash (also folded into validation
  identity above, and recorded separately for direct lookup).
- Source compositions and control/subject (and campaign tune/replay/stock,
  where applicable) build identities.
- Hardware identity/architectures actually exercised.
- Named check results (not just an aggregate pass/fail).
- Artifact digests for anything the record references.
- Blockers, if any (structured, not free text, for `deferred-hardware`).
- The final qualification the evidence supports.

**Historical records are append-only.** Never rewrite, reinterpret, or
"migrate" an old record's meaning when the evidence schema evolves — a
schema change adds a new record shape going forward; it does not touch
what a v1 or v2 record already says happened.

## Current-pin freshness

Evidence is qualified against the **resolved upstream commit SHA**, not
against the pin's tag string alone. A pin bump does not make old evidence
wrong — it makes it historical. A patch's tracked-status can stay
`ported-benched` from that old evidence indefinitely; what changes is
whether `patch-verify-evidence` currently considers it *qualified* for the
active pin. Revalidating against a new pin **creates new evidence**; it
never erases or overwrites the previous record.

## Structural grandfathering

A one-time exemption exists so this standard doesn't retroactively break
every existing nonconforming patch the day it's adopted. Its exact
boundary:

- It is a **lint-shape exemption only** — it lets `patch-lint` report an
  old, untouched package as grandfathered/non-current instead of failing
  outright.
- It is bound to: the current policy/version identifier of this standard,
  the patch's implementation digest (`patch.py`), the `patch.toml` digest,
  and the patch's normalized tracked-status set. **Any** change to any of
  these invalidates the exemption immediately — there is no way to touch a
  grandfathered patch's metadata or status and keep the exemption.
- There are **no hardcoded per-patch exceptions** by name.
- It **never authorizes starting a new validation execution.** Regardless
  of lint-side grandfather status, actually running a new validation for a
  patch requires the real README + `validation.toml` + a resolved
  Experiment Contract to exist first. A grandfathered patch that has never
  been touched cannot silently accumulate new "validated" evidence without
  first getting a real package.

## Deferred hardware

`deferred-hardware` still requires the full package (README +
`validation.toml` + a resolved contract) — missing hardware is not the
same thing as missing methodology, and the methodology should describe the
validation this patch *would* need once the hardware exists. The dynamic
evidence side records a structured `BLOCKED` entry, not silence. Never
convert "we could not reproduce the originating fault/condition" into an
affirmative correctness claim — absence of an observed failure is not
proof the underlying issue is fixed, especially where the fault was itself
hard to reproduce in the first place.

## Real contract-execution architecture

The real per-lane executor that produces a contract's actual measured
evidence (as opposed to `validation.toml`'s adapter checklist, which only
proves an adapter-declared check ran, not that it satisfies the bound
contract's own thresholds) lives in `tools/bigcherry/experiment/`, not in
`tools/bigcherry/campaign/`. The implementation authority is the current
source under `tools/bigcherry/experiment/`, the patch campaign in
`tools/bigcherry/patch/validation_campaign.py`, the adapter/evidence logic in
`tools/bigcherry/patch/validation.py`, and the contract definitions in
`config/experiment-contracts.toml`.

- `tools/bigcherry/experiment/execution.py` — the paired control/subject
  lane runner:
  - `Runner` / `RunnerOutput` (`returncode`, `stdout`, `stderr`, `.combined`)
    — the execution contract. A real runner shells out via `subprocess.run`;
    tests inject a fake one. **A nonzero `returncode` from either arm raises
    `LaneExecutionError` before any metric is extracted** — a failed
    benchmark can never masquerade as a valid measurement.
  - `run_paired_lane(metric, control_command, subject_command, pattern,
    pairs, runner, ...)` — alternates control/subject execution order each
    pair (so clock/thermal drift can't bias the comparison), reuses
    `campaign/benchmark.py`'s already arm-name-neutral
    `block_bootstrap_effect()` for the paired geometric effect + bootstrap
    CI. Returns a `PairedLaneRun` (raw `.runs`, `.stats`).
  - `metric_for_workload()` / `WORKLOAD_METRIC` — the one place workload
    tags (`decode`, `prefill`) map to llama-bench metric names
    (`tg128`, `pp512`). Extend only when a new workload is actually being
    validated.
  - `lane_effect_from_run(role, metric, run)` — turns a `PairedLaneRun`
    into a `LaneEffect` for `contract.py`'s aggregation functions.
  - `trigger_evidence_from_marker_probe(lane_id, role, positive_hit)` —
    turns a boolean trace-marker observation into real `TriggerEvidence`.

- `tools/bigcherry/experiment/contract.py`'s execution-facing surface
  (schema/hashing covered above under [Contract shape](#contract-shape)):
  - `aggregate_contract_effects(contract, effects, target_metric, ...)` —
    computes `target_kernel_gain_pct` from positive-role effects and
    `max_control_regression_pct` across **all control-role effects,
    regardless of their metric** — a control lane legitimately measures a
    structurally different workload than the positive lane (RD08:
    positive=decode/tg128, control=prefill/pp512), so this must never
    filter control effects by the positive lane's target metric.
  - `evaluate_correctness_gate()` / `evaluate_trigger_proof()` /
    `evaluate_resource_gate()` / `evaluate_promotion_gate()` — the
    independent, fail-closed gates a contract's evidence must pass. Missing
    or ambiguous evidence is always a FAIL/BLOCKED, never a silent pass.
  - `evidence_ref_for_lane(contract, role, workload_tag, model_ref, ...)` /
    `ContractEvidenceRef.document()` — the real per-lane provenance sidecar
    (contract id, contract hash, optimization id, role, workload, model
    ref) attached to persisted lane evidence. Use `contract.id`, never a
    `contract_id` attribute — `ExperimentContract`'s real field is `id`.

- `tools/bigcherry/patch/validation_campaign.py`'s validation-domain build
  and lane wiring:
  - The validation domain's **subject** build is a real, independently
    built `validation-subject` binary from `patched_src`, built with the
    *exact same* `extra_cmake_args` as `control` (`[]`) — **not** the
    `tune` build, which carries `GGML_HIP_AUTOTUNE`/`AUTOTUNE_RECORD`/
    `ROUTING_TRANSFORM` instrumentation the control build never had. Using
    the tune build would confound a measured lane effect with
    instrumentation overhead, not just the patch under test.
  - `assert_validation_subject_parity(control_build_evidence,
    validation_subject_build_evidence, patch_id)` — fails closed
    (`PatchCampaignError`) unless `effective_configure` AND
    `effective_build_id` both match control. Deliberately does **not**
    compare `runtime_bundle_hash`/`compile_verification_id`/full
    `campaign_identity()` — source content legitimately differs between
    control and subject; that's the entire point of the comparison.
  - `rd08_validation_lane_commands()` / `run_rd08_validation_lanes()` —
    RD08's real positive(decode)/control(prefill) lane pair, executed in a
    sanitized environment (`campaign.benchmark.sanitize_environment(mode=
    "stock")` plus stripping `BIGCHERRY_*`/`GGML_CUDA_DISABLE_FUSION`, not
    raw `_hip_env()`, so inherited dispatch/tune overrides from the ambient
    shell can never contaminate a lane run). Persists a bound
    `artifacts/validation-lanes.json` (contract id/hash, per-lane metric,
    raw commands, raw stdout/stderr/returncode, paired runs, bootstrap
    stats, resolved model ref/path, and the actual control/subject build
    identities that ran) — bound into `evidence.py`'s `_artifact_refs()`
    so it is tracked in the validation record, not merely written.
  - Gated behind an opt-in `--run-rd08-lanes` CLI flag. It is a diagnostic
    lane producer and cannot by itself populate a promotion verdict.

### Current campaign capabilities

These flags belong to `python -m bigcherry.patch.validation_campaign`, not to
the top-level `bigcherry` command:

| Mode | Purpose | Can populate final promotion evidence? |
| --- | --- | --- |
| `--run-rd08-lanes` | RD08 paired diagnostic lanes | No |
| `--run-rd08-contract` | RD08 full contract qualification, including correctness and trigger proof | Yes, for RD08 |
| `--run-rd04-benchmark` | RD04 benchmark diagnostic | No |
| `--run-rd58-state-restore` | RD58 state-restore correctness/activation diagnostic | No |
| `--run-rd73-contract` | RD73 full contract qualification, including activation/resource/paired performance/control/correctness gates | Yes, for RD73 |

The RD08 and RD73 final paths compose their named evidence through the
contract aggregation and promotion gates and populate `contract_promotions`.
`compute_persisted_validation_eligible()` still requires complete applicable
evidence for every bound Experiment Contract. A clean generic campaign,
adapter verdict, RD04 benchmark, or RD58 diagnostic cannot make a patch
eligible for `ported-validated`.

## Validation workflow

**Current implementation note:** `patch-validation-campaign` is not a
generic final qualifier for every contract. The authoritative full paths
currently exposed here are `--run-rd08-contract` for RD08 and
`--run-rd73-contract` for RD73. `--run-rd08-lanes`, `--run-rd04-benchmark`,
and `--run-rd58-state-restore` are diagnostic-only. For any other bound
contract, a clean campaign run alone is not proof of a `ported-validated`
claim; inspect the current campaign implementation and contract-specific
evidence producers.

1. Bind (or, if none exists yet, first author) the patch's Experiment
   Contract.
2. Write `README.md` + `validation.toml`.
3. Run `patch-lint` — confirms the static package is well-formed and every
   required capability has a producer.
4. Run the real, isolated patch validation campaign (see
   [TEST.md](TEST.md) for the concrete Brutus commands).
5. Inspect the named check verdicts individually, not just an aggregate
   pass/fail.
6. Let the validation campaign/evidence writer append the compact evidence
   record to `patches/<patch-id>/evidence/validation.json`; do not hand-author
   or edit a record to manufacture a PASS. If a custom execution path writes
   evidence, it must use the same schema, identity bindings, artifact hashes,
   and append-only writer contract.
7. Run `patch-verify-evidence` — confirms the evidence is current-pin and
   sufficient for the status being claimed.
8. Update the tracked status to **exactly** what the evidence actually
   proves — never a status the evidence doesn't support, even if that
   means a patch stays at `ported-benched` rather than advancing to
   `ported-validated`.

## Worked examples

- **Simple single-contract performance patch** — bind one contract
  declaring a performance threshold and required architectures;
  `validation.toml` wires `apply`/`build`/`correctness`/`performance` to
  built-in validators; evidence records the control/subject build
  identities and the real measured effect against the contract's
  threshold.
- **Deferred-hardware patch** — README documents the target hardware and
  the validation that would run once available; `validation.toml` still
  declares the full intended check list; evidence contains a structured
  `BLOCKED` entry naming the specific hardware limitation, with no
  correctness/performance claim recorded.

`patches/1204_rd08_q6k_mmvq_vdr2/` is useful **historical precedent** for
the general shape (it has both a `validation.toml` and an
`evidence/validation.json`), but its committed evidence is legacy schema
and the executor plumbing behind it is still evolving — do not copy it as
a literal template without checking it against the current schema and
`tools/bigcherry/patch/validation.py`.

## Anti-patterns

- Thresholds or hypothesis text duplicated into `validation.toml` or
  README instead of living only in the Experiment Contract.
- README-only "evidence" — a claim with no `evidence/validation.json`
  behind it.
- Raw logs committed as if they were the evidence record, instead of a
  compact tracked summary pointing at externally-stored raw output.
- Stale-pin evidence treated as satisfying a current-pin status claim.
- `tune`/`replay` used as aliases for `control`/`subject`, or vice versa.
- A generic `correctness` capability PASS presented as satisfying multiple
  named contract checks.
- A validation-ready patch with no resolved Experiment Contract, silently
  falling back to apply/build-only validation.
- A multi-contract patch validated against only one of its bound
  contracts and reported as fully validated.
- A tracked-status promoted (e.g. `ported-benched` → `ported-validated`)
  because a benchmark came back clean, when the evidence doesn't actually
  support the stronger claim (e.g. the original fault being fixed was
  never independently reproducible in the first place).

## Related commands and references

- `patch-lint` — static package/policy gate.
- `patch-verify-evidence` — dynamic, current-pin evidence/freshness gate.
  `patch-validate` delegates to evidence verification (it inspects
  existing evidence — it does not itself execute a validation run).
- `bigcherry.patch.validation_campaign` — the real validation execution
  path (materializes isolated subject/control trees and runs the real
  campaign).
- [TEST.md](TEST.md) — concrete Brutus benchmark/dispatch-mode commands.
- `config/experiment-contracts.toml` — the Experiment Contract registry.
