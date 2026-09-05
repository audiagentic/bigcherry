# Patch validation policy

This is the sole canonical policy for proving what a BigCherry patch does or
does not establish. It covers package requirements, Experiment Contract
binding, campaign capability, evidence identity, lifecycle decisions, and
promotion/demotion. Use [TEST.md](TEST.md) for commands and host procedures;
use [../patches/PATCH_AUTHORING.md](../patches/PATCH_AUTHORING.md) for creating
the package.

## Authority and package shape

Keep the four authorities separate:

| Authority | Owns |
| --- | --- |
| `patch.toml` + `SUMMARY.md` | Patch identity, composition, dependencies, and tracked lifecycle state |
| `config/experiment-contracts.toml` | Scientific hypothesis, scope, required checks, thresholds, and acceptance policy |
| `validation.toml` | Adapter wiring: how declared check producers are invoked |
| `evidence/validation.json` | Append-only, identity-bound results and the exact claim supported |

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

## When validation is required

| Tracked state | Package | Current evidence | Contract | Hardware |
| --- | --- | --- | --- | --- |
| `planned` | No | No | No | No |
| `ported-untested` | No, until touched | No | No | No |
| `ported-benched` | Yes | Current pin | Yes | Yes |
| `ported-validated` | Yes | Current pin; every required named check passes | Yes | Yes |
| `deferred-hardware` | Yes | Fresh structured `BLOCKED` record | Yes | No; blocked and recorded |
| `superseded` / `excluded` | No, but preserve supporting history | No | No | No |

A historical status may remain in the catalog after a pin bump; only fresh
current-pin evidence makes it currently qualified. A stale pin, missing
hardware, or harness error removes current qualification but is not by itself
a rejection.

## Experiment Contract binding

`patch.toml` must resolve every Experiment Contract before a validation-ready
patch is executed. `apply` and `build` are universal capabilities; all other
obligations come from the bound contract. `validation.toml` may add producers,
but cannot remove, replace, or change contract obligations or thresholds.

The canonical descriptor field is `experiment_contracts` (zero, one, or many
IDs). The singular `.experiment_contract` property is a compatibility helper
for zero/one-contract callers and fails closed for multiple IDs. A final claim
must have complete applicable evidence for every bound contract; validating
only one contract is incomplete.

Required named correctness checks are individually authoritative. One generic
`correctness` PASS cannot stand in for multiple named checks such as
`backend_reference` and `ppl_equality`.

## Status and verdict layers

Do not collapse these layers:

| Layer | Values/meaning | Effect |
| --- | --- | --- |
| Patch lifecycle | `untested`, `ported-benched`, `ported-validated`, `rejected`, `superseded`, etc. | Tracked package state |
| Individual check | `pass`, `fail`, `blocked`, `error`, `not_applicable` | Result of that named check only |
| Contract gate | Contract-specific gate results and promotion verdict | Whether scientific obligations passed |
| Persisted eligibility | `eligible_for_validated_state` | Whether evidence can support `ported-validated` |

Diagnostic PASS, adapter PASS, partial campaign, or single-contract evidence
does not authorize a lifecycle transition. `BLOCKED` means an external
prerequisite is unavailable; `ERROR` means the harness/identity/configuration
failed; `FAIL` means the required requirement was disproved.

## Hardware-free preflight

Run from the repository root before reserving hardware:

```bash
PYTHONPATH=tools python -m bigcherry check --quick
PYTHONPATH=tools python -m unittest discover -s tools/tests
PYTHONPATH=tools python -m bigcherry patch-lint --json
PYTHONPATH=tools python -m bigcherry patch-validate <patch-id>
PYTHONPATH=tools python -m bigcherry patch-explain <patch-id>
PYTHONPATH=tools python -m bigcherry patch-graph --roots <patch-id>
```

Use `patch-rebase-check --source <source-name>` before applying to an isolated
source. For application mechanics, use an explicit scoped dry-run:

```bash
PYTHONPATH=tools python -m bigcherry apply --source <source-tree> --dry-run
```

Do not use an unscoped apply as a generic test, and do not run a non-dry apply
as an offline gate. Never hand-build inside an identity-bound campaign build
directory.

## Real contract-execution architecture

The live implementation is authoritative when this page and code disagree:

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

## What a qualification must prove

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

## Interpret and persist the result

| Result | Meaning | Action |
| --- | --- | --- |
| `PASS` | Requirement ran and passed with bound evidence | May contribute to qualification |
| `FAIL` | Requirement ran and was disproved | Do not promote; investigate or reject |
| `BLOCKED` | Required external prerequisite unavailable | Preserve the blocker; do not call it a patch failure |
| `ERROR` | Harness, identity, or infrastructure malfunction | Fix and rerun; no claim established |
| Missing/invalid required result | Incomplete evidence | Fail closed; not qualified |

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

Verify existing evidence against the active pin:

```bash
PYTHONPATH=tools python -m bigcherry patch-verify-evidence <patch-id>
PYTHONPATH=tools python -m bigcherry patch-validate <patch-id>
```

## Promote, demote, and re-promote deliberately

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

## Final handoff

The handoff must identify the patch/plan/contract IDs, implementation and
validation digests, upstream tag/SHA, source compositions, model/workload/
metric, architecture/hardware/toolchain/topology, every named check and
artifact digest, activation/correctness method, measured effect and
uncertainty, blockers/errors, exact lifecycle decision, evidence/raw-artifact
paths, review, and ledger event.

For patch mechanics, use [PATCH_SYSTEM.md](../patches/PATCH_SYSTEM.md) and
[PATCH_AUTHORING.md](../patches/PATCH_AUTHORING.md). For host commands, use
[TEST.md](TEST.md). For the contract schema, use
[EXPERIMENT_CONTRACT.md](../experiments/EXPERIMENT_CONTRACT.md).

