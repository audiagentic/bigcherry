# Patch validation runbook

Use this page to decide whether a patch helps, does not help, or remains
unknown. It is the operational guide for running and interpreting validation;
the complete package policy, contract schema, validator semantics, and
evidence provenance are in
[`../testing/PATCH_VALIDATION.md`](../testing/PATCH_VALIDATION.md).

The key rule is simple: a clean run is useful only when it ran the right
patch, workload, architecture, control, subject, and contract. A missing
prerequisite is `BLOCKED`, not a negative result and never a pass.

## Qualification levels

Choose the intended claim before running anything:

| Intended claim | Minimum proof | Honest result if a prerequisite is missing |
| --- | --- | --- |
| Applies/builds | Anchored apply, idempotent re-apply, correct source identity, and verified build | `FAIL` or `ERROR`; do not call it validated |
| `ported-benched` | Current-pin control/subject benchmark, real build and hardware identities, and no required correctness failure | `BLOCKED` or remain historical |
| `ported-validated` | Every named contract check passes, activation/control evidence is causal where needed, required architectures are covered, and evidence verifies against the active pin | `BLOCKED`; do not promote |
| `deferred-hardware` | Complete README/adapter/contract plus a fresh structured `BLOCKED` record naming the unavailable hardware prerequisite | Keep `deferred-hardware` |
| Rejected candidate | The required test actually ran and disproved the requirement | Do not reject solely because the harness or hardware was unavailable |

`patch.toml state = "validated"` is a deliberate package lifecycle decision;
it is not the same field as an external-source tracked status and it is never
set by the campaign. See the promotion section below.

## Phase 1: static and applicability preflight

Run these from the repository root before reserving a GPU:

```text
PYTHONPATH=tools python -m bigcherry patch-lint --json
PYTHONPATH=tools python -m bigcherry check --quick
PYTHONPATH=tools python -m unittest discover -s tools/tests
PYTHONPATH=tools python -m bigcherry patches --source <source-name>
PYTHONPATH=tools python -m bigcherry patch-explain <patch-id>
PYTHONPATH=tools python -m bigcherry patch-graph --roots <patch-id>
```

Interpret the gates separately:

- `patch-lint` is static. It checks package shape, metadata, summary header,
  validation adapter structure, contract resolution, required producers, and
  path containment. It does not prove evidence freshness.
- `check --quick` is deterministic, hardware-free, non-mutating local CI. It
  must not launch ROCm builds, models, or campaigns. Use `--default` and
  `--full` for the broader repository gates when the shared overlay is in an
  accepted state.
- The unit suite tests framework behavior, not GPU correctness or performance.
- `patch-explain` and `patch-graph` are the fastest way to catch an unintended
  prerequisite, conflict, source selection, or contract binding before a
  campaign.

Check applicability against the current upstream revision in an isolated
worktree:

```text
PYTHONPATH=tools python -m bigcherry patch-rebase-check \
    --source <source-name> --json <rebase-report.json>
```

For release-wide coverage, use `--all` instead of `--source`. A report status
of `CLEAN`, `CLEAN_NOOP`, or `NOT_APPLICABLE_BY_DESIGN` is eligible for apply.
Failed or quarantined patches require reconciliation or an explicitly scoped
disposition; never silently drop a patch from a selected source.

Apply only the exact source selection after the audit gate:

```text
PYTHONPATH=tools python -m bigcherry apply --source <source-name> --dry-run
PYTHONPATH=tools python -m bigcherry apply --source <source-name>
```

If a fresh rebase report authorises a partial known-good subset, use:

```text
PYTHONPATH=tools python -m bigcherry apply \
    --rebase-report <rebase-report.json> --known-good
```

Do not combine `--rebase-report` with `--source`. The report owns the exact
selection and is rejected if its upstream revision, BigCherry revision,
selection identity, overlay identity, or patch digests are stale. A partial
apply does not advance the release stage.

## Phase 2: select the correct validation execution

The generic campaign interface is:

```text
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
    --patch <patch-id> \
    --model <model.gguf> \
    --hip-path <rocm-toolchain> \
    --amdgpu-targets <gfx-target>[,<gfx-target>...] \
    --manifest <hip-autotune-manifest.json> \
    --workdir <dedicated-run-directory> \
    --build-root <identity-bound-build-root> \
    --worktree-root <content-addressed-worktree-root> \
    <patch-specific-execution-flag>
```

The patch's own `README.md`, the bound Experiment Contract, and the current
implementation of `validation_campaign.py` determine the final flag and
whether it can produce qualification. Do not assume that the generic
record/tune/promote/replay pipeline is a contract proof for every patch.

Current specialized modes have deliberately different ceilings:

| Flag | What it runs | Qualification meaning |
| --- | --- | --- |
| `--run-rd08-contract` | RD08 paired lanes, bit-identical correctness producer, subject-hit/control-miss trigger, and contract promotion gate | Current authoritative full-qualification path for RD08; still does not edit lifecycle metadata |
| `--run-rd08-lanes` | RD08 lane execution and evidence without the full correctness/trigger gate | Diagnostic only; cannot make a patch eligible |
| `--run-rd04-benchmark` | RD04 paired decode/prefill benchmark evidence | Real performance evidence; no correctness/activation contract proof, so `ported-benched` is the ceiling until the missing producers exist |
| `--run-rd58-state-restore` | RD58 real state-save/load correctness, trigger, and repeated control/subject execution | Diagnostic evidence today; does not attempt contract promotion |
| `--run-rd73-contract` | RD73 activation, resource, correctness, and paired performance contract evidence | Produces a real contract verdict, but the current generic adapter wiring may still leave `eligible_for_validated_state` false; follow the patch README and plan item |

For any other patch, a successful generic campaign can still be valuable
evidence, but it is not automatically a `ported-validated` proof. If the
bound contract's named correctness or activation producer is not wired into
the campaign, the result must remain `BLOCKED`/ineligible.

## Hardware and environment preconditions

Record the exact conditions, not just the command line:

- Use the required model, workload flags, and manifest named by the contract.
- Set both `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` to the intended
  device set. Preserve the topology required by the workload; do not reduce a
  multi-GPU test to one device to avoid a topology failure.
- Confirm the actual GPU architecture and toolchain. `--hip-path` must expose
  the compiler/tools the campaign expects (including `bin/clang` and
  `bin/clang++` on the documented Brutus setup).
- Use an idle or appropriately reserved device and record active processes,
  VRAM headroom, driver/ROCm/HIP identity, architecture, and model digest.
- Use a dedicated `--workdir`. Reuse `--build-root` only when its identity
  metadata validates. Never run `cmake --build` manually inside an
  identity-bound campaign build directory or add an unrequested target; let
  the campaign build the required binaries.
- If a required architecture, model, toolchain, or topology is unavailable,
  stop as `BLOCKED`. Do not fabricate a hardware record or treat a CPU-only
  smoke run as GPU evidence.

The concrete Brutus setup, compiler shim rules, visibility requirements, and
benchmark discipline are maintained in
[`../testing/TEST.md`](../testing/TEST.md). Read it before using a shared
ROCm host.

## Phase 3: control, subject, and stock discipline

For focal patch `X`, the campaign must materialize separate, content-addressed
source trees:

```text
BASELINE = explicit named source composition
CONTROL  = BASELINE + X's prerequisites, without X
SUBJECT  = BASELINE + the same prerequisites + X
STOCK    = pristine pinned upstream, for context only
```

The control and subject builds must be parity-checked: same requested
architecture, compiler/toolchain, CMake options, runtime bundle, workload,
model, and execution environment, differing in the intended source
composition. Record the source-tree and build identities for both arms.

The causal question is `SUBJECT - CONTROL`. A subject-versus-stock result may
measure the total overlay effect, but it cannot attribute the change to `X`
when unrelated patches are also present. If `X` is already in the baseline,
conflicts with it, or a baseline patch depends on it, the comparison is
`BLOCKED`; do not silently remove the dependency or use stock as the control.

Do not rename tuning build roles (`tune`, `replay`, `stock`) as validation roles
(`control`, `subject`). A tune campaign's promoted replay cache is not itself
proof that the patch under test activated or improved the target path.

## Phase 4: prove the right thing

### Apply and build

The apply result must identify the patch, anchors, expected match counts,
guards, already-applied behavior, and resulting source-tree identity. The
build result must bind the source tree, requested/effective configuration,
compile commands, runtime bundle, architecture, and produced binaries. A
successful compiler exit without identity binding is not sufficient evidence.

### Correctness

Run every named required correctness check from the Experiment Contract. A
generic `correctness` capability does not satisfy multiple named checks such
as `backend_reference` and `ppl_equality`. Capture the command, mode, target
architecture, return code, comparison method, pass/fail rows, and artifact
digest for each named check.

If the originating fault was not reproduced, record that fact as an inability
to establish the fix, not as proof that the patch fixed it. Absence of a crash
is not automatically correctness proof.

### Activation

For a runtime/performance claim, prove that the subject executed the claimed
path and that the disabled or unpatched control did not produce the same
positive signal. A positive marker in the subject plus a marker in the
control invalidates the selectivity proof. The negative control must be
specific to the patch mechanism; the generic
`GGML_CUDA_DISABLE_FUSION` control is not valid for flash-attention,
graph-cache, state-restore, or other unrelated mechanisms.

### Performance and controls

Use the contract's workload, metric, threshold, and required repetitions.
Control/subject runs should be paired and alternated to reduce order,
thermal, and clock bias. Preserve raw measurements and the statistical
summary. A 1/1 or one-completion-per-arm run can be a smoke signal, but cannot
resolve a low-single-digit performance claim or establish that a patch does
not help.

The acceptance threshold belongs only in the Experiment Contract. Do not copy
it into `validation.toml`, and do not promote because a benchmark artifact
exists or because a single median looks faster.

## Phase 5: interpret outcomes

Every required check and the aggregate verdict must be read individually:

| Result | Meaning | Allowed action |
| --- | --- | --- |
| `PASS` | The check ran and proved its requirement with bound evidence | May contribute to qualification |
| `FAIL` | The check ran and disproved its requirement | Do not promote; investigate, reject, or retain as a measured non-win |
| `BLOCKED` | An external prerequisite or required producer was unavailable | Do not call it a failure of the patch and do not promote |
| `ERROR` | Adapter, infrastructure, identity, or validation malfunction | Fix the harness/configuration and rerun; no claim is established |
| missing result / required `not_applicable` | Fail-closed incomplete or invalid result | Not qualified |

For a patch that may not help, distinguish these cases in the handoff:

- **Measured non-win:** the correct subject/control experiment executed, the
  path activated, correctness passed, and the measured effect missed the
  contract threshold. This is useful negative evidence, but it is not a
  promotion.
- **Not activated:** the subject did not exercise the claimed path. This says
  nothing reliable about performance of the patch.
- **Invalid control:** the control also triggered or source/build parity was
  broken. The comparison is unusable.
- **Blocked/error:** the experiment was not capable of answering the question.

## Phase 6: persist and verify evidence

The campaign normally writes the compact tracked record to:

```text
patches/<patch-id>/evidence/validation.json
```

and keeps raw build/campaign output under:

```text
artifacts/patch-validation/<patch-id>/<campaign-identity>/
```

Do not hand-edit a record to turn a result green. The evidence writer binds
the active pin tag and resolved commit SHA, patch implementation digest,
validation/framework/contract identity, source compositions, control/subject
build identities, architecture/hardware, named check results, artifact hashes,
blockers, and final eligibility. Records are append-only; a new run creates a
new campaign identity.

Verify the result against the active local pin:

```text
PYTHONPATH=tools python -m bigcherry patch-verify-evidence <patch-id>
PYTHONPATH=tools python -m bigcherry patch-verify-evidence <patch-id> \
    --no-legacy-grandfather --json
```

`patch-validate <patch-id>` is the equivalent existing-evidence command. If
the configured pin cannot resolve locally, the verifier fails closed with a
CLI error; it does not skip freshness. A passing verifier proves only the
status level it checks.

## Phase 7: promote, demote, or defer deliberately

There is no automatic patch-state promotion command. The campaign's
`eligible_for_validated_state` is a recommendation backed by its recorded
gates; a human/agent must make the metadata decision and preserve the review
record.

Before changing lifecycle metadata:

1. Confirm the evidence verifier passes for the active resolved pin.
2. Confirm the exact patch implementation, validation identity, contract set,
   source composition, workload, hardware, and named results match the claim.
3. Check the relevant plan item/review and record the decision through the
   repository's planning and ledger processes.
4. Update `patch.toml state`, the matching `SUMMARY.md` header, and the
   external-source tracked status when that logical change is tracked.
5. Rerun `patch-lint`, `check --quick`, and `patch-verify-evidence`.

For a demotion or retirement, also record the old/new values, decision owner,
reason, failing or stale campaign/evidence identity, active pin SHA, dependency
impact, and the exact conditions for re-promotion. The registry does not
cascade-edit dependent patches: if prerequisite `Y` is no longer composable,
dependent `X` must be removed from affected production selections or
reworked, and its composition/evidence must be rechecked before it is used
again.

Use these decisions:

- `untested` → `validated`: only after all required current-pin proof and
  review support production use. A clean apply or benchmark alone is not
  enough.
- `ported-untested` → `ported-benched`: only after a real current-pin
  control/subject benchmark with build/hardware identity and no required
  correctness failure.
- `ported-benched` → `ported-validated`: only after every named required
  check passes and the required architecture coverage is meaningful.
- `deferred-hardware`: keep the structured `BLOCKED` record until the missing
  execution runs. Do not convert it to a pass because the patch cannot be
  tested today.
- `validated` → `untested` or `rejected`: use only for an explicit current
  policy decision supported by evidence/review. A stale pin or harness error
  alone calls for revalidation, not silent rejection.
- `superseded`: use when upstream independently contains the same change;
  do not use it for a failed candidate.

Demotion triggers should be explicit:

- A required correctness failure or confirmed regression can justify
  `validated` → `untested` while the patch is being repaired, or
  `validated` → `rejected` when the candidate is conclusively unsuitable.
- A stale pin, changed contract/framework, missing hardware, or harness error
  removes *current qualification* but is not by itself a rejection. Rebase or
  revalidate, or use `deferred-hardware` when the methodology is ready and
  the hardware prerequisite is genuinely unavailable.
- A retired prerequisite or conflict blocks the dependent composition. It
  does not silently prove that the dependent patch is wrong; record the
  dependency decision and re-promotion criteria.
- Upstream independently implementing the same behavior is
  `superseded`, not `rejected`.

For a demoted patch, retain all prior evidence and use a new campaign identity
for re-promotion. Re-promotion requires a fresh current-pin verifier pass,
the full required check set for the target status, dependency/source
resolution, and a reviewed transition record. Do not revive a `rejected`
patch by changing one string without recording why the implementation is now
a different candidate.

Never delete or rewrite historical evidence to make a demotion or promotion
look consistent. Preserve the old record and add the new result.

## Final handoff checklist

The validation handoff is complete only when it states:

- patch ID, plan item(s), contract ID(s), and implementation/validation
  digests;
- upstream pin tag and resolved SHA;
- control, subject, baseline, and stock composition identities;
- model digest, workload/metric, target architecture(s), GPU/toolchain, and
  visibility/topology conditions;
- each named check's command, result, summary, and artifact digest;
- activation selectivity and correctness method;
- measured effect, uncertainty/repetitions, and the contract decision;
- blockers or errors, if any;
- the exact lifecycle change requested or the reason no change is justified;
- paths to `evidence/validation.json`, raw artifacts, plan/review, and ledger
  event.

For the full validator schema, custom-check restrictions, multi-contract
rules, structural grandfathering, evidence identity, and real executor
architecture, read [`../testing/PATCH_VALIDATION.md`](../testing/PATCH_VALIDATION.md).
