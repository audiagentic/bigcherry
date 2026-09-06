---
name: bigcherry-patch-qualification
description: Plan and execute explicitly authorized BigCherry patch qualification, interpret evidence, and preserve scientific provenance.
---

# BigCherry Patch Qualification

Purpose

Own the validation/qualification domain:

determine what the bound Experiment Contract requires;

author/review validation methodology and adapter wiring;

inspect existing evidence;

plan hardware execution;

run GPU campaigns only behind explicit intent;

interpret named results and current-pin qualification.

This skill does NOT mutate patch lifecycle/status automatically.

Triggers

Use when asked to:

prepare a patch for validation;

author/review README.md or validation.toml;

determine required validation;

run a real GPU validation campaign;

interpret evidence/validation.json;

determine whether evidence supports ported-benched or ported-validated;

diagnose PASS, FAIL, BLOCKED, or ERROR;

assess stale/current evidence;

determine whether generic qualification exists for a patch.

Non-triggers

Do not use for:

ordinary anchored patch implementation;

hardware-free patch mechanics;

changing patch lifecycle state/status;

revision-specific disposition/quarantine;

inventing an Experiment Contract on behalf of missing scientific requirements without explicit contract-authoring scope.

Source of truth

Policy:

docs/reference/patches/PATCH_VALIDATION.md

docs/reference/testing/PATCH_VALIDATION.md

docs/reference/testing/TEST.md

Identity/scientific authority:

config/experiment-contracts.toml

patch patch.toml

Implementation authority:

tools/bigcherry/patch/validation.py

tools/bigcherry/patch/validation_policy.py

tools/bigcherry/patch/evidence.py

tools/bigcherry/patch/validation_campaign.py

tools/bigcherry/experiment/

relevant patch-local validation/

relevant tests under tools/tests/patch/

If the canonical doc and current implementation disagree, implementation wins and the drift must be reported.

Inputs

Required for planning:

patch ID;

current patch descriptor/metadata;

bound Experiment Contract(s);

active upstream pin;

intended qualification claim.

Required before hardware execution:

explicit user/task authorization to execute hardware;

model path;

HIP path;

AMDGPU target(s);

manifest path;

isolated work directory;

any patch-specific required inputs/flags;

explicit GPU visibility suitable for the test where required.

Do not infer missing hardware authorization.

Outputs

Planning mode:

required capabilities/checks;

validation package gaps;

supported execution path, if any;

exact hardware prerequisites;

expected evidence/result interpretation;

explicit statement that no hardware was run.

Execution mode:

campaign invocation/result;

CONTROL/SUBJECT/STOCK attribution;

named check outcomes;

contract gate outcome where implemented;

current-pin evidence verification;

strongest qualification actually supported;

blockers;

lifecycle handoff recommendation only.

Never mutate lifecycle automatically.

Core authority model

Keep these four objects separate:

Experiment Contract:
scientific authority for hypothesis, scope, required checks, thresholds and acceptance.

validation.toml:
execution adapter describing how required evidence is produced.

evidence/validation.json:
compact tracked record of what actually happened.

tracked status:
historical lifecycle record, not proof that current evidence is fresh.

Never duplicate contract thresholds or hypothesis text into validation.toml or README.

Validation-ready package requirements

For validation-ready RD patches, policy may require:

patches/<id>/
  patch.py
  patch.toml
  README.md
  validation.toml
  validation/              # optional
  evidence/
    validation.json

Large/generated raw campaign output belongs outside the patch directory under:

artifacts/patch-validation/<patch-id>/<campaign-identity>/

Do not commit large logs as the authoritative evidence record.

Workflow
1. Determine tracked/current obligation

Inspect:

patch implementation state;

tracked status(es);

bound contract(s);

validation architectures;

existing evidence.

Statuses requiring a full validation package include:

ported-benched

ported-validated

deferred-hardware

A deferred-hardware patch still requires methodology, adapter, and resolved contract. Missing hardware is represented as structured BLOCKED, not missing methodology.

2. Resolve the Experiment Contract

Fail closed if validation requires a contract and it cannot be resolved.

The Experiment Contract determines non-universal obligations.

Universal capabilities:

apply

build

Correctness, performance, activation, controls, resources, architectures, and similar obligations are contract-derived.

Adapter-required supplementary checks may strengthen requirements but may not remove or weaken contract requirements.

3. Build/review validation.toml

Use the current schema/parser in tools/bigcherry/patch/validation.py.

Current canonical schema uses schema = 1 and [[check]] entries.

Each check identifies:

id;

capability;

validator;

required.

Built-in validators must be those currently registered by implementation.

Custom validators:

validator = "custom"
callable = "validation/checks.py:function_name"

The callable must:

remain inside the patch package;

exist;

be synchronous;

have exactly check(ctx);

have no extra parameters, *args, **kwargs, or async form.

Unknown validators fail closed.

4. Lint before execution

Run:

cd $BC
PYTHONPATH=tools python -m bigcherry patch-lint

A grandfathered package shape does not authorize a new campaign.

Real validation execution requires the actual execution package regardless of grandfather status.

5. Determine whether a real qualification path exists

Do NOT assume generic qualification.

Current validation_campaign.py is patch-specific.

Verified execution modes currently include:

--run-rd08-lanes

RD08 only;

diagnostic lane evidence;

does not determine eligibility.

--run-rd08-contract

RD08 only;

authoritative RD08 full contract path;

real lanes + correctness + trigger proof;

can populate the contract promotion needed for validated eligibility when all other gates pass.

--run-rd04-benchmark

RD04 only;

real paired performance evidence;

diagnostic for eligibility;

does not prove correctness/activation/full promotion.

--run-rd58-state-restore

RD58 only;

real state-restore correctness/activation/control evidence;

diagnostic for eligibility;

does not perform contract promotion.

--run-rd73-contract

RD73 only;

real activation/resource/performance/control/correctness contract evaluation;

populates RD73 contract_promotions;

current implementation explicitly states this flag alone cannot make eligible_for_validated_state=True because generic adapter rebinding is not complete;

requires --rd73-corpus.

For other patches, or unsupported contract shapes, generic N-contract full qualification is unavailable unless current code has since gained an explicit producer/path.

Never convert an ordinary clean campaign into full contract qualification when the implementation cannot produce it.

6. HARDWARE INTENT GATE

Before launching any campaign, evaluate:

Was real hardware execution explicitly requested/authorized in the current task?

If NO:

stop at planning/static inspection;

provide the verified command form and prerequisites if useful;

do not run model/GPU/build/campaign commands;

state "hardware execution not authorized/run".

If YES:

continue only with the exact implemented path for that patch.

Do not infer YES from environment availability.

7. Prepare isolated hardware execution

Base verified command shape:

cd $BC
export PYTHONPATH=tools

python3 -m bigcherry.patch.validation_campaign \
  --patch <patch-id> \
  --model <model.gguf> \
  --hip-path <hip-path> \
  --amdgpu-targets <gfx-target> \
  --manifest <manifest.json> \
  --workdir <isolated-workdir> \
  [--build-root <build-root>] \
  [--worktree-root <worktree-root>] \
  <supported-patch-specific-mode>

Use only flags present in the current parser.

For multi-GPU paths requiring ambient visibility, set the exact intended visibility explicitly. The current Brutus procedure requires both where applicable:

export HIP_VISIBLE_DEVICES=0,1
export ROCR_VISIBLE_DEVICES=0,1

Do not expose unintended heterogeneous devices.

--hip-path must resolve to a toolchain layout containing the compiler names expected by the campaign.

Never manually alter identity-bound campaign build directories.

8. Preserve validation provenance roles

Validation roles:

SUBJECT: source composition with the patch under test.

CONTROL: same intended source composition without the patch under test.

STOCK: separately unpatched stock baseline only when genuinely built/recorded.

Never use STOCK as an alias for CONTROL.

Never relabel tuning-domain builds:

tune

replay

stock

as validation control/subject.

The domains are distinct.

For causal patch qualification, compare the actual validation CONTROL and SUBJECT generated for that claim.

9. Interpret correctness

Every named Experiment Contract correctness check is individually authoritative.

A generic correctness capability PASS does not satisfy several named checks.

Missing named required evidence is not PASS.

A failing required correctness result forbids current ported-benched qualification regardless of benchmark performance.

10. Interpret activation

A performance claim requires proof that the target path actually executed.

Activation evidence normally requires:

positive SUBJECT/path hit;

disabled/CONTROL negative proof as required by the contract.

An artifact merely existing or a caller saying passed=true is insufficient.

Liveness is not causal activation or performance proof.

11. Interpret performance

Use the actual contract thresholds/statistics.

Do not substitute:

one-off A/B;

single-sample results;

benchmark-file existence;

tuning results from a differently instrumented build.

CONTROL and SUBJECT configuration parity must be preserved except for the patch-under-test source difference.

12. Interpret outcomes fail closed

Valid validator outcomes:

PASS

FAIL

BLOCKED

ERROR

Rules:

BLOCKED is never converted to PASS.

ERROR is not FAIL/PASS evidence.

inability to reproduce an originating fault is not proof that the fault is fixed.

missing/stale/tampered/fabricated evidence is not PASS.

13. Preserve evidence immutability/history

Evidence must bind, as applicable:

human pin/ref;

resolved upstream commit SHA;

patch implementation identity;

validation/framework identity;

Experiment Contract ID/hash;

CONTROL/SUBJECT/STOCK build identities;

actual hardware/architectures;

named check results;

referenced artifact digests;

blockers;

supported qualification.

Historical evidence is append-only.

On:

pin change;

patch implementation change;

validation-plan/framework change;

contract semantic/hash change;

old evidence becomes historical as appropriate. Do not rewrite it to look current.

New validation creates new evidence.

14. Verify persisted evidence

Use:

cd $BC
PYTHONPATH=tools python -m bigcherry patch-verify-evidence <patch-id>

Equivalent evidence-inspection alias:

PYTHONPATH=tools python -m bigcherry patch-validate <patch-id>

These inspect existing evidence. They do NOT launch hardware.

Evidence freshness is checked against the resolved active-pin commit SHA, not merely a tag/ref string.

15. Determine strongest supported qualification

ported-benched current qualification requires at least:

real CONTROL/SUBJECT benchmark execution;

recorded build identities;

recorded hardware identities;

no failing required correctness result;

current-pin evidence.

ported-validated requires:

current-pin evidence;

every required named check passing;

required contract gates satisfied by real producers;

non-empty meaningful validation architecture coverage;

current implementation capable of establishing full validated eligibility.

If the patch lacks a supported full contract execution path, say so and stop below ported-validated.

16. Handoff; never auto-promote

Campaign execution and evidence verification must not mutate lifecycle automatically.

Return the strongest evidence-supported state as a recommendation to bigcherry-patch-lifecycle.

Verified commands

Hardware-free:

PYTHONPATH=tools python -m bigcherry patch-lint
PYTHONPATH=tools python -m bigcherry patch-verify-evidence <patch-id>
PYTHONPATH=tools python -m bigcherry patch-validate <patch-id>

Hardware command entrypoint:

python3 -m bigcherry.patch.validation_campaign

Required base arguments currently include:

--patch
--model
--hip-path
--amdgpu-targets
--manifest
--workdir

Known current patch-specific options:

--run-rd08-lanes
--run-rd08-contract
--run-rd04-benchmark
--run-rd58-state-restore
--run-rd73-contract
--rd73-corpus

Always re-read the current parser before constructing a real invocation.

Stop conditions

Stop qualification when:

required contract cannot resolve;

more than one contract is bound but current execution cannot genuinely qualify all required obligations;

validation.toml cannot produce a required capability;

named correctness evidence is missing;

activation/control proof is missing for a causal performance claim;

required hardware is unavailable -> record/report BLOCKED;

hardware authorization was not explicit;

current code lacks a full qualification path for the patch;

evidence is stale/tampered/mismatched;

CONTROL/SUBJECT attribution is ambiguous;

build parity fails;

any required gate reports FAIL/BLOCKED/ERROR.

Do not weaken the requested status to force a PASS.

Safety rules

Explicit hardware authorization before any GPU execution.

Fail closed on missing evidence/producers.

Never fabricate hardware evidence.

Never rewrite historical evidence.

Never manually mutate identity-bound campaign build trees.

Never substitute ad-hoc A/B for contract evaluation.

Never conflate CONTROL, SUBJECT, STOCK, tune, or replay.

Never treat a campaign as lifecycle authority.

Never duplicate contract thresholds/hypotheses in adapter/README.

Never claim generic full qualification where current implementation is patch-specific.

Handoff rules

To AUTHOR:

missing instrumentation/producer must be implemented;

exact required evidence signal and contract obligation.

To VERIFY:

static/package/mechanical failure.

To LIFECYCLE:

patch ID;

current pin/resolved SHA;

evidence verification status;

named check results;

contract promotion/gate result;

architecture coverage;

blockers;

strongest supported qualification;

explicit note that lifecycle was not mutated.

Self-validation

Before returning:

Did I use the Experiment Contract as scientific authority?

Did I keep adapter/evidence/status separate?

Did I check whether the patch actually has a supported full qualification path?

If hardware ran, was authorization explicit?

Did I preserve CONTROL/SUBJECT/STOCK attribution?

Did I inspect named correctness checks individually?

Did I treat BLOCKED/ERROR correctly?

Did I preserve append-only evidence?

Did I verify current pin by resolved revision?

Did I avoid lifecycle mutation?

Any "no" means qualification is incomplete.

