---
name: bigcherry-patch-workflow
description: Route BigCherry patch work across authoring, hardware-free verification, qualification, and deliberate lifecycle decisions.
---

# BigCherry Patch Workflow

Purpose

Route BigCherry patch tasks to the smallest owning skill and preserve the boundaries between:

patch implementation,

hardware-free verification,

hardware qualification/evidence,

lifecycle/disposition decisions.

This skill is an orchestrator only. Do not reimplement authoring, verification, validation, evidence interpretation, or lifecycle policy here.

Triggers

Use when the request:

spans multiple patch phases;

asks to create/port/fix a patch and then determine how it should be tested or promoted;

asks "what next?" for an existing patch;

is ambiguous between patch mechanics, validation, and lifecycle;

requests an end-to-end patch workflow.

Non-triggers

Do not use as the substantive worker when the request is clearly limited to one domain:

patch package or anchored edit creation -> bigcherry-patch-author;

hardware-free lint/apply/rebase checks -> bigcherry-patch-verify;

validation methodology, GPU campaign execution, or evidence interpretation -> bigcherry-patch-qualification;

promotion, demotion, re-promotion, rejection, supersession, quarantine, disposition, or dependency impact -> bigcherry-patch-lifecycle;

comparative measurement -- A/B of builds, flags, or caches, regression checks, cost/benefit of a patch, or challenging an existing performance claim -> bigcherry-benchmark.

QUALIFICATION and BENCHMARK are distinct and both are required for a
performance claim. QUALIFICATION owns what the contract demands and what the
evidence means; BENCHMARK owns whether the two numbers being compared may be
compared at all -- arm isolation, build-composition proof, graceful teardown,
activation evidence, work equivalence. A contract-clean campaign built on a
confounded comparison is still worthless, and that failure is invisible to
QUALIFICATION because the evidence looks well-formed.

Do not use this skill as a generic llama.cpp code-review skill.

Source of truth

Read only as needed for routing:

docs/reference/patches/PATCH_SYSTEM.md

docs/reference/patches/PATCH_AUTHORING.md

docs/reference/patches/PATCH_VALIDATION.md

docs/reference/testing/PATCH_VALIDATION.md

docs/reference/tooling/TOOLING.md

tools/bigcherry/patch/

tools/bigcherry/cli/patch.py

tools/bigcherry/cli/main.py

If policy text conflicts with implementation, treat the current implementation as authoritative and flag the documentation drift.

Inputs

Minimum:

requested operation;

patch ID if one exists;

active repository/branch context.

Useful when available:

plan item;

source/fork provenance;

desired lifecycle outcome;

target upstream revision;

whether real hardware execution is explicitly authorized.

Outputs

Return a compact routing/workflow result containing:

owning skill for the current operation;

prerequisite handoffs, in order;

current blocking condition, if any;

whether GPU execution is authorized or prohibited;

final downstream handoff once the owning skill completes.

Do not duplicate the downstream skill's detailed workflow.

Workflow
1. Classify the requested operation

Route by ownership:

implementation/package/anchors/tests -> AUTHOR;

lint/check/apply/idempotence/rebase-readiness -> VERIFY;

validation package/contract/GPU/evidence/current qualification -> QUALIFICATION;

state/status/disposition/quarantine/promotion/dependency impact -> LIFECYCLE.

If several apply, preserve this default sequence:

AUTHOR -> VERIFY -> QUALIFICATION -> LIFECYCLE

Skip phases that are genuinely unnecessary.

2. Preserve the independent axes

Never collapse these concepts:

patch implementation identity/state;

tracked historical lifecycle status;

current-pin qualification from evidence;

revision-specific rebase compatibility/disposition.

Examples:

stale evidence after a pin bump does not erase historical ported-benched;

a rebase failure does not automatically make a patch rejected;

a clean validation campaign does not automatically promote lifecycle metadata.

3. Enforce the hardware intent gate

Before routing to any real GPU execution, require explicit present-task intent to run hardware. This applies to BENCHMARK exactly as it does to QUALIFICATION.

Hardware authorization is NOT implied by:

GPU paths being supplied;

a host being known to have GPUs;

a patch requiring GPU qualification;

prior hardware runs;

a request to plan validation;

a request to inspect evidence.

Without explicit authorization, route QUALIFICATION in planning/inspection mode only.

4. Route lifecycle only after evidence interpretation

A campaign may produce evidence. It must never mutate lifecycle state automatically.

After qualification, hand the interpreted result to LIFECYCLE only if the user requested a lifecycle decision or mutation.

5. Keep handoffs narrow

Each handoff must state:

patch ID;

relevant source revision/pin;

files changed or evidence inspected;

exact result that downstream work may rely on;

unresolved blocker;

whether hardware execution was authorized.

Do not forward unsupported conclusions.

Verified commands

The router may use these observational commands only when needed to classify state:

cd $BC
PYTHONPATH=tools python -m bigcherry patch-status
PYTHONPATH=tools python -m bigcherry patch-explain <patch-id>
PYTHONPATH=tools python -m bigcherry patch-graph --roots <patch-id>

Do not execute a validation campaign from this orchestrator.

Stop conditions

Stop routing and surface the blocker when:

the patch ID cannot be resolved;

package metadata is malformed;

requested lifecycle state is ambiguous about which axis it means;

a requested GPU action lacks explicit hardware authorization;

the requested qualification path does not exist for the patch;

a downstream skill reports a fail-closed condition.

Do not route around a failure by selecting a weaker check.

Safety rules

Production patches are package directories only.

Never create new production flat patch modules.

Never infer PASS from missing or unverifiable evidence.

Never infer GPU authorization.

Never make campaign completion imply lifecycle promotion.

Never rewrite historical validation evidence.

Never use revision-specific rebase failure as a reason to silently change durable patch state.

Never silently drop dependencies or conflicts from a resolved composition.

Respect shared-worktree source-control rules; do not stash/reset/rebase another actor's work.

Handoff rules

AUTHOR -> VERIFY:

implementation/package complete;

focused tests identified;

no hardware claim made.

VERIFY -> QUALIFICATION:

hardware-free gates pass;

patch mechanics/rebase readiness known;

no lifecycle promotion implied.

QUALIFICATION -> LIFECYCLE:

evidence verdict and current-pin qualification explicitly stated;

CONTROL/SUBJECT/STOCK provenance preserved;

blockers retained as blockers;

no state mutation already performed.

Any phase -> user:

surface unsupported generic qualification, dependency breakage, or explicit stop condition immediately.

Self-validation

Before completing:

Did I route rather than duplicate?

Did I keep implementation, tracked status, evidence qualification, and rebase compatibility separate?

Did I require explicit GPU intent?

Did I avoid automatic lifecycle mutation?

Did I use only commands implemented by the current repository?

Did I preserve fail-closed behavior?

If any answer is no, correct the routing before returning.

