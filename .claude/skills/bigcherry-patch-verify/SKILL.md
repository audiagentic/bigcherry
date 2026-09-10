---
name: bigcherry-patch-verify
description: Verify BigCherry patch mechanics, local checks, apply/idempotence, dependencies, and rebase readiness without hardware.
---

# BigCherry Patch Verify

Purpose

Prove patch mechanics and repository consistency without requiring hardware.

Own:

static package lint;

focused patch tests;

deterministic local CI;

anchored apply/idempotence verification;

negative-path checks;

dependency/conflict checks;

revision-specific rebase readiness.

Do not own scientific GPU qualification or lifecycle mutation.

Triggers

Use when asked to:

test a patch without GPUs;

verify a newly authored/modified patch;

check anchors or idempotence;

test missing/duplicate-anchor failure;

run patch lint/check;

assess whether patches still apply after an upstream pin/revision change;

prepare a rebase-readiness report;

determine whether a failure is mechanical/revision-specific.

Non-triggers

Do not use for:

GPU benchmarks or correctness execution;

final Experiment Contract qualification;

promotion/demotion/re-promotion;

changing state or tracked status;

recording known_broken dispositions;

treating historical evidence as current qualification.

Evidence inspection may be requested separately, but current qualification belongs to bigcherry-patch-qualification.

Source of truth

docs/reference/patches/PATCH_SYSTEM.md

docs/reference/patches/PATCH_AUTHORING.md

docs/reference/patches/PATCH_REFACTOR_RUNBOOK.md

docs/reference/testing/PATCH_VALIDATION.md

docs/reference/testing/TEST.md

docs/reference/tooling/TOOLING.md

tools/bigcherry/patch/apply.py

tools/bigcherry/patch/registry.py

tools/bigcherry/patch/patchset.py

tools/bigcherry/patch/catalog.py

tools/bigcherry/patch/rebase.py

tools/bigcherry/patch/validation_policy.py

tools/tests/patch/

tools/bigcherry/cli/main.py

tools/bigcherry/cli/patch.py

Inputs

patch ID or changed patch set;

target source selection when applicable;

current vendor/upstream revision;

changed files;

expected dependencies/conflicts.

Optional:

requested source name for --source;

desired all-registry rebase coverage.

Outputs

Return:

static lint result;

focused/unit test result;

deterministic check result;

apply/idempotence result;

negative-path result;

dependency topology result;

rebase status per relevant patch;

exact blockers;

handoff recommendation.

Do not output a GPU qualification verdict.

Workflow
1. Establish package identity

Confirm:

production representation is packaged;

patch.toml/patch.py resolve through the registry;

state vocabulary is valid;

metadata, dependency, and implementation IDs are consistent.

Legacy flat discovery is compatibility-only and must not be accepted as a new production representation.

2. Run static package lint

Run:

cd $BC
PYTHONPATH=tools python -m bigcherry patch-lint

Treat failures as blockers.

Important: structural grandfathering is only a lint-shape exemption for unchanged historical packages. It does NOT:

prove current qualification;

authorize a new validation campaign;

authorize missing README/adapter/contract for new validation.

Any implementation, patch.toml, or tracked-status change invalidates the matching grandfather identity.

3. Run patch-domain tests

Run:

cd $BC
PYTHONPATH=tools python -m unittest discover -s tools/tests/patch

For changed anchors, ensure tests exercise:

successful transform;

idempotent second application;

missing-anchor failure;

ambiguous/duplicate-anchor failure when applicable;

guard behavior;

relevant upstream-shape branches;

dependencies/conflicts.

Do not weaken negative tests to accommodate drift.

4. Run deterministic local CI

Run:

cd $BC
PYTHONPATH=tools python -m bigcherry check

bigcherry check is:

deterministic;

non-mutating;

hardware-free;

not a GPU build/model/campaign launcher.

Do not interpret a passing check as hardware qualification.

5. Exercise patch application safely

For a configured source selection:

cd $BC
PYTHONPATH=tools python -m bigcherry audit
PYTHONPATH=tools python -m bigcherry apply --dry-run --source <source>

Use dry-run for shared-tree verification.

Do not reset/edit the shared vendor tree merely to force reapplication.

Test true second-application idempotence in controlled unit/isolated test trees.

An already-owned edit should be recognized by its guard, not inserted twice.

6. Inspect dependencies

Run:

cd $BC
PYTHONPATH=tools python -m bigcherry patch-explain <patch-id>
PYTHONPATH=tools python -m bigcherry patch-graph --roots <patch-id>

Verify:

every requires dependency is known;

conflicts do not coexist in the selected composition;

dependencies precede dependents;

no caller assumes resolve_exact() silently expands REQUIRES.

expand_composition() exists as a separate API, but exact composition does not auto-expand.

7. Check revision-specific rebase readiness

For the exact production source selection:

cd $BC
PYTHONPATH=tools python -m bigcherry patch-rebase-check --source <source>

For all non-retired registry patches:

cd $BC
PYTHONPATH=tools python -m bigcherry patch-rebase-check --all

Optional structured report:

PYTHONPATH=tools python -m bigcherry patch-rebase-check --all --json <report-path>

This check is observational and uses isolated revision-specific probing. It does not advance release stage or mutate the vendor checkout.

Interpret clean statuses according to current implementation, including:

CLEAN

CLEAN_NOOP

NOT_APPLICABLE_BY_DESIGN

Do not convert a rebase failure into a durable patch lifecycle decision here.

8. Check full offline regression suite after substantive changes

Repository reference procedure:

cd $BC
PYTHONPATH=tools python -m unittest discover -s tools/tests

Use when patches/ or patch tooling changed materially.

9. Classify failures

Mechanical/source-shape failure:

return to AUTHOR.

Revision-specific compatibility failure:

hand to LIFECYCLE for a deliberate disposition/quarantine decision if requested.

Hardware/scientific unknown:

hand to QUALIFICATION.

Do not repair one class by mutating another axis.

Verified commands
PYTHONPATH=tools python -m bigcherry patch-lint
PYTHONPATH=tools python -m bigcherry check
PYTHONPATH=tools python -m unittest discover -s tools/tests/patch
PYTHONPATH=tools python -m unittest discover -s tools/tests
PYTHONPATH=tools python -m bigcherry audit
PYTHONPATH=tools python -m bigcherry apply --dry-run --source <source>
PYTHONPATH=tools python -m bigcherry patch-explain <patch-id>
PYTHONPATH=tools python -m bigcherry patch-graph --roots <patch-id>
PYTHONPATH=tools python -m bigcherry patch-rebase-check --source <source>
PYTHONPATH=tools python -m bigcherry patch-rebase-check --all
PYTHONPATH=tools python -m bigcherry patch-rebase-check --all --json <report-path>

Do not invent a single-patch patch-rebase-check selector; current CLI selection is --source or --all.

Stop conditions

Stop and report failure when:

patch-lint fails;

deterministic tests fail;

check fails;

dry-run apply fails;

an anchor becomes missing or ambiguous;

a dependency/conflict invariant fails;

a rebase report requires reconciliation;

a package requiring validation is malformed.

Do not continue into hardware qualification merely because hardware could hide or bypass a mechanical failure.

Safety rules

Hardware-free only.

Never invoke bigcherry.patch.validation_campaign.

Do not run ROCm builds/models/campaigns.

Do not mutate lifecycle metadata.

Do not record dispositions.

Do not silently skip a failed selected patch.

Do not silently expand dependencies.

Do not edit/reset the shared vendor tree for retesting.

Do not treat lint grandfathering as validation authorization.

Rebase failure is revision-specific, not automatically rejected.

Handoff rules

To AUTHOR:

exact failed anchor/edit/test;

observed source shape;

rebase context.

To QUALIFICATION:

patch mechanics pass;

relevant current revision;

no claim beyond hardware-free verification.

To LIFECYCLE:

exact rebase report status;

implementation digest;

target revision;

source/recipe membership;

affected dependency graph.

Do not recommend a known-broken disposition for a recipe-selected patch; lifecycle policy forbids using disposition to excuse a selected patch.

Self-validation

Before returning:

Did I run/require static lint?

Did I keep every test hardware-free?

Did I test idempotence/negative paths where implementation changed?

Did I check dependencies?

Did I use --source or --all correctly for rebase checking?

Did I keep rebase compatibility separate from lifecycle state?

Did I avoid evidence/promotion claims?

If any answer is no, verification is incomplete.

