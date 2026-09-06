---
name: bigcherry-patch-author
description: Create and modify package-only BigCherry patches with fail-closed anchored edits, metadata, dependencies, and focused tests.
---

# BigCherry Patch Author

Purpose

Create or modify a production BigCherry patch as a package under:

patches/<patch-id>/

Own:

patch.toml;

patch.py;

anchored source transforms;

implementation-local metadata;

focused patch-mechanics tests.

Do not own hardware validation, evidence interpretation, promotion/demotion, or rebase dispositions.

Triggers

Use when asked to:

create a new production patch;

port an upstream commit/PR/fork change;

convert a legacy patch to package form;

change an existing patch implementation;

repair an anchor;

add or change requires/conflicts;

add focused tests for anchored patch mechanics.

Non-triggers

Do not use for:

GPU validation execution;

Experiment Contract scientific design except identifying an existing binding needed by package metadata;

interpreting validation evidence;

changing tracked lifecycle status;

promotion/demotion/rejection/supersession decisions;

revision-specific quarantine/disposition;

generic upstream source changes outside the patch system.

Source of truth

Read before implementation:

docs/reference/patches/PATCH_SYSTEM.md

docs/reference/patches/PATCH_AUTHORING.md

docs/reference/testing/PATCH_VALIDATION.md when the patch is already validation-ready

patches/_template/patch.toml

patches/_template/patch.py

patches/_template/README.md

relevant neighbouring packaged patches

tools/bigcherry/patch/apply.py

tools/bigcherry/patch/registry.py

tools/bigcherry/patch/patchset.py

relevant files under tools/tests/patch/

Implementation authority for anchor semantics is tools/bigcherry/patch/apply.py.

Inputs

Require enough information to determine:

canonical patch ID;

plan item(s), if any;

target source file(s);

exact source transformation;

source/upstream/fork provenance when applicable;

patch group and initial implementation state;

backend/scope metadata when known;

dependencies and conflicts;

whether an existing Experiment Contract binding already applies.

Do not invent unknown scientific metadata.

Outputs

Normally:

patches/<patch-id>/
  patch.toml
  patch.py

As applicable:

  SUMMARY.md
  README.md
  validation.toml
  validation/

README.md/validation.toml become mandatory when validation policy requires them; substantive validation design belongs to bigcherry-patch-qualification.

Focused tests belong under:

tools/tests/patch/

Do not create a new production patches/<id>.py flat module.

Workflow
1. Inspect the authoritative shape

Read the template, registry implementation, anchor implementation, and at least one comparable packaged patch.

For an existing patch, inspect its current patch.toml, patch.py, dependencies, tests, and any validation package before editing.

2. Create or preserve package identity

The directory name and patch.toml id must identify the same patch.

Required template-level metadata includes:

schema = 1
id = "<patch-id>"
order = <integer>
group = "<group>"
state = "<supported-state>"
plan-ids = []
requires = []
conflicts = []

Use additional fields only when supported by the current registry and supported by real source facts, for example current packages may use:

kind

origin

backend

plan-item

external-source

experiment-contract

requires-options

forbids-options

subsystems

hardware

validation-architectures

backends

Do not place Experiment Contract hypothesis, workload, controls, thresholds, boundaries, or hardware claims in adapter configuration.

3. Implement fail-closed anchored edits

Use the current Edit/FilePatch semantics.

Every edit must:

attach to semantic source text rather than line numbers;

have a stable unique edit ID;

use a guard recognizing its own output;

declare the expected match count;

fail if the expected anchor shape is absent or ambiguous;

include a rationale for the anchor;

remain bounded by the anchor span protections;

use applies_if only for genuinely distinct upstream shapes, never to hide a missing required anchor.

Prefer exact/narrow anchors.

Do not anchor into comments or string literals as a workaround; the patch engine intentionally noise-strips supported languages before anchor matching.

Use replace_all only for genuinely repetitive sites and set the exact expected match count.

4. Preserve atomic application

BigCherry performs an in-memory trial before writes. Do not bypass the patch engine with direct arbitrary writes into upstream files.

Patch target paths must remain relative and contained under the source root.

Never write through symlinks or .. path escapes.

5. Design for idempotence

A second application must become already-applied through the guard, not duplicate the change.

Do not use a guard that can match unrelated upstream code.

6. Handle dependencies explicitly

requires and conflicts are enforced composition semantics.

Do not assume dependencies will be silently added. Exact resolution requires the complete set to be present.

When changing either field:

inspect dependents;

ensure ordering remains valid;

add/update tests covering missing requirement or conflict where relevant.

7. Add focused tests

Test the transform, not merely metadata existence.

For each materially changed anchor, cover where applicable:

expected source shape applies correctly;

output bytes/content are correct;

second application is a no-op/idempotent;

missing anchor fails closed;

duplicate/ambiguous anchor fails closed;

guard does not match the unpatched source accidentally;

applies_if positive/negative shapes behave intentionally;

dependencies/conflicts fail closed;

target-path containment remains intact if path handling changes.

Do not require real GPUs for patch-mechanics tests.

8. Legacy migration rule

Only migrate a legacy flat representation when explicitly required.

Before deleting the old representation:

materialize old and new representations from the same immutable base;

require identical resulting source-tree hashes;

retain flat representation only where required as a compatibility fixture.

Never declare migration equivalence from visual similarity.

9. Hand off to verification

Do not mark implementation work "validated" merely because anchors and tests pass.

Pass the package to bigcherry-patch-verify.

If validation-ready metadata is also required, hand validation design to bigcherry-patch-qualification.

Verified commands

Hardware-free repository tests:

cd $BC
PYTHONPATH=tools python -m unittest discover -s tools/tests/patch

Static patch gate:

cd $BC
PYTHONPATH=tools python -m bigcherry patch-lint

Repository offline regression suite after substantive patches/ or tools/ changes:

cd $BC
PYTHONPATH=tools python -m unittest discover -s tools/tests

Do not substitute GPU campaigns for these implementation tests.

Stop conditions

Stop rather than weakening the transform when:

an anchor matches zero times unexpectedly;

an anchor matches more sites than expected;

a guard is not distinctive;

the proposed target escapes the source root;

required dependency identity is unknown;

a dependency cycle/conflict appears;

metadata would require inventing provenance or contract claims;

a legacy migration does not produce identical source-tree identity;

the requested change can only be made by making the anchor permissive enough to hide upstream drift.

Safety rules

Package-only for new production patches.

Deterministic patch implementation only.

Fail closed on ambiguous source shape.

Guard every output for idempotence.

Never use applies_if as "best effort".

Never silently auto-add or remove dependencies.

Never fabricate provenance.

Never add scientific thresholds/hypotheses to patch adapter metadata.

Never fabricate hardware evidence.

Never promote lifecycle state as part of implementation.

Never edit the shared vendor tree merely to retest a patch; use isolated/test paths owned by the tooling.

Handoff rules

To VERIFY provide:

patch ID;

files changed;

target source files;

anchor/guard behavior;

dependencies/conflicts changed;

focused tests added;

known upstream-shape assumptions.

To QUALIFICATION provide only implementation facts, not a claimed validation verdict.

To LIFECYCLE provide no recommendation unless verification/qualification has independently established the necessary facts.

Self-validation

Before returning:

Is the production patch package-only?

Does package directory equal patch identity?

Are edits anchored and fail-closed?

Does every edit have reliable idempotence behavior?

Are expected match counts explicit?

Are dependencies/conflicts intentional?

Are positive and negative patch tests present where the transform changed?

Did I avoid lifecycle and hardware claims?

If migrating legacy representation, did I prove identical source-tree output?

If any answer is no, the authoring work is incomplete.

