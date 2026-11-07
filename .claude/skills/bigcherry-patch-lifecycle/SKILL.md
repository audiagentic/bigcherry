---
name: bigcherry-patch-lifecycle
description: Make deliberate BigCherry patch promotion, demotion, re-promotion, quarantine, disposition, and dependency-impact decisions.
---

# BigCherry Patch Lifecycle

Purpose

Own deliberate lifecycle decisions after implementation/verification/qualification facts exist.

Handle:

promotion;

demotion;

re-promotion;

rejection;

supersession;

current qualification loss;

revision-specific quarantine;

known_broken disposition;

dependency/source-selection impact.

Never infer lifecycle mutation directly from a campaign.

Triggers

Use when asked to:

promote a patch;

demote a patch;

re-promote after repair/revalidation;

request a rerun of a patch's qualification (by demoting it);

mark rejected/superseded;

quarantine a revision-specific broken patch;

record/clear a disposition;

decide whether a code change or pin bump warrants a rerun;

assess dependency impact of changing a patch's state/status;

decide whether a rebase failure is permanent or revision-specific.

Non-triggers

Do not use to:

author anchors;

run focused patch mechanics;

run GPU campaigns;

invent validation evidence;

automatically react to campaign results without an explicit lifecycle request.

Source of truth

docs/reference/testing/PATCH_VALIDATION.md

docs/reference/patches/PATCH_SYSTEM.md

docs/reference/tooling/TOOLING.md

config/external-sources.toml

patch-local patch.toml

config/recipes.toml

tools/bigcherry/patch/lifecycle.py

tools/bigcherry/patch/disposition.py

tools/bigcherry/patch/patchset.py

tools/bigcherry/patch/catalog.py

tools/bigcherry/patch/evidence.py

tools/bigcherry/patch/validation_policy.py

tools/bigcherry/patch/rebase.py

tools/bigcherry/cli/patch.py

tools/bigcherry/cli/main.py

For plan-item lifecycle changes under docs/planning/, use the repository-prescribed planning tooling; do not hand-edit generated/managed planning surfaces.

Inputs

Require:

patch ID;

requested lifecycle action;

reason;

current implementation state;

tracked status history;

current evidence verification;

target/current upstream revision;

dependency graph;

source/recipe membership.

For a known_broken disposition additionally require:

exact target revision full SHA;

exact patch implementation digest;

observed failure status;

reason;

owner;

tracking item.

Outputs

Return or perform only the explicitly requested lifecycle action, with:

axis being changed;

previous value;

new value;

evidence/rebase basis;

dependency/source-selection impact;

retained historical evidence;

required follow-up.

Do not alter unrelated axes.

## Radiance promotion and experiment retirement

`bigcherry --engine radiance patch-promote <id> --evidence @file [--profile-only]` is supported; do not assume promotion is llama.cpp-only. The configured Radiance production set is `serving-core`. Promotion places the patch there and retires it from **every other patch set and experiment**; it must refuse promotion if that would cause a source to lose the patch. The lab source inherits production plus evaluated fixes and any active experiments. Retire empty experiment declarations and migrate all build callers in the same change; no compatibility aliases. Use the CLI and qualified evidence for actual promotion, not hand-edited state.

Independent axes

Always classify the requested change before mutating anything.

Axis A: patch implementation state

Current patch registry states include durable implementation/candidate concepts such as:

untested

validated

rejected

superseded

Meaning is not revision-specific.

Current implementation describes:

rejected: candidate failed its own validation;

superseded: upstream independently implemented the same fix; the patch itself was not necessarily wrong.

Do not set either merely because an anchor broke on one revision.

Axis B: tracked historical status

Tracked statuses in source registry describe historical work such as:

planned

ported-untested

ported-benched

ported-validated

deferred-hardware

superseded

excluded

evidence-only

These are historical lifecycle records.

Axis C: current qualification

Computed from the patch's evidence records: a record qualifies when it is intact (record_digest matches), is for this patch and says the patch passed (eligibility, activation, correctness, a passing verdict for every contract the patch binds today by id, architecture coverage).

Evidence is NOT invalidated by code or digest changes, nor by pin bumps. A record taken at an older pin, against an earlier form of the patch, or under an earlier contract/validation-package hash keeps qualifying; the difference is reported as a provenance note (patch-verify-evidence, patch-gates, build and pin-bump evidence output). A contract the patch newly binds, with no verdict in the record, does disqualify: that is a deliberate requirement change.

We judge when a rerun is needed, not blanket code or digest changes. A rerun is requested deliberately by demoting the patch (Workflow 4), never triggered by a note.

This is not a mutable historical status.

Axis D: revision-specific rebase compatibility

Represented by:

patch-rebase-check;

exact revision/digest status;

optional known_broken disposition.

Never collapse this into patch state.

Workflow
1. Inspect current state before deciding

Run:

cd $BC
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-status --item <plan-item>
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-explain <patch-id>
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-graph --roots <patch-id>
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-verify-evidence <patch-id>

Where rebase compatibility matters:

PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-rebase-check --all --json <report-path>

Also inspect source/recipe membership before quarantine/demotion.

2. Check dependency impact

Before any lifecycle mutation:

inspect requires;

inspect conflicts;

inspect dependents in the dependency graph;

inspect source/recipe selections containing the patch;

determine whether dependents remain valid.

Exact composition fails closed on missing requirements.

Do not "fix" a demotion/quarantine by silently dropping the patch or its dependents.

If the patch is required by selected production patches, state the full affected closure.

3. Promotion decision

Only promote to a stronger claim when evidence actually supports it.

For current ported-benched qualification require the qualification skill's corresponding evidence result.

For ported-validated, require:

qualifying evidence (patch-verify-evidence; provenance notes do not block);

all required named correctness checks;

required activation/control/performance/resource gates;

required architecture coverage;

supported full qualification path;

patch-verify-evidence success for the claim.

A campaign completing successfully is not enough by itself.

Promotion is a deliberate mutation after qualification, never an automatic campaign side effect.

4. Demotion decision

Determine the reason first.

Evidence never goes stale on its own: a pin bump, a patch edit or a digest change is not a reason to demote.

If a change is judged to need a rerun (for example the edit changes what the patch measures or claims):

demote the patch explicitly with the lifecycle tooling, stating the change that warrants the rerun;

do NOT rewrite or delete the old evidence;

hand off to QUALIFICATION for the rerun, then re-promote on the new result.

If new scientific evidence disproves the patch's claim:

preserve old evidence;

append new evidence;

choose the lower/durable lifecycle state explicitly based on the real result.

If the implementation itself is now an invalid candidate, rejected may be appropriate only after the evidence supports that durable conclusion.

5. Supersession decision

Use superseded only when the implementation has genuinely been replaced by an upstream/other implementation satisfying the same need.

Do not use superseded as a synonym for:

broken anchor;

a pin bump or patch edit;

temporarily unavailable hardware;

performance miss;

quarantine.

Preserve provenance/history explaining the supersession.

6. Revision-specific quarantine/disposition

A rebase failure is a revision-specific fact.

Current disposition mechanism supports only:

known_broken

It is bound to the exact triple:

patch ID;

target revision;

patch digest.

Changing revision or patch digest invalidates the disposition automatically.

A disposition is NOT a standing waiver.

Recipe-selected patches may not use a disposition to excuse a failing rebase status.

A selected patch must be clean/applicable; otherwise the selection remains blocked.

For a non-recipe patch, after explicit human/agent decision, record:

PYTHONPATH=tools python -m bigcherry patch-disposition set \
  --patch-id <patch-id> \
  --revision <full-upstream-sha> \
  --digest <implementation-digest> \
  --failure-status <status> \
  --reason "<reason>" \
  --owner <owner> \
  --tracking-item <item>

List:

PYTHONPATH=tools python -m bigcherry patch-disposition list
PYTHONPATH=tools python -m bigcherry patch-disposition list --json

Clear after reconciliation:

PYTHONPATH=tools python -m bigcherry patch-disposition clear \
  --patch-id <patch-id>

Never create a disposition merely to make coverage appear green.

7. Re-promotion after repair

A changed patch.py changes the implementation digest. That does not void its validation evidence (the recorded digest becomes a provenance note), but it does void a revision+digest-bound known_broken disposition, which records a specific failure.

When a patch was deliberately demoted for a rerun, re-promotion requires:

AUTHOR fixes implementation;

VERIFY passes;

QUALIFICATION produces the requested rerun evidence;

preserve previous evidence;

deliberately promote only to the newly proven level.

Never edit a record's digest/revision fields: those are provenance, and editing them breaks record_digest integrity.

8. Deferred hardware

Use/retain deferred-hardware only when:

methodology exists;

required package/contract exists;

the blocker is genuinely hardware/prerequisite availability;

evidence contains structured BLOCKED.

Do not call a missing methodology "deferred hardware".

9. Re-run consistency gates after mutation

At minimum:

cd $BC
PYTHONPATH=tools python -m bigcherry patch-lint
PYTHONPATH=tools python -m bigcherry check
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-explain <patch-id>
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-graph --roots <patch-id>

Where the claim depends on evidence:

PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-verify-evidence <patch-id>

Where rebase/disposition changed:

PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-rebase-check --all
10. Planning/release handoff

If the lifecycle mutation changes a tracked plan item, use the repository's planning tooling rather than editing generated planning surfaces by hand.

After substantive implementation/lifecycle work, follow repository ledger requirements.

Do not edit generated release outputs as a substitute for the authoritative ledger process.

Verified commands
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-status
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-status --item <plan-item>
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-explain <patch-id>
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-graph --roots <patch-id>
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-verify-evidence <patch-id>
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-rebase-check --all
PYTHONPATH=tools python -m bigcherry --engine llamacpp patch-rebase-check --all --json <report-path>

PYTHONPATH=tools python -m bigcherry patch-disposition set \
  --patch-id <patch-id> \
  --revision <full-sha> \
  --digest <implementation-digest> \
  --failure-status <status> \
  --reason "<reason>" \
  --owner <owner> \
  --tracking-item <item>

PYTHONPATH=tools python -m bigcherry patch-disposition list
PYTHONPATH=tools python -m bigcherry patch-disposition list --json
PYTHONPATH=tools python -m bigcherry patch-disposition clear --patch-id <patch-id>

PYTHONPATH=tools python -m bigcherry patch-lint
PYTHONPATH=tools python -m bigcherry check
Stop conditions

Stop before mutation when:

evidence does not support requested promotion;

evidence status is unknown or does not qualify and the request assumes qualification;

dependency/source-selection impact has not been assessed;

a recipe-selected patch is being proposed for a known-broken waiver;

target revision/digest for a disposition is not exact;

requested demotion gives no reason beyond a pin bump or digest change (ask whether a rerun is actually wanted);

requested rejection is based only on an anchor/rebase failure;

requested supersession lacks an actual replacement;

re-promotion after a deliberate rerun request reuses the evidence the rerun was meant to replace;

required qualification path does not exist.

Safety rules

Lifecycle mutation is always deliberate.

Campaigns never auto-promote/demote.

Preserve historical evidence.

Revalidation appends new evidence.

Keep tracked history and current qualification separate.

Keep rebase status separate from lifecycle state.

Revision-bound disposition is never permanent.

Never disposition-excuse a recipe-selected patch.

Never silently drop dependencies or dependents.

Never call BLOCKED a PASS.

Never fabricate evidence to justify state.

Do not stash/reset/rebase shared-worktree changes.

Use repository-prescribed planning/ledger surfaces for their owned state.

Handoff rules

To AUTHOR:

exact implementation failure or repair required.

To VERIFY:

changed lifecycle metadata/package requires static regression;

rebase fix requires new applicability check.

To QUALIFICATION:

promotion/re-promotion requires new proof (no qualifying record, or a deliberate rerun request);

current qualification is insufficient;

scientific failure needs authoritative evidence.

From QUALIFICATION:
accept only explicit evidence facts:

pin the evidence was recorded at (provenance);

named checks;

contract gates;

architecture coverage;

strongest supported qualification;

blockers.

Never accept "campaign passed" as a replacement for those facts.

Self-validation

Before returning:

Did I identify the exact axis being changed?

Did I distinguish historical tracked status from current qualification?

Did I distinguish revision-specific rebase failure from durable state?

Did I inspect dependency/source-selection impact?

Does evidence support any promotion?

Did I preserve append-only evidence?

If setting a disposition, is it non-recipe and exact revision+digest bound?

If re-promoting after a rerun request, was the rerun evidence produced?

Did I avoid automatic campaign-driven lifecycle mutation?

Any "no" blocks the lifecycle action.

Implementation order

bigcherry-patch-author — establish package/anchor semantics first.

bigcherry-patch-verify — consumes authored packages; proves mechanics/rebase readiness.

bigcherry-patch-qualification — consumes mechanically sound packages and owns the hardware gate/evidence interpretation.

bigcherry-patch-lifecycle — consumes verification/qualification facts and owns deliberate state/disposition changes.

bigcherry-patch-workflow — implement last so its routing targets already exist and remain thin.

Cross-skill consistency rules

Production patch = engines/llamacpp/patches/<id>/ package; flat production patches are forbidden.

AUTHOR changes implementation; VERIFY proves mechanics; QUALIFICATION proves claims; LIFECYCLE changes status. No skill should absorb another's authority.

Anchors fail closed: exact expected matches, bounded spans, distinctive guards, idempotence, no permissive applies_if.

patch-lint, bigcherry check, patch tests, dry-run apply, and rebase checking are hardware-free; hardware campaigns require explicit current-task authorization.

Experiment Contract owns scientific obligations. validation.toml only wires producers.

Validation provenance is CONTROL/SUBJECT/(actual) STOCK; tuning tune/replay/stock is a different namespace.

Evidence is append-only and is not invalidated by code, digest or pin changes; recorded digests and pins are provenance notes. Reruns are deliberate (demote the patch). Never rewrite history.

Campaign execution never mutates lifecycle automatically.

Rebase compatibility/disposition and patch lifecycle are independent. known_broken is exact revision+digest bound and cannot excuse a recipe-selected patch.

Dependencies/conflicts are enforced composition. Never silently remove, substitute, or auto-add dependencies in exact resolution.

Generic full qualification must never be assumed. Re-read current validation_campaign.py; unsupported patches stop below ported-validated until real required producers exist.
## Promotion command

For qualified patches of any engine, prefer `bigcherry --engine <name> patch-promote <id>... --evidence @file` over hand-editing lifecycle state. It updates `patch.toml`, `patch.py` STATE, SUMMARY, README Promotion record, a deterministic `releases/evidence/<id>-promotion.md` record and the engine's production patch set (`[promotion] patch-set` in `engines/<name>/engine.toml`: llama.cpp `validated-enhancements`, radiance `serving-core`) atomically, gating against the engine's `[promotion] source`; removes duplicate experiment membership; runs evidence, lint, governance and production-composition checks; commits on a slice branch; opens the conventional `feat(patch):`/`fix(patch):` PR; and prints the `record_change_event` payload with plan-item-ids. Promotion evidence needs two distinct `Model:`/`Model-ID:` lines unless `--profile-only` is explicitly selected. Validated optimization evidence must include the native llama.cpp / BigCherry baseline / subject comparison required by patch-lint. Never bypass a failed promotion gate by editing `STATE` manually.
