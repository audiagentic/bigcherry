---
id: RUM08
order: 8
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:40.856862+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P1
work: M
---

# Reuse upstream MTP replay and reconcile scheduling policy

## Description

Compare upstream recording against rad0013 before retirement; select one scheduler controller and preserve rank-consistent draft/history handling. Keep adaptive/traversal/suffix modes outside the initial acceptance matrix unless explicitly enabled.

## Steps

- Compare target Qwen4exp pass_key and recorded draft/verify/decode behavior with rad0013's exact intended fix, including split cap and QSA exactness/grid boundary.
- Instrument only enough to show the selected fixed-depth production draft and verify paths hit the expected replay keys. Retire rad0013 if target coverage is equivalent; otherwise keep a small semantic fix rather than its old full override.
- Inspect target scheduler prefill-interleave auto defaults, mixed-step timing controller, preemption and prefix adoption against rad0036. Select one policy authority and map its settings through native YAML.
- Preserve rad0028 rank-consistent draft decisions and rad0029 ownership behavior with actual verify rows. Keep suffix/adaptive/traversal modes disabled unless they are part of the chosen production configuration.
- Remove superseded governor/config parsing and associated dead tests/aliases once all callers are migrated. Keep unrelated diagnostic experiments out of production composition.
- Use upstream defaults only after an explicit decision; initial comparison pins the old production behavior so policy drift cannot hide behind a version bump.

## Detailed Solution & Technical Design

Upstream commit 7f5218ee already records Qwen4exp draft/verify/decode passes; subsequent pass keys include split geometry and indexer exactness decisions. Reapplying the local override wholesale can discard those newer correctness distinctions. Review the current function and compare semantic key cases instead of restoring a removed textual comment.

Latest target commits add token/block/traversal verification, adaptive depth and suffix proposals. They are available features, not compulsory work for the minimum bump. Fixed depth and the current verification mode are enough for initial acceptance. A variable-row future mode must fit admission, ownership and cache replay; do not assume testing fixed depth proves it.

rad0034/35 are optional policy experiments, rad0036 is the production governor. Port the current production objective with the existing upstream controller when equivalent; retain a small typed scheduler policy hook only for a demonstrated semantic gap. Never let two independent controllers adjust the same cap. Put timing observation at scheduler completion, not inside GPU kernels.

Astra refinement: target pass_key keys the exact QSA-bound side and slot-indexed state for decode/verify/MTP; mixed remains live. Prefer removing rad0013's full local definition after one boundary-crossing fixed-depth comparison and one ineligible mixed shape. rad0036's aged-long-prompt reservation, remaining-token order and admission budget differ from native time control. Pin prefill-interleave/mixed-step-target-ms explicitly. Keep a demonstrated required long-profile decision or withhold that profile; do not claim native equivalence from option names. rad0034/35 were already superseded locally, not new deletions.

## Code Samples & Guidance

Minimal replay key cases to inspect:
```text
same rows + same split cap + same QSA/grid side => same key
changed split/grid/exactness decision          => distinct key or explicit live path
unknown/unrecordable shape                    => explicit no-replay
```
The actual upstream key algorithm remains authoritative; these are proof cases, not a new hashing implementation.

### Deep scan implementation refinement

Main net-deletion candidates are rad0013 replay override after target semantic coverage proof and rad0036 governor after policy equivalence. Remove production entries, obsolete option/state mapping and superseded tests in the same slice. Diagnostics rad0010/39/43/44 may share route identity and bounded sink plumbing, but histogram/stats/per-token records retain different contracts and off-path costs.

Source evidence and removal conditions: docs/reviews/radiance-upstream-2026-11-18/CONSOLIDATION.md. Use patch-footprint.json to inspect every overlapping patch before changing an attachment site.

## Files

- engines/radiance/patches/rad0013_mtp_pass_replay/patch.py
- engines/radiance/patches/rad0028_rank_consistent_block_drafts/patch.py
- engines/radiance/patches/rad0029_rank_own_split/patch.py
- engines/radiance/patches/rad0036_prefill_governor/patch.py
- engines/radiance/recipes.toml
- engines/radiance/components/serving/config_loader.cpp

## Validation

One focused replay-key case table, one rank-agreement/accepted-history fixture, one mixed prefill plus decode/prefix-preemption case using the chosen controller. One fixed-depth canary in RUM09. Do not sweep all depth/verification/suffix combinations while proving the basic bump.

## Effort & Risk

M: upstream replay/governor overlap can silently change behavior. Pin one production mode and test key distinctions; leave new adaptive/suffix/traversal features outside initial qualification.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

The production mode has one controller and correct replay shape discrimination; no duplicate override loses target semantics. All ranks agree on admitted rows/history and the explicit default comparison is documented. Additional upstream modes stay unqualified until separately selected.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM04, RUM06, RUM07. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM04/RUM06/RUM07. Reuse ENG02 measurement ownership for later depth tuning; no broad tuning campaign in this migration.

Astra source review ASTRA-CONSOLIDATION.md: resolve native replay/controller choices before porting their hunks; no new generic scheduler interface. Integrated qualification remains RUM09.

## Change Log

- 2026-11-18T01:28:40.856862+00:00 (created-by): Created by codex
- 2026-11-18T01:40:00.933069+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:35.850667+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:57:12.077870+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:20:05.918412+00:00 (updated-by): Updated: section:detailed_solution, section:notes
- 2026-11-18T02:23:54.480453+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:26:51.232442+00:00 (updated-by): Updated: section:notes
