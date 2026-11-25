---
id: RSC12
order: 0
plan: run-serving-configuration
state: in_progress
created-at: '2026-11-13T22:56:54.680645+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Native YAML deployment variants and measured-cache reuse

## Description

RUM04 delivered one native YAML per deployment with x-bigcherry, variant-of/differs and removal of TOML catalogue/layout. Retain those decisions. This item owns readable named YAML selection, variants and explain equivalence, not another inheritance system. Cached measurements are data, not launch configuration. No --set or --manifest.

## Steps

Use existing 16 deployment YAMLs and native parser/closed extension; validate variant-of/differs against resolved native and BC values, no cross-file inheritance engine. Authored site/profile policy YAML belongs to RSC01/PWR08; lab arms edit one targeted YAML field. Named start always resolves current sealed YAML; RSC02/RSC03 own measured cache and fresh admission. Preserve exact execution evidence separately (WSL01/MRC06). Update actual config/list/explain/release/campaign consumers, no second model catalogue.

## Detailed Solution & Technical Design

docs/design/radiance-native-yaml-launch.md sections 2 and 6; docs/design/radiance-native-config-x-bigcherry.md remains authoring authority except superseded launch transport. Existing variant checks and arm_yaml tooling, no TOML/flag translator or generated effective YAML. Three separate data classes: authored YAML, measured cache, run evidence. Only measured data reused on compatible named start.

## Code Samples & Guidance



## Files

- `engines/radiance/components/serving/` (config loader, variants, explain provenance, stored manifests; tests)
- `engines/radiance/configs/flashnext.toml`, `engines/radiance/configs/qwen27b.toml`, `engines/radiance/configs/README.md`
- `engines/radiance/layouts/` (removed at step 6)
- `tools/bigcherry` layout readers (switch to `bc-serve explain --json`)
- `docs/reference/engines/radiance/SERVING_MANIFEST.md`, `RUNBOOK.md`

## Validation

Card-free strict YAML and variant-of/differs refusals, duplicate/unknown/wrong level/missing variant, explain provenance, measured-cache identity changes, semantic projection of all 16 deployment YAMLs. Named start revalidates YAML and live room; old manifest launch refused. Hardware dense and helper native starts preserve equivalent effective settings and output/quality under RSC11/RHP12. No legacy catalogue or translator.

## Effort & Risk

M. Risks: two input formats side by side during the move (bounded by the projection parity test and by retiring
layouts at S16); a stored manifest reused when it should not be (the three-way split above: anything that is not a
static identity is checked live at every start); the model identity (`plan_cache.cpp` model_identity) hashes size,
header and tables, not every tensor byte, so the static key uses a full artefact digest where exact weights matter;
concurrent edits of `serving.cpp`, `manifest_writer.cpp` and the generated registry (RSC11's single integrator owns
those; the loader and adapter land in their own files).

## Standards

Architecture standards: config over code (no model or layout named in code), one resolver, strict validation.

## Acceptance Criteria

Every shipped variant is one native YAML; variant changes explicit and checked, native/BC fields unique, no legacy catalogue or override path. Semantic equivalence across all 16 configs, explain provenance, named cache compatibility and fresh admission; run evidence not executable. RSC11/RSP01 installed gate.

## Notes

Consumed by RSC02 (`bc-serve serve|start CONFIG`); RSC02's engine-only lifecycle can start on today's layouts, so
neither blocks the other. Independent of the manifest contract: configs are inputs to the resolver; nothing here
activates v2 ahead of RSC11 S13.

First PR (card-free): `config_loader` + variants + adapter + one dense example and its variant
(`qwen27b-tp3-int8`, `-556`) + the projection parity test against the layout. No start/stop, no stored manifests, no
caller migration, no layout deletion.

Owner direction 2026-11-14: one language for the start path (C++, radiance's standard), clean readable configs,
variants as named configs; GPT as backup reviewer and helper. GPT review 2026-11-14 (req_rsc02-config-launcher_957d54878bcc):
three P0 (owner process lifetime, the two plan keys, digest-equality gate) folded in above and in RSC02.

Step 4 (2026-11-14): the other 12 layouts converted (`configs/flashnext.toml`: 9, with `flashnext-2xtx` the base of the
R9700-engine helper configs and `flashnext-tp2xtx-r9700` of `flashnext-tp2xtx`; `configs/qwen27b.toml`: `qwen27b-tp3`,
`-capacity` as its variant, `qwen27b-2xtx`). `serving_test` `converted_layout_parity`: each config and its layout,
started as the launcher starts it (`launch-flashnext-best.sh` defaults and server-args, `plan-check.sh` engine-only
values, the derived pool and KV tier given to both), resolve to the same values and build the same manifest spec, and
both manifests pass the strict reader. Helper configs carry the launcher's 1 s histogram period (the resolver default is
60 s) and, with TP ranks, `tuning.p2p = "off"`; the two pinned TP pools stay, marked interim (RSC03).

**2026-11-14 candidate variants of `qwen27b-tp3-int8-556-long` (pending VTP36 step 9 / VTP39).** A lone-prefill wide-step variant (checkpoint interval, step budget, prefill span) and a decode-share variant (prefill interleave). Each must be expressible with the resolver's own keys; `prefill_span`, `prefill_interleave` and `busy_prefill_chunk` are not yet `engine.*` config fields (only `max_num_batched_tokens` and `checkpoint_interval` are), so adding them is part of taking either variant. No `--set` in a served start.

Target-version coordination (RV4225, November 18): RUM04 owns migration to upstream 2.x native YAML plus typed x-bigcherry intent. Preserve current-pin TOML implementation history and RSC12 semantic projection/cache/live-admission guarantees; do not add a second resolver or parallel authoritative config path. All migrated internal callers switch together after RUM01 design approval. Existing item acceptance/state is unchanged.

Warm-start implementation packet: docs/reviews/astra-performance-2026-11-18/IMPLEMENTATION-HANDOFF.md. Separate stable selected topology scope from fresh room checks. Keep streaming hashing and operation-scoped reuse in separate slices, with a generation mutation refusal test. Review the stable projection and digest version before changing the reader/writer.

2026-11-25 power recovery: the historical TOML model/variant design above is superseded by RUM04 and docs/design/radiance-native-config-x-bigcherry.md. Keep one native YAML per deployment; machine facts/power/calibration stay in existing site/profile authority. PWR08 owns power resolution and effective cache identity, RSC02 native lifecycle. Do not add a new authored manifest, model catalogue or native engine option mirror.

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. Historical TOML example and stored-manifest replay design are displaced by RUM04 and S18; do not restore. Keep unique semantic projection, static key, fresh-live admission and run receipt tests from original plan. RUM12 fast named measurement reuse preserved.

## Status 2026-11-19

Steps 1-5 are in production: `engines/radiance/configs/{flashnext,qwen27b}.toml` with named variants, strict loader, `bc-serve explain`, and stored manifests (a start of an unchanged config replays the stored manifest; policy-on replay fixed in #1177). Step 6 (retire `server-args` and `[layout.NAME.settings]`, the layout files) depends on RSC04's leftovers. Open review RV4230 (reuse the verified Radiance bundle identity within a named start) stays open. Note for operators: the replay key hashes every operator `--set`, so any differing `--set` plans afresh instead of replaying.

## Change Log

- 2026-11-13T22:56:54.680645+00:00 (created-by): Created by agent
- 2026-11-14T06:59:38.869898+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4225
- 2026-11-18T01:41:05.379942+00:00 (updated-by): Updated: section:notes
- RV4230
- 2026-11-18T03:40:48.591484+00:00 (updated-by): Updated: section:notes
- 2026-11-19T07:37:15.960699+00:00 (updated-by): Updated: section:notes
- 2026-11-19T07:37:55.296359+00:00 (state-transition): State: pending → in_progress
- RV4587
- 2026-11-25T01:09:27.855214+00:00 (updated-by): Updated: section:notes
- RV4594
- 2026-11-25T01:38:33.802309+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:detailed_solution, section:notes, section:acceptance_criteria
- 2026-11-25T01:44:11.707235+00:00 (updated-by): Updated: section:validation
