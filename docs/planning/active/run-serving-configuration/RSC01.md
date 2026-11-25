---
id: RSC01
order: 1
plan: run-serving-configuration
state: pending
created-at: '2026-11-02T00:52:58.694609+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P0
work: M
---

# One native YAML serving configuration, strict BigCherry extension and explain

## Description

Existing C++ resolver and native YAML inspector are retained. Radiance owns every native option and validation; BigCherry owns one closed x-bigcherry extension plus site/profile YAML and derived typed facts. Remove duplicate authored serving TOML/flags/env and full launch-manifest schema dependency. Facts/constraints are distinct from preferences, calibration fills only auto values, no hand-picked resource constants. Evidence output is separate.

## Steps

Inventory each existing field/native option and actual consumer; classify as native YAML, BC extension YAML, site/profile YAML, derived runtime fact or evidence-only. Reuse native fleet::parse and existing rapidyaml x-bigcherry decoder; share one strict BC reader with worker and owner. Migrate authored site/profile/helper settings to YAML, update all active consumers at once; general build metadata and measured evidence stay in their owning formats. Separate typed resource/power/transport settings from manifest serialization; keep one C++ resolver/explain, Python thin diagnostics/lab clients only. Verify native upstream-only option passes through and duplicate native keys are refused in extension.

## Detailed Solution & Technical Design

docs/design/radiance-native-yaml-launch.md sections 2–4. No generated effective YAML/JSON launch envelope, no second native parser, no synthetic native flag list. Preserve provenance and schema/strictness of actual BC options; output-only capture can reuse existing codec isolated from launch. Matched parser inspector output is a narrow fact report, not serialized Config. Profile/site decoder one authority, no TOML compatibility path.

## Code Samples & Guidance



## Files



## Validation

Strict YAML duplicate/unknown/type/range/units, wrong level, hard-cap and missing required refusal, native provenance, variant equivalence, authored-byte change/root, card-free real explain, release Python-free, active caller inventory and absence of launch reader. Semantic projection against current baseline, not full capture digest equality.

## Effort & Risk



## Standards



## Acceptance Criteria

One native model YAML and BC extension/site/profile YAML authorities; no duplicated native values, no config getenv/--set or manifest launch. Preserve strict validation, provenance, hard constraint/calibration rules, existing explain and equivalent effective configuration. Python has no second serving schema/resolver. All release/campaign consumers migrated together.

## Notes

Source: GPT configuration assessment 2026-11-02 (gateway request req_main_fa1c9eed1169), reviewed. Runs in parallel with RHP stage 1; does not block it. Owner rules: [no hand-picked resource constants], no backward-compatibility shims.

2026-11-25 owner direction: no launch manifest, all authored serving configuration YAML, no compatibility shims or legacy backports. docs/design/radiance-native-yaml-launch.md supersedes earlier manifest-launch instructions; completed RUM04/05/06/09/12 evidence is historical, not proof of the new launch path. Existing capture manifests may remain output-only. No new programme or second resolver. Replaces old RSC01 TOML/schema/manifest authoring steps; original inventory and parity requirements retained.

## Change Log

- 2026-11-02T00:52:58.694609+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:53:44.556553+00:00 (updated-by): Updated: section:ledger-events

- chg_20261102_035356_serving-start-up-plan-one-str_4985

## Reviews

- RV4224
- 2026-11-02T03:51:36.726474+00:00 (updated-by): Updated: section:title, section:steps, section:detailed_solution, section:validation, section:acceptance_criteria
- 2026-11-02T03:52:49.575691+00:00 (updated-by): Updated: section:title, section:steps, section:detailed_solution, section:validation, section:acceptance_criteria
- 2026-11-02T03:54:00.264966+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_050920_new-bc-serve-tool-one-strict_1973
- 2026-11-02T05:09:24.836486+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T12:12:43.760997+00:00 (updated-by): Updated: section:ledger-events
- RV4589

## Update 2026-11-10 (GPT power manifest review, req_main_eb983900e21b)

Reconciliation: the C++ resolver, generated contract, strict reader, writer and explain path exist (serving/main.cpp, manifest_writer.cpp, contract/README.md); the item is implemented in core and open on acceptance (Python command migration, release-path equivalence, obsolete configuration surfaces). Do not read the pending state as 'no resolver': do not build a second one. The v1 contract is closed; power profiles (PWR08) and any other added field need bigcherry/v2, so batch them (RSC09 literals, RSC10).
- 2026-11-25T01:36:01.665422+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:detailed_solution, section:validation, section:acceptance_criteria, section:notes
