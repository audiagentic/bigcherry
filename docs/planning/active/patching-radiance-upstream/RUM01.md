---
id: RUM01
order: 1
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:27:53.871983+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: M
---

# Approve target identities and the minimal Radiance 2.x migration architecture

## Description

Decide the exact target commit, native YAML ownership, generated runtime-contract boundary, small extension interfaces and single-model-first scope. Link RSC12/RSC13, RCS01/RCS08/RCS14, RHP04/RHP09/RHP13 and PRQ qualification owners. No new allocator or generic plugin framework.

## Steps

- Freeze baseline 19d4e8fe and reviewed target b8bca7885a5a118bfa65a62dde74e0a6e2d4e42e. If target moves, regenerate applicability and inspect only the additional delta before coding.
- Approve the YAML/native-versus-extension ownership table and the single-model minimum scope. Record target contract and ABI fingerprint implications.
- Use RSC13's callable inventory and RCS01's component boundaries to identify the few lifecycle/build hooks; do not invent a generic extension framework.
- Choose one representative currently deployed Flash-Next layout and one helper-off control; record explicit scheduler/speculation defaults and source/build/model identities.
- Record retire/reuse/port disposition for the 42 local patches. Retire only after equivalent behavior is demonstrated; treat zero anchor failures as a port lead, never acceptance.

## Detailed Solution & Technical Design

Keep the bump focused on one working production configuration. Upstream main.cpp now parses FleetConfig, Registry storage is set_->..., RadBuildCtx gains slot_base and the plugin ABI is 41. These are design inputs, not defects to mask with more regex replacements.

Native options stay owned by upstream ConfigFlag/config_parse. BigCherry owns helper intent, calibrated placement, measured admission and receipts. Reuse the existing generated contract registry and strict v2 reader where semantics remain true. A version change for incompatible fields must migrate every internal caller together; no v1/v2/v3 runtime fallback.

Scope the initial extension to one engine plus its helper processes. Multi-engine helper support (RUM10), new optimization algorithms, new model families and exhaustive quantization tuning are nonblocking later work. RUM05 integrates existing placement policy rather than building a solver.

Change boundaries: extraction-only PR; dependency/ABI PR; YAML and launcher PR; placement/lifecycle PRs; KV and scheduling ports; final acceptance/pin PR. Adjacent slices may combine only when they cannot compile independently, with the coupling recorded.

## Code Samples & Guidance

Proposed decision checklist, not an executable API:

- native config authority: upstream fleet/config.cpp Reader -> config_parse
- BigCherry runtime authority: generated typed manifest + v2::read_manifest
- model identity: slot_base/rad_build_slot/rad_slot
- collective identity: tp_group
- ownership identity: immutable placement generation + owner-map digest
- admission identity: build/model/topology/layout/native-config projection
- initial supported mode: one explicitly selected production configuration

Do not equate a source commit, ABI layout fingerprint, contract digest and live receipt; each identifies a different boundary.

## Files

- docs/reviews/radiance-upstream-2026-11-18/README.md
- docs/reviews/radiance-upstream-2026-11-18/MANIFEST-AND-EXTENSIONS.md
- engines/radiance/recipes.toml
- engines/radiance/engine.toml

## Validation

Planning only: read target symbols and resolve dependency order. For implementation approval, repeat canonical patch composition once on base and target and review changed failures; do not run benchmarks for this design gate.

## Effort & Risk

M: the main risk is expanding scope beyond the first deployed configuration. Exit this design gate with explicit exclusions and owners; do not delay the bump for optional fleet/optimizer work.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

A short decision record identifies approved native/extension authority, exact target, hook owners, per-patch disposition and one minimal acceptance layout. RUM02-RUM09 have executable entry/exit conditions; RUM10 is explicitly optional after the bump.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: none. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites: none. RUM02/RUM03 follow this gate; RUM04 then feeds RUM05/RUM06; RUM07/RUM08 precede RUM09. Current-pin RSC/RCS/RHP work stays owned by those groups. No implementation or deep tests in this plan-detail task.

## Change Log

- 2026-11-18T01:27:53.871983+00:00 (created-by): Created by codex
- 2026-11-18T01:33:53.788616+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:42:41.477373+00:00 (updated-by): Updated: section:effort_risk
