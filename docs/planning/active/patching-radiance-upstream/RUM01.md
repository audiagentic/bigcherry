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

# Approve an upstream-native migration with explicit deletion boundaries

## Description

Decide the exact target commit, native YAML ownership, generated runtime-contract boundary, small extension interfaces and single-model-first scope. Link RSC12/RSC13, RCS01/RCS08/RCS14, RHP04/RHP09/RHP13 and PRQ qualification owners. No new allocator or generic plugin framework.

## Steps

- Freeze target/build identities and the one production configuration.
- Adopt Astra's native-YAML ownership decision: bind original immutable YAML bytes plus engine/build identity to the existing measured BigCherry contract; do not create a normalized writer, FleetConfig serializer, bundle index or generic extension registry.
- Classify every current native binding as upstream-native, local-patched-native, derived admission, helper policy or diagnostic. Remove only the native-only mirror and migrate its actual callers.
- Implement dependency/ABI foundation RUM03, then native-boundary/contract narrowing RUM04, then extract surviving code RUM02. Avoid packaging code whose purpose will be deleted.
- Validate explicit engine-card order in the actual child-local namespace and propagate the same selected devices to offline measuring/KL mode.
- Use one model with disjoint engine/helper cards initially; broker/shared-card features stay in RUM10.

## Detailed Solution & Technical Design

Astra's source review supersedes the normalized-config/bundle-first proposal. Upstream Reader::build_model sorts effective native options, calls config_parse, sets Config.given provenance and retains ModelEntry.args/cfg. These are the reusable foundations; a second native schema/resolver adds maintenance rather than removing it.

Bind one immutable original YAML snapshot to the existing strict measured BigCherry contract and engine identity. Native consumers parse those same bytes through fleet::parse; parsing once per startup process is acceptable and needs no shared ryml tree API. Conservative cache misses from comment/format changes are acceptable for the initial bump; semantic-key normalization is optional later work, not a new serializer now.

For one model upstream use_broker() is false, choose_cards takes the first tp cards, and run preserves explicit list order. Keep existing per-BDF engine/helper/host admission, require disjoint assignments and refuse drift. Do not require Fleet::grant hooks that cannot account for helpers on this path. Offline main directly creates Engine and must receive the same chosen cfg.devices explicitly.

Retain measured resource facts, owner-map verification, helper configuration, power policy and exact per-rank constraints. Current contract contains 113 radiance_option bindings, including local/derived fields: classify them rather than claiming all are removable. Full fleet external-reservation semantics are deferred.

## Code Samples & Guidance

Native boundary to reuse: upstream fleet/config.cpp Reader::build_model -> config_parse -> ModelEntry.cfg/args/given.

Proposed minimal persisted relationship, not an implemented schema:
```text
admitted start = original YAML bytes + content hash
              + exact engine/build/default identity
              + measured BigCherry-specific helper/resource/power contract
```
No normalized YAML writer, Config serialization, new bundle-index service or native-field mapping registry. Source evidence: docs/reviews/radiance-upstream-2026-11-18/ASTRA-CONSOLIDATION.md.

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

Decision record names concrete native mirrors/bridges to remove, survivor code to extract, required retained admission guarantees, exact child-device mapping and reordered stages. No new serialization/broker framework is introduced for the first bump.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: none. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites: none. RUM02/RUM03 follow this gate; RUM04 then feeds RUM05/RUM06; RUM07/RUM08 precede RUM09. Current-pin RSC/RCS/RHP work stays owned by those groups. No implementation or deep tests in this plan-detail task.

Execution decisions: use the named TP1 Flash-Next production lane for first native canary, and separately use helper-off for functional/KL control. This does not qualify mixed engine ranks. Follow docs/reviews/radiance-upstream-2026-11-18/IMPLEMENTATION-HANDOFF.md for frozen names, inputs and ownership inventory; record installed artifact identities before beginning gates.

## Change Log

- 2026-11-18T01:27:53.871983+00:00 (created-by): Created by codex
- 2026-11-18T01:33:53.788616+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:42:41.477373+00:00 (updated-by): Updated: section:effort_risk

## Reviews

- RV4227
- 2026-11-18T01:56:23.440036+00:00 (updated-by): Updated: section:code_samples
- RV4228
- 2026-11-18T02:16:19.600984+00:00 (updated-by): Updated: section:title, section:steps, section:detailed_solution, section:code_samples, section:acceptance_criteria
- RV4232
- 2026-11-18T03:39:43.858967+00:00 (updated-by): Updated: section:notes
