---
id: RUM01
order: 1
plan: patching-radiance-upstream
state: completed
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

Frozen-source correction (2026-11-19): active candidate is Radiance v2.1.0 commit 3deabdaeca17b58bad90d332f4b204cf7bb8a10e (ABI43); production remains v1.17.0. The prior source review target b8bca7885a5a118bfa65a62dde74e0a6e2d4e42e is an intermediate ABI41 candidate, and the v2.1.0 tag preflight at ABI40 is a different source. The frozen branch candidate delta from b8bca is 71 files (+7,179/-324). README and implementation handoff now identify historical source-specific claims as leads requiring candidate verification. Compatibility acceptance uses both ABI version and exported layout fingerprint. Plain Flash-Next E4M3 remains a separate RUM07/RUM09 runtime gate; no candidate production qualification is implied. The candidate baseline for matched policy-off comparison is docs/reference/engines/radiance/baselines/flashnext-4gpu-parity-main-20261119.json.

RUM01 execution decision is now tied to frozen endpoints: BigCherry baseline patch set at 1f1547e72953; Radiance production v1.17.0 source 19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d (ABI35); frozen candidate 3deabdaeca17b58bad90d332f4b204cf7bb8a10e (ABI43). Exact ordered serving-core composition dry-run used resolve_exact/load_resolved/apply_patch with shared in-memory texts: 30 packages, base 474 applied / 0 failed; candidate 334 applied, 1 not applicable, 6 already applied, 141 failed edits across rad0013 (1), rad0037 (56), rad0038 (60), rad0040 (10), rad0041 (14). See evidence checkpoint C8. This remains patch applicability only; no pin movement or runtime qualification. rad0013 retirement proof belongs RUM08; KV package rework/retirement belongs RUM07. The first lane uses the current-main policy-off baseline as an old-pin control reference only; its two launches do not qualify ABI43. Normal Fleet and offline Engine entry points require the same ordered Config.devices explicitly; parse origin becomes FleetConfig.path, so managed reload must refuse both admin and SIGHUP before reopening config. Current handoff, README authority banners and PIN_BUMP ABI/layout wording reflect these decisions.

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
- RV4302
- 2026-11-19T02:29:07.618375+00:00 (updated-by): Updated: section:notes
- 2026-11-19T02:43:15.359892+00:00 (updated-by): Updated: section:notes
- 2026-11-19T02:44:07.245775+00:00 (state-transition): State: pending → completed
