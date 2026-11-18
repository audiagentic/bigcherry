---
id: RUM04
order: 4
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:13.721371+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: L
---

# Reuse native fleet YAML and compile typed BigCherry intent into an immutable runtime bundle

## Description

Implement native settings through upstream YAML validation with closed top-level x-bigcherry intent. Keep generated per-model contracts, digest/provenance and live admission. Migrate current configuration callers together; coordinate RSC12 rather than create a competing resolver.

## Steps

- Inventory current TOML/native-setting duplication in config_loader.{h,cpp}, serving.cpp, manifest_writer.cpp, session.cpp and rad0009. Use RSC12 as current input owner; implement target YAML migration here without creating another resolver.
- Factor upstream fleet/config.cpp parse ownership just enough to let native Reader and BigCherry extension decoder inspect the same resolved location-bearing ryml tree. Never return borrowed nodes after the parser/tree is destroyed.
- Decode a closed versioned top-level x-bigcherry object keyed by native model ID. Native options are read by upstream ConfigFlag/config_parse; extension intent contains only helper/layout/power/calibration policy references and necessary role overrides.
- Compile effective native Config and extension intent into one normalized per-model projection, then feed existing v2/S8/S10 measured resolution/admission. Keep generated runtime JSON immutable; no independently authored YAML/JSON policy copies.
- Replace bc_serving_engine_argv and g_emitted startup relabeling with an admitted Config/FleetConfig adapter. Update session launch, native metadata, smoke/fake-engine fixtures and all internal config callers together.
- Bind native semantic projection plus extension/owner generation into stored-plan key and managed reload identity. Raw file paths, comments and key order are provenance, not execution equality.
- For first cutover permit one remote-MoE engine. Make stock binary capability/refusal and extension-required checks happen before helpers allocate. Port stored-manifest exact replay and measuring-versus-serving behavior without quietly reverting to mutable authored YAML.
- Remove obsolete native-option alias tables, --set/argv conversions and replaced TOML/layout entry paths once every caller and test is migrated. Keep only an explicit offline conversion tool if a named external need is approved.

## Detailed Solution & Technical Design

Upstream already provides fleet::parse(text,origin,out), validates duplicate/root/model-level keys, resolves anchors/merge keys and feeds native keys into config_parse. x- keys are ignored; arbitrary model extension keys are not accepted. Adding x-bigcherry text alone executes nothing.

Use a minimal YAML document owner whose lifetime encloses both Reader::run and extension decoding; expose one shared tree view rather than a generic parser plugin system. Choose signatures in RUM01, then migrate parse/load call sites together. Reuse upstream defaults/alias behavior and source locations. Limit supported merge/anchor semantics to what the native parser actually resolves and validate extension fields after resolution.

Current v2::read_manifest already performs identity -> digest -> structural decode -> provenance -> semantics and distinguishes PlanUse::measuring/serving. Keep that path for helper/admission data; do not drop it in favor of untyped YAML. S8/S10 continue owning resource arithmetic and power qualification. Do not invent a second JSON parser in this slice. Independently pinned nlohmann from RUM03 keeps this migration small.

Bundle design starts small: immutable native normalized projection plus per-model strict contract, bound in one index/receipt. Leaf artifacts do not hash the index, avoiding circular digests. Existing plan_cache and stored_manifest_cache should own cache/replay; extend their key_inputs once rather than add a parallel bundle cache. Runtime instance ID and live room/power readings are not static plan identity.

Upstream Fleet::reload compares entry.args, so extension changes would be missed unless explicitly incorporated. For the first single-model managed deployment, reject direct authored-file reload that bypasses resolution/admission; use existing supervisor drain-and-restart of a newly admitted snapshot rather than implementing full transactional fleet reload now. RUM10 can later integrate concurrent fleet reload.

## Code Samples & Guidance

Proposed parse-once orchestration, not current upstream API:
```cpp
auto doc = parse_yaml_document(bytes, origin); // owns parser, tree and locations
FleetConfig native = read_native_fleet(doc);  // same upstream Reader validation
BcIntent intent = read_bigcherry_intent(doc);  // closed x-bigcherry schema
ResolvedStart start = resolve_existing_plan(native, intent, measured_inputs);
validate_runtime_bundle(start);              // existing strict contract + live admission
```

Native/extension mapping rule:
```text
models.NAME.tp / kv-cache-dtype / max-model-len -> upstream native ownership
x-bigcherry.models.NAME.helper-profile/layout -> BigCherry intent only
plan bytes / owner maps / measured memory -> generated, never authored
```
All names above are design sketches; implement the smallest typed adapter through actual current libraries.

### Deep scan implementation refinement

Correct earlier broad consolidation premise: rad0009 already iterates generated v2::kFields; reuse serving-v2.toml/contract/generate.cpp metadata rather than invent a second mapping registry. stored_manifest_cache.cpp already shares writer/replay static projection. Preserve distinct plan_cache.cpp::plan_key_inputs measured-memory and static-resolution identities; share typed effective inputs, not one blanket digest. Remove obsolete g_emitted/argv adapter only after all callers migrate.

Source evidence and removal conditions: docs/reviews/radiance-upstream-2026-11-18/CONSOLIDATION.md. Use patch-footprint.json to inspect every overlapping patch before changing an attachment site.

## Files

- engines/radiance/components/serving/config_loader.h
- engines/radiance/components/serving/config_loader.cpp
- engines/radiance/components/serving/session.cpp
- engines/radiance/components/serving/stored_manifest.cpp
- engines/radiance/components/serving/stored_manifest_cache.cpp
- engines/radiance/components/serving/plan_cache.cpp
- engines/radiance/components/serving/manifest_writer.cpp
- engines/radiance/components/serving/v2/manifest.cpp
- engines/radiance/components/serving/v2/s10_resolver.cpp
- engines/radiance/patches/rad0009_serving_manifest/patch.py
- engines/radiance/engine.toml

## Validation

Use existing typed-settings, stored-manifest and session fixtures with a small case set: native-only parity; extension decode plus exact model reference; duplicate/unknown/misplaced key; digest/build/topology mismatch; fresh live-room refusal on cached start; helper/engine same generation; managed reload refusal or drain/re-admit. Parameterize close variants instead of writing parallel test engines. One one-model CPU/fake launch is enough before RUM09 hardware canary.

## Effort & Risk

L: duplicated setting authority, dangling YAML node views and stale cached/reloaded plans are the main risks. One parser lifetime, one semantic projection and existing strict admission reduce both code and test scope.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

One authored YAML file owns native and typed extension intent without duplicated engine options. Both decode paths share resolved tree lifetime. Engine/helpers consume the admitted snapshot and exact replay remains strict. No token-path config parsing; no legacy argv bridge remains reachable after target cutover.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM01, RUM02, RUM03. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM01/RUM02/RUM03. Coordinate RSC12 current-config ownership and RSC07/RSC11 measured-plan semantics; do not rewrite their current-pin implementation history. Initial cutover does not implement arbitrary multi-model helper reload.

## Change Log

- 2026-11-18T01:28:13.721371+00:00 (created-by): Created by codex
- 2026-11-18T01:37:35.422476+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:04.055379+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:56:44.422220+00:00 (updated-by): Updated: section:code_samples
