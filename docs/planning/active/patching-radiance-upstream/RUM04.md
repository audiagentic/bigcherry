---
id: RUM04
order: 3
plan: patching-radiance-upstream
state: in_progress
created-at: '2026-11-18T01:28:13.721371+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: L
---

# Use original native YAML and remove duplicated engine configuration contracts

## Description

Replace the local native-setting mirror and argv bridge with upstream YAML/Config behavior. Bind immutable original YAML to the existing measured BigCherry contract, retaining only helper/local/derived guarantees that upstream does not supply.

## Steps

- Classify contract radiance_option bindings and actual readers; separate native sampling/chat/limits from local-patched flags and resource-derived values.
- Use upstream Reader::build_model/config_parse to obtain native Config and provenance from the original admitted YAML bytes. Use existing BC profile selection first; add only a narrow x-bigcherry selector if needed, not a new schema for native settings.
- Bind exact YAML bytes/digest and engine/build identity to the existing measured contract; never reopen a mutable authored path during exact replay.
- Migrate native consumers in main.cpp/session.cpp and model/home reads in v2/p1_producer.cpp to native Config/build identity. Remove the full local Resolved::typed Engine mirror after those production callers migrate.
- Remove native-only engine_of/leaf_of_key/engine_option_fields mappings and corresponding contract fields through the existing registry/code generator. Preserve local/derived fields until their consumers have equivalent typed inputs.
- Delete bc_serving_engine_argv, bc_serving_mark_given, flag_text and g_store/g_ptrs/g_emitted/g_label after daemon, measuring, stored-replay and offline callers use the native path.
- Preserve helper inheritance and exact rank order separately from native defaults/YAML anchors. Preserve fresh resource/power admission on every start.
- For the initial managed path refuse unqualified authored-file hot reload; drain and restart from a newly admitted immutable snapshot using the existing supervisor.

Implementation boundary: store exact original UTF-8 YAML in the existing strict measured session manifest, with its byte digest and existing build admission. The linked implementation handoff fixes the consumer, cache and replay migration order. Migrate in one call chain at a time and remove the native mapping only after all real callers use the sealed native input.

## Detailed Solution & Technical Design

Native defaults, validation and YAML provenance already exist upstream. Do not duplicate them in a normalized projection, a generated YAML writer, a serialized FleetConfig or another binding table. Reparse the same immutable bytes in each actual consumer through fleet::parse; there is no need to expose private ryml tree ownership to BigCherry.

Remove only proven native duplication. The current generated contract also describes derived engine memory, patched per-rank/plugin settings, helper execution, owner maps, power and provenance; those remain necessary. Remove fields and every internal reader together with an explicit contract/fingerprint migration, never a compatibility fallback.

stored_manifest_cache already has one writer/replay projection; measured plan_cache has different exclusions. Extend existing identity inputs with original YAML bytes/hash and build identity rather than unifying these caches. Conservative formatting-only invalidation is simpler and safe for this bump. Secrets remain provider/environment references; do not persist resolved api-key values as YAML snapshots.

The measured runtime contract may persist the YAML bytes or bind an immutable recorded snapshot using its existing storage mechanism; choose the smaller existing mechanism. Do not add an independently managed bundle index or artifact service.

## Code Samples & Guidance

Deletion candidates after caller migration: rad0009 bc_serving_engine_argv, bc_serving_mark_given, flag_text, g_store/g_ptrs/g_emitted/g_label; native-only branches of manifest_writer::engine_of/leaf_of_key and serving::engine_option_fields; full Resolved::typed Engine mirror when model/home production reads move to native Config/build identity.

Keep: strict digest/identity, owner-map verification, measured host/VRAM/pin/power admission and helper settings.

Test one native YAML option's value/provenance against upstream, exact snapshot replay, changed-build invalidation, fresh-live-room refusal and offline cfg.devices mapping. These prove the replacement without generating a second Config schema.

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

Reuse existing session, stored-manifest and typed-setting fixtures. One native option absent from the BC registry must pass through the actual named launcher; wrong-level native key fails. One retained helper setting binds correctly. Normal/measure/KL/stored starts consume identical admitted bytes despite author-file change, or refuse changed digest before child/power mutation. Check stale build/topology and fresh-room refusal; managed reload refuses. One fake launch before the RUM09 canary.

## Effort & Risk

L: duplicate authority and mutable config inputs can invalidate admission. Reuse native parsing on identical immutable bytes with matching build/default/environment identity; retain existing measured contract and fresh admission.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

Native settings have one upstream parser/default/provenance authority and no duplicate BigCherry mirror. Exact replay uses admitted immutable bytes with build identity; all helper/resource/power guarantees remain. No new normalized writer, serializer, generic registry or bundle layer is required.

## Notes

Prerequisites RUM01/RUM03; this slice precedes RUM02. Coordinate RSC12 authoring and RSC07/RSC11 measured-plan ownership. ASTRA-CONSOLIDATION.md supersedes earlier normalized bundle and parser-tree proposals. Keep current-pin history; managed single-model reload is refused at the common operation and requires reprepare/restart.

Executable guidance: docs/reviews/radiance-upstream-2026-11-18/IMPLEMENTATION-HANDOFF.md. This storage choice supersedes the earlier alternative snapshot storage sketch.

First prove YAML transport and entry points using a supported fixture. The v1.17 Flash-Next `fp8` spelling is not equivalent to target native `fp8`; use RUM07 disposition before starting the real production-format lane. Target host-pool capacity is aggregate before division by TP; preserve actual admitted bytes and validate this in a multi-rank negative test.

C3 exact native parser fixture and C4 managed bootstrap marker recorded under docs/evidence/radiance-latest-bump. Full persisted successor contract and native exec remains open.

C11 (2026-11-19): extracted NativeConfigSnapshot into the existing strict/GPU-free manifest library and made ABI43 NativeFleetConfig consume the validated snapshot before fleet::parse. The standalone library configured/built against the candidate SDK and focused manifest_snapshot_roundtrip CTest passed 1/1, covering exact CRLF/UTF-8 bytes, author-file removal replay, cfg.given, digest/root and malformed-wire refusal. See evidence C11 and Ledger event d77203d03d59e805ac04c299da2a12972d67be682327b88e56dfa22a3645cf47. Remaining RUM04: generated successor wire identity, strict writer/reader role views, measured plan and stored cache identity, actual main/session/offline/KL callers, sealed config descriptor for fresh and stored exec, root/device-order propagation, reload refusal and native mirror deletion. This seam alone does not change production behavior.

C12 (2026-11-19, Sol): the explicit generated v3 edition now carries required native_config bytes/digest/root through shared writer and reader validation; measured-plan identity and role views bind snapshot identity, with exact bytes only in engine view. The card-free test verifies measured write/read, old-wire refusal, role views and plan invalidation. Existing v2 reader passes and all seven regenerated v2 artifacts are byte-identical. Production still selects v2; named fresh/measure/stored launch through matched ABI43 parser and existing sealed descriptor is still open. Full CTest output remains in task-local build scratch; Ledger event 6ae8508eaead1f703b22358b34d55d84991366f6250726ff9d9fef469abbf5f8.

## Change Log

- 2026-11-18T01:28:13.721371+00:00 (created-by): Created by codex
- 2026-11-18T01:37:35.422476+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:04.055379+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:56:44.422220+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:16:28.315942+00:00 (updated-by): Updated: section:title, order=3, section:description, section:steps, section:detailed_solution, section:code_samples, section:acceptance_criteria, section:notes
- 2026-11-18T02:22:48.978686+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:25:52.100615+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:28:11.086931+00:00 (updated-by): Updated: section:validation, section:effort_risk, section:notes
- 2026-11-18T03:39:58.662306+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-11-18T03:41:39.090024+00:00 (updated-by): Updated: section:notes
- 2026-11-19T00:54:27.126833+00:00 (updated-by): Updated: section:notes
- 2026-11-19T00:54:34.826531+00:00 (state-transition): State: pending → in_progress
- 2026-11-19T02:54:58.912371+00:00 (updated-by): Updated: section:notes
- 2026-11-19T03:06:59.128634+00:00 (updated-by): Updated: section:notes
