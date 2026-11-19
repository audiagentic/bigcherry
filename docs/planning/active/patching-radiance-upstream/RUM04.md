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

2026-11-19: explicit `resolution_root` is now passed into NativeConfigSnapshot and the rad_core-linked NativeFleetConfig fixture, instead of recording owner cwd implicitly. ABI43 fleet::parse uses the supplied origin only for diagnostic/config path; actual path resolution requires execution under the recorded canonical root. The fixture also sends native prefill-interleave: 0 through the parser and checks cfg.prefill_interleave_set and provenance. Wire snapshot tests reject relative, non-directory, embedded-NUL, invalid-UTF8 and mismatched roots. Build/test is pending.

C13 (2026-11-19): built and ran `bc_native_fleet_config_test` against exact pinned source 3deabdaeca17b58bad90d332f4b204cf7bb8a10e in an isolated source/build copy. CMake revealed core is added before top-level enable_testing, so add_test in core was silently unregistered; moved registration to tests/CMakeLists before final test-label pass. CTest now discovers and passes 1/1, covering ABI43 fleet::parse, explicit canonical resolution root, prefill_interleave=0, cfg.given and provenance. rad0009 package fixture suite passes 11 tests. This validates parser seam only; production launch/caches/callers and successor contract remain open.



C15 (2026-11-19): fresh measuring and serving writer outputs are now read with no-follow/bounded regular-file checks, validated for their exact role, and sealed before power admission or child launch. EngineStart keeps the durable manifest path for receipts/indexing and exposes execution_manifest() for sealed reads; measuring helper and engine children inherit the sealed descriptor, and runtime guards, calibration reads, power verification, and digest checks use the same admitted bytes. Stored replay binds its attestation to the already-read authenticated bytes. Focused serving, stored replay, native option projection, and generated v3 wire tests pass; the first serving run used a stale binary, so it was rebuilt before the passing run. Direct test execution was used because CTest's launcher is blocked by this container's unprivileged-user-namespace restriction. This is a sealed launch handoff slice; native FleetConfig/YAML wiring and remaining RUM04 consumers/mirror deletion remain open. Ledger event chg_20261119_100847_pinned-freshly-written-manifes_8772.

Prerequisites RUM01/RUM03; this slice precedes RUM02. Coordinate RSC12 authoring and RSC07/RSC11 measured-plan ownership. ASTRA-CONSOLIDATION.md supersedes earlier normalized bundle and parser-tree proposals. Keep current-pin history; managed single-model reload is refused at the common operation and requires reprepare/restart.

Executable guidance: docs/reviews/radiance-upstream-2026-11-18/IMPLEMENTATION-HANDOFF.md. This storage choice supersedes the earlier alternative snapshot storage sketch.

First prove YAML transport and entry points using a supported fixture. The v1.17 Flash-Next `fp8` spelling is not equivalent to target native `fp8`; use RUM07 disposition before starting the real production-format lane. Target host-pool capacity is aggregate before division by TP; preserve actual admitted bytes and validate this in a multi-rank negative test.

C3 exact native parser fixture and C4 managed bootstrap marker recorded under docs/evidence/radiance-latest-bump. Full persisted successor contract and native exec remains open.

C11 (2026-11-19): extracted NativeConfigSnapshot into the existing strict/GPU-free manifest library and made ABI43 NativeFleetConfig consume the validated snapshot before fleet::parse. The standalone library configured/built against the candidate SDK and focused manifest_snapshot_roundtrip CTest passed 1/1, covering exact CRLF/UTF-8 bytes, author-file removal replay, cfg.given, digest/root and malformed-wire refusal. See evidence C11 and Ledger event d77203d03d59e805ac04c299da2a12972d67be682327b88e56dfa22a3645cf47. Remaining RUM04: generated successor wire identity, strict writer/reader role views, measured plan and stored cache identity, actual main/session/offline/KL callers, sealed config descriptor for fresh and stored exec, root/device-order propagation, reload refusal and native mirror deletion. This seam alone does not change production behavior.

C12 (2026-11-19, Sol): the explicit generated v3 edition now carries required native_config bytes/digest/root through shared writer and reader validation; measured-plan identity and role views bind snapshot identity, with exact bytes only in engine view. The card-free test verifies measured write/read, old-wire refusal, role views and plan invalidation. Existing v2 reader passes and all seven regenerated v2 artifacts are byte-identical. Production still selects v2; named fresh/measure/stored launch through matched ABI43 parser and existing sealed descriptor is still open. Full CTest output remains in task-local build scratch; Ledger event 6ae8508eaead1f703b22358b34d55d84991366f6250726ff9d9fef469abbf5f8.

2026-11-19: explicit `resolution_root` is now passed into NativeConfigSnapshot and the rad_core-linked NativeFleetConfig fixture, instead of recording owner cwd implicitly. ABI43 fleet::parse uses the supplied origin only for diagnostic/config path; actual path resolution requires execution under the recorded canonical root. The fixture also sends native prefill-interleave: 0 through the parser and checks cfg.prefill_interleave_set and provenance. Wire snapshot tests reject relative, non-directory, embedded-NUL, invalid-UTF8 and mismatched roots. The parser fixture and patch-package checks passed as recorded in C13.

C13 (2026-11-19): built and ran `bc_native_fleet_config_test` against exact pinned source 3deabdaeca17b58bad90d332f4b204cf7bb8a10e in an isolated source/build copy. CMake revealed core is added before top-level enable_testing, so add_test in core was silently unregistered; moved registration to tests/CMakeLists before final test-label pass. CTest now discovers and passes 1/1, covering ABI43 fleet::parse, explicit canonical resolution root, prefill_interleave=0, cfg.given and provenance. rad0009 package fixture suite passes 11 tests. This validates parser seam only; production launch/caches/callers and successor contract remain open.



C15 (2026-11-19): fresh measuring and serving writer outputs are now read with no-follow/bounded regular-file checks, validated for their exact role, and sealed before power admission or child launch. EngineStart keeps the durable manifest path for receipts/indexing and exposes execution_manifest() for sealed reads; measuring helper and engine children inherit the sealed descriptor, and runtime guards, calibration reads, power verification, and digest checks use the same admitted bytes. Stored replay binds its attestation to the already-read authenticated bytes. Focused serving, stored replay, native option projection, and generated v3 wire tests pass; the first serving run used a stale binary, so it was rebuilt before the passing run. Direct test execution was used because CTest's launcher is blocked by this container's unprivileged-user-namespace restriction. This is a sealed launch handoff slice; native FleetConfig/YAML wiring and remaining RUM04 consumers/mirror deletion remain open. Ledger event chg_20261119_100847_pinned-freshly-written-manifes_8772.

C24 (2026-11-19): merged current origin/main at 5e814a1d8; merge base matches origin/main. Rebuilt and ran bc_session_guards_test, including proof_lag; passed. Evidence: docs/evidence/radiance-latest-bump/README.md C24; Ledger event chg_20261119_115029_integrated-latest-main-into-th_1067.

C25 (2026-11-19): measuring preflight now reads host pool/KV/reserve/TP from the exact strict parsed measuring manifest and uses freshly detected host room. A session regression drives exit-11 to raise pool to 2048 MiB/rank, forces a second cache miss, and verifies TP3 refusal before a second measuring child. bc_resources_parity_test and bc_serving_test passed. Evidence: docs/evidence/radiance-latest-bump/README.md C25; Ledger event chg_20261119_120029_fixed-measuring-start-memory-a_1732.

C26 (2026-11-19): removed worker_policy_config.h's hard-coded generated/v2 header include and rebuilt/reran r3_policy_config_test successfully with the current v2 target selected. This is an edition-neutral include seam, not a full v3 worker build. Evidence: docs/evidence/radiance-latest-bump/README.md C26; Ledger event chg_20261119_120946_removed-the-helper-workers-ha_3343.



C27 (2026-11-19): shared writer requires NativeConfigSnapshot for generated v3 and rejects it for v2; v3 records exact bytes/digest/root/provenance before strict decode. An isolated real-writer test removes the author file and covers measuring/serving outputs plus missing-snapshot refusal; focused CTest passed 1/1 and direct rerun exited 0. Production still selects v2; native consumer and launch integration remain open. Evidence C27; Ledger event `chg_20261119_122249_added-a-verified-successor-man_4213`.


C30 (2026-11-20): added the matched-engine read-only native YAML inspect seam. rad0009 now compiles `native_fleet_config.cpp` once into `rad_core`, propagates its header to `radiance`, and handles `--bc-inspect-native-config FD` before manifest argv rewriting. The handler requires a bounded regular file with all four memfd seals, strict-decodes the existing three-field snapshot payload, checks its digest and execution root through NativeConfigSnapshot, and runs the exact upstream fleet parser plus `fleet::unsupported`. The closed report emits the snapshot identity and admission inputs without raw YAML or credentials, with provenance for runtime-home/kernel values. Focused rad0009 tests passed 5/5; both C++ units compiled with warnings-as-errors against pinned ABI44 headers; exact pinned-source apply and replay were byte-identical. A fresh engine link/runtime invocation is still pending, as are session descriptor inheritance and consumer migration. Evidence: C30 in docs/evidence/radiance-latest-bump/README.md.



C31 (2026-11-20): added `session::inspect_native_config` using the existing captured-output planner child path. The typed child role is allowed before power admission, and stays registered/reaped through the owner; it receives the sealed snapshot memfd, uses a root directory fd plus child-local `fchdir`, and executes the absolute matched engine path. Both captured streams are bounded; the owner strict-decodes the closed report and checks protocol, snapshot digest/root, and typed admission fields. Card-free `bc_session_test` positive coverage proves exact CRLF/UTF-8 bytes after authored-file replacement, all seals, recorded-root cwd, unrelated-FD close-on-exec, parent/owner cwd preservation, journal/reap, and inspector-before-power ordering. Negative coverage returns a mismatched digest and proves refusal before power admission or serving-engine start, with inspector reaped. The focused build and CTest passed 1/1. This completes only the session-owner inspector boundary; production planning still selects the TOML/v2 path, so no YAML consumer/cache/manifest migration is claimed. Evidence C31 in `docs/evidence/radiance-latest-bump/README.md`; actual matched-engine invocation and GPU/runtime gates remain open.

C31 clarification: `inspect_native_config` takes an explicit native-environment overlay so the production planner can use the exact admitted environment-backed values for parsing and subsequent launch; it does not silently derive a second environment. The test uses an empty overlay.



C32 (2026-11-20, Sol/GPT review follow-up): the owner inspector now returns a closed `NativeConfigInspection` value, not raw JSON. It rejects unknown/missing keys, wrong array item types, invalid card ordinals, ABI mismatch versus compiled Radiance ABI, nonpositive TP/sequences, invalid listener ports, negative context, and empty model/listener. The positive fake-engine session test proves explicit native environment overrides supersede deliberately wrong inherited runtime-home and kernel environment values; negative tests require the actual digest refusal and malformed-card diagnostic before power admission/engine start and verify child reap. `bc_session_test` passes. Next RUM04 production slice: capture a single effective sanitized environment and reuse it for inspection and engine execution; add a BC-only native named-profile/YAML input with canonical root only after verified executable selection; reconcile the native model, runtime location, listener, TP, card ordinals and rank order to local BDF admission; carry the same snapshot through writer/P1/measure/replay and both cache identities. Do not advertise public YAML launch yet: current main still builds the TOML/v2 plan and v2 writer rejects native snapshot fields. The matched engine CLI probe remains unverified.



C33 (2026-11-20): the rad0009 inspection hook is now freshly linked into the ABI44 candidate `radiance` executable and exercised through the actual `main.cpp` command boundary. A sealed native snapshot produced a closed report with exact identity, ABI 44, parsed model/listener/cards and native TP default; unsealed and wrong-digest descriptor probes both exited 2 with no stdout. The temporary source/build remains task-local and was not installed. This closes only the matched parser-command probe; the production bc-serve planner is still TOML/v2 and does not select a YAML snapshot, pass one effective environment from inspection into execution, or bind native bytes into writer/P1/measurement/replay/cache. No GPU/runtime launch was made.



C34 (2026-11-20): native snapshot SHA-256 and canonical resolution root now enter the operational measured-plan key when a NativeConfigSnapshot is provided (cache version 3); existing non-native version-2 keys remain unchanged. The real v3 writer refuses measured finalization unless that key carries the exact snapshot identity it persists. `bc_serving_test` and `bc_serving_v3_writer_test` both passed (1/1 each). Evidence C34 in `docs/evidence/radiance-latest-bump/README.md`; Ledger event `chg_20261119_143902_bound-successor-measured-plan_8730`. Still open: owner-attest the measured plan source rather than trusting the CLI `--measured-plan` import path; carry one prepared snapshot through P1's owner-map writer, native inspection, fresh measurement, final writer, stored replay and native-aware static/cache keys; production remains TOML/v2 until that chain is complete.



C35 (2026-11-20): added the internal-only `NativeAdmissionRequest` planner gate. `plan_engine_only` selects/verifies the matched engine and local devices, reads one immutable YAML snapshot, composes the filtered child environment once, invokes the tracked inspector with that complete environment, reconciles TP and ordered native rank BDFs to the selected local profile, and requires loopback listener compatibility. A successful inspection deliberately refuses before named-cache lookup, writer/P1, power admission, measuring or serving child. ABI44 `bc_serving_test` and `bc_session_test` pass (1/1 each); the planner fixture proves zero writer calls, zero power admission and no measuring/serving launch. Evidence C35 in `docs/evidence/radiance-latest-bump/README.md`; Ledger event `chg_20261119_145235_added-and-verified-a-private-f_1489`. This is not a public native launch path. Next: shared prepared ManifestInputs and edition-selected P1 owner-map producer, then native measurement/stored replay/cache identity coupling.



C36 (2026-11-20): hardened C35 with matched runtime-home validation, exact local ordinal-card order checks, reserved device selector variables (`RADIANCE_HOME`, ROCR/HIP/CUDA visibility and GPU_DEVICE_ORDINAL), and a direct legacy stored-replay refusal when the private native request is present. The native planner test now sets a named config and malformed cache-index tripwire and proves the gate stops before cache parsing. Negative cases cover card order, home, selector override and replay bypass. ABI44 `bc_serving_test`, `bc_session_test`, and `bc_session_stored_test` pass (1/1 each). Evidence C36; Ledger event `chg_20261119_150725_hardened-the-private-native-ad_5410`. Still no public native launch: P1 must accept the same prepared snapshot, and writer/measurement/stored replay/native-aware cache paths must be migrated together.

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
- 2026-11-19T09:02:16.301516+00:00 (updated-by): Updated: section:notes
- 2026-11-19T09:40:16.121520+00:00 (updated-by): Updated: section:notes
- 2026-11-19T10:09:03.548031+00:00 (updated-by): Updated: section:notes
- 2026-11-19T10:09:50.104454+00:00 (updated-by): Updated: section:notes
- 2026-11-19T12:01:22.900750+00:00 (updated-by): Updated: section:notes
- 2026-11-19T12:10:17.693225+00:00 (updated-by): Updated: section:notes
- 2026-11-19T12:28:07.874455+00:00 (updated-by): Updated: section:notes
- 2026-11-19T13:39:26.821638+00:00 (updated-by): Updated: section:notes
- 2026-11-19T14:04:37.285506+00:00 (updated-by): Updated: section:notes
- 2026-11-19T14:06:54.254829+00:00 (updated-by): Updated: section:notes
- 2026-11-19T14:19:49.888198+00:00 (updated-by): Updated: section:notes
- 2026-11-19T14:27:34.787625+00:00 (updated-by): Updated: section:notes
- 2026-11-19T14:39:17.405809+00:00 (updated-by): Updated: section:notes
- 2026-11-19T14:52:50.038889+00:00 (updated-by): Updated: section:notes
- 2026-11-19T15:07:40.385146+00:00 (updated-by): Updated: section:notes
