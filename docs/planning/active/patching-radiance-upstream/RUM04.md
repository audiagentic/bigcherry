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

All 18 serving configs are single YAML files (native sections plus x-bigcherry v1); no TOML catalogue, layout file, v2 serving contract or reader of them remains; every caller listed in the design is migrated; schema, agreement, single-source, variant and release tests pass; the resolved values of all 18 equal the pre-switch reference except the deletions the mapping lists; RUM09 start/canary/stop of each config passes on the 2.2 build.

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

C37 (2026-11-20): both P1 entry points now accept the caller's prepared ManifestInputs. The measured write flow constructs the selected resolved config once and passes it to both P1 owner planning and final writer; P1's temporary owner-map document copies only resolved plus native_config and retains its production/fresh-identity/stock-power projection. The standalone --publish-v2-p1 keeps its request JSON unchanged. Built bc-serve, bc_serving_test, bc_v2_p1_producer_test; focused CTest passed bc_serving_test and bc_v2_p1_producer_test (1/1 each). This transports a future admitted snapshot through P1 but does not enable native production selection, v3 writer, measurement attestation, or stored replay. Evidence C37.

C38 (2026-11-20): the isolated bc_serving_v3_writer_test now compiles v2/p1_producer.cpp, p1_measurement.cpp and p1_live_room.cpp against only the v3 wire library. After admitting and deleting authored YAML, the fixture runs real P1 using the same ManifestInputs and feeds its measured record into the real v3 writer; fake sysfs supplies live GTT usage and an embedding consumer plus margin matches the measured host-pool plan. Build and focused CTest passed (1/1). This is card-free v3 successor coverage only; production remains v2, and owner-attestation, cache/replay migration, and native GPU launch remain open. Evidence C38.

C39 (2026-11-20): native version-3 measured-plan keys now bind ordered engine/helper devices (role, ordinal, BDF, HIP ordinal, architecture, VRAM) alongside exact snapshot SHA/root. Cache files persist the key's version and lookup requires the envelope version to match; native v3 and legacy v2 remain separate. Tests prove helper reordering invalidates the native key, v3 cache store/read succeeds, and a v2 envelope is refused under a v3 key. Built bc-serve and bc_serving_test; focused CTest passed 1/1. Still open: production native selection and owner publication must bind plan/ranks/resident data to the exact tracked measuring attempt before native cache use. Evidence C39.

C40 (2026-11-20): integrated latest origin/main at 224456e53 (including 068776438) into the isolated bump branch; verified origin/main is an ancestor of HEAD and the worktree was clean after merge. Focused host_facts, run_records, and benchmark workloads Python tests passed (32). Only the existing pytest-asyncio fixture-loop-scope deprecation warning appeared. This keeps ongoing RUM04 work based on current main; no Radiance runtime behavior was changed by the merge. Evidence C40.

C41 (2026-11-20): native inspection now reconciles the parser-reported model with the selected model before the fail-closed planner boundary. Relative authored model paths resolve under the sealed snapshot root; both selected/reported paths are weakly canonicalized. Fake inspector and negative fixture cover model mismatch. ABI44 bc_serving_test and bc_session_test passed (1/1 each). Production still stops before cache, writer, power, or child startup. Evidence C41.

C42 (2026-11-20): moved measured-plan publication from CLI raw-file import into the tracked session owner after engine exit, helper reap and VRAM release. The owner freezes resolver values, model/build identity and key before measurement, recomputes before publish and refuses drift rather than re-keying. Unique attempt paths/IDs, strict plan/rank/config validation, owner receipt digests (including resident memory and optional native snapshot identity), unique atomic cache publication, and expected-attempt cache-only writer checks are implemented. Focused ABI44 `bc_serving_test` and `bc_session_test` passed (2/2), including two measured rounds after exit-11 pool adjustment and refusal after tampering with resident memory. Evidence C42 is in docs/evidence/radiance-latest-bump/README.md; Ledger event recorded for RUM04. Production remains v2 and the private native admission gate remains closed before writer/P1/measurement/replay. No GPU execution or performance claim.

C43 (2026-11-20): closed the gap where the session froze a plan key before invoking a writer that independently resolved mutable files. The session now supplies `--expected-plan-key`; the real `bc-serve` child recomputes it from the same `Resolved` object passed to the unmeasured manifest builder and refuses drift before writing. The session verifies the pinned draft's config name, model and ordered engine BDFs against the frozen selection. The reported attempt is assigned only after final-round success. `bc_serving_test` invokes the real built writer for both successful pool rounds and changes `engine.num_speculative_tokens` between freeze and draft; the child refuses and no measurement-attempt directory is created. `bc_serving_test` and `bc_session_test` passed (2/2); `git diff --check` passed. Evidence C43 and Ledger event recorded. This closes owner-key-to-v2-draft binding only; native writer/P1/replay production migration remains open.

C44 (2026-11-20): extracted rad0011 engine rank-demand parsing into shared `rank_demand.{h,cpp}`. Both `plan_cache_store_owned` and the P1 producer now require the exact four-field v1 envelope and five-field per-rank rows, selected engine BDF order, nonempty rad0011 evidence, positive required/fixed/claimable byte values within the measured 53-bit wire range, and `fixed <= required <= claimable`. Owner cache coverage rejects wrong BDF and zero required bytes. Focused `bc_serving_test` and `bc_v2_p1_producer_test` passed (2/2); `git diff --check` passed. Evidence C44 and Ledger event recorded. The shared decoder improves existing v2 owner/P1 safety only; native v3 session launch remains gated.

Astra's ABI44 review (2026-11-20) sets the next RUM04 implementation boundary: keep v2 and v3 generated readers isolated because they export identical symbols with incompatible `Spec` layouts; create a separate candidate engine build selecting the single v3 reader via `bc_add_manifest_library` and successor-generated codec, never link `bc_serving_v3_wire` alongside the existing v2 `rad_core`. Expose a typed `native_config_view(const Manifest&)` that works after strict `read_manifest` for both measuring and serving roles; `engine_view` is measured-only because it requires `EngineMemory`. Then prepare `rad::Config` from that admitted snapshot through `NativeFleetConfig::replay` under its verified root/environment, preserving YAML provenance (`cfg.given`) and role/device order. Test actual v3 writer outputs for both states after deleting/replacing authored YAML; verify an upstream-only option/provenance, ordered devices, and `plan_mismatch` when roles/documents are swapped. Keep production admission closed until the complete session/P1/cache/stored/offline chain uses that v3 reader; managed reload must refuse before reopening `FleetConfig.path`. Astra checked candidate source compatibility with v3 headers using syntax-only compilation; that is not a link/runtime or production-enable result.

C45 (2026-11-20): exposed `v2::native_config_view(const Manifest&)`, returning the generated edition's exact admitted YAML bytes and SHA/root identity after `read_manifest` for any role. The real isolated v3 writer test asserts this typed view for unmeasured measuring and measured serving outputs after the authored file is deleted; the existing `engine_view` remains measured-only because it requires EngineMemory. Rebuilt v2 serving and isolated v3 writer targets; `bc_serving_test` and `bc_serving_v3_writer_test` passed (2/2). Evidence C45 and Ledger event recorded. This API does not yet call upstream `NativeFleetConfig::replay`, migrate session/stored/offline consumers or open native production admission.

C46 (2026-11-20): added a typed strict-reader snapshot replay seam and a separate ABI44 consumer linked to the generated v3 reader only. The real v3 writer fixture supplies measuring and serving documents after authored YAML removal; the upstream fleet parser preserves the native-only mixed-step-target-ms value and cfg.given source, exact snapshot identity, and refuses role swaps. Fresh card-free ABI44 build and CTest pass; rad0009 patch class passes with TMPDIR on an executable filesystem. Evidence C46 in docs/evidence/radiance-latest-bump/README.md; Ledger event chg_20261119_165938_the-native-yaml-snapshot-now-p_9902. This remains an isolated consumer proof. Next, migrate one real owner path at a time (native P1, measuring/session/cache receipt, stored/offline replay), retain the v2 default, and keep the native session gate closed until all consumers consume the same admitted snapshot and role contract.

C47 exact-recipe follow-up (2026-11-20): corrected candidate provenance to a fresh current `recipes.toml` materialization: upstream `3deabdaeca17b58bad90d332f4b204cf7bb8a10e`, source tree `8ca1308f23c0eaef5fdaa0ce29b9cfa7225925fd`, all 33 selected serving-core patches including rad0050–rad0053. Hardened candidate CMake overrides to require explicit test-only v3 mode and required writer/config/upstream parser. Serial ABI44 candidate build and exact-composition `bc_native_v3_engine_test` passed (1/1); link has one manifest archive. Focused rad0009 patch tests passed (5/5); diff check clean. Evidence: `docs/evidence/radiance-latest-bump/README.md` C47. This supersedes manually staged candidate provenance, without changing C46 behavioral scope. Next slice: use the actual production v3 plan key including ordered rank_devices and owner cache receipt/store/lookup, then exercise P1 and writer against that owner record. Keep production gate closed.

C48 (2026-11-20): build_manifest now recomputes the complete v3 plan key from resolved values/facts, admitted YAML snapshot, and measured model/build/engine-argument identities; requires owner provenance and the cache record key to match. This fixes the former SHA/root-only check that rejected production `rank_devices`. The real v3 writer fixture now publishes and reloads the plan/rank-demand via plan_cache_store_owned/lookup using the measuring-manifest digest, feeds that exact cache entry into P1 and final writer, and verifies real identities plus refusal of changed digest/root/device BDF. Focused `bc_serving_v3_writer_test`, exact-recipe ABI44 `bc_native_v3_engine_test` (writer child included), and default v2 `bc_serving_test` passed (1/1 each); single generated manifest archive confirmed; diff check clean. Evidence C48 in docs/evidence/radiance-latest-bump/README.md. No GPU or production gate change. Next remaining seam is session-owned native snapshot measurement and real cache receipt propagation through stored/offline replay; keep production admission closed.

C49 (2026-11-20): factored sealed memfd validation/snapshot reconstruction into `NativeConfigSnapshot::from_sealed_fd()` and reused it in Radiance's native inspector. The private v3-only `bc-serve --bc-native-config-fd` writer input populates `ManifestInputs.native_config` before both key calculations. Round-trip coverage passes in `bc_serving_v3_writer_test`; exact-recipe ABI44 consumer CTest and default v2 `bc_serving_test` pass; real `bc-serve` compiles against the isolated generated v3 codec and rejects the private fd option outside write-manifest mode. Evidence C49. No session caller or native measurement path is enabled; session admission still fails closed in default v2, and named/stored/offline replay and GPU start remain open.

C50 (2026-11-20): isolated generated-v3 session now carries the owner-admitted NativeConfigSnapshot through a sealed fd to the real `bc-serve --write-manifest --unmeasured` child, with a frozen expected native plan key. It strictly pins measuring-role output, verifies snapshot bytes/SHA/root and ordered engine devices, confirms no engine_memory, then refuses before named cache/P1/power/GPU. Test deletes authored YAML after inspector exit, verifies writer success from sealed bytes, both children reaped, cache tripwire untouched and no plan cache. Focused `bc_native_v3_session_test`, `bc_serving_v3_writer_test` and default v2 `bc_serving_test` pass (1/1 each). Evidence C50. Still no native GPU measurement/cache publication/stored replay; default v2 behavior remains inspector-only refusal. Follow-up gates: route native plan-memory through NativeFleetConfig; reconcile native context/sequences/pool and parser-affecting environment; check the frozen key on first measured lookup and owner retry; define relationship between manifest integrity cache key and owner plan-cache v3 key for stored/offline replay; make managed reload refuse or require fresh admission; only then consider production gate.



C51 (2026-11-20): Astra review RV4314 found that the isolated v3 writer could accept raw `--measured-plan`/`--rank-demand` inputs before enforcing the sealed native snapshot, allowing a legacy v2 key/cache touch before later v3 refusal. The writer now requires `--bc-native-config-fd` for every v3 write and rejects both raw imports at CLI argument validation before file access. Added real `bc-serve` card-free refusal coverage. Rebuilt and directly ran `bc_native_v3_session_test` and ordinary `bc_serving_test`; both passed. Evidence C51 in `docs/evidence/radiance-latest-bump/README.md`. This closes only the writer ingress gap; native `--plan-memory`, owner cache publication and production admission remain open.



C53 (2026-11-20): retained the native resolution-root directory descriptor from before YAML snapshot read through owner inspection, rechecking path/inode identity after child completion and before future measuring spawn. Inspector and native measuring entry borrow the same CLOEXEC root fd; spawn fchdir occurs only after durable registration, while the sealed manifest remains the only explicitly inherited fd. Future native measurement argv uses absolute engine and plan/rank output paths and the inspector's captured environment. Reconciliation now permits omitted optional rank_devices while still checking selected TP/BDF count and card ordinals; any explicit native rank-device order must match local BDF order. Focused bc_session_test passed 1/1 and diff check passed. The v3 planner remains fail-closed before cache/P1/power/measurement; this is descriptor/reconciliation plumbing, not a launched native measurement. Evidence C53.

C54 (2026-11-20): the isolated v3 writer now accepts an owner-frozen `--expected-plan-key` for measured sealed-snapshot cache reads and verifies it before lookup/publication. Added real-CLI regression coverage to `bc_native_v3_session_test`; execution is pending because the existing `native-v3-measuring-c51-build` does not define that target. Refresh the isolated candidate build from this worktree and run the focused target before accepting this slice. See evidence C54 and Ledger event `chg_20261119_191813_added-the-v3-measured-writer-f_1596`.

C55 (2026-11-20): `replay_stored_manifest()` explicitly rejects `bigcherry/v3` envelopes even when callers omit `NativeAdmissionRequest`; a direct call can no longer rely on incidental v2 reader rejection. Added a card-free `bc_session_stored_test` case. Runtime test pending because the available candidate build needs refreshing. See evidence C55.

C56 (2026-11-20): retained the strict inspector result and now reconcile its listener port, context, sequence limit and ordered runtime kernels against the pinned native measuring draft before power admission or GPU child startup. Refreshed candidate builds from this task worktree: generated-v3 `bc_native_v3_session_test` passes (1/1); default-v2 `bc_session_stored_test` passes (1/1). C54 measured-key and C55 direct-v3-replay coverage are included in those verified targets. See evidence C54-C56. Production admission remains closed pending actual owner measurement/cache/P1/replay chain.

C57 (2026-11-20): rad0011 now records rank demand only when each post-split `rad::Config.devices` ordinal resolves via `rad::dev_pci_address()` to the selected ordered BDF. `plan_cache_store_owned()` also requires each rank receipt's claimable bytes to equal the paired EnginePlan card's free-VRAM fact. Targeted verification passed: rad0011 patch composition (1/1), default-v2 serving + stored-session CTests (2/2), generated-v3 session + real writer CTests (2/2). See evidence C57. No GPU used.

C58 (2026-11-20): v3 measured writer cache reads now require both owner-frozen `--expected-plan-key` and `--expected-owner-attempt` before snapshot/cache processing. The effective native key is recomputed before cache lookup; the selected cache receipt is checked against the expected attempt. Fresh generated-v3 `bc_native_v3_session_test` passes (1/1), including refusal for a missing owner attempt. See evidence C58. This ingress gate is not the actual session-owned measurement path.

C59 (2026-11-20): Sol review RV4315 found that the committed rad0011 check used copied `rank_cfg_[i].devices`, which is not the physical per-rank device after Engine::plan() clones cfg_. Corrected the rad0011 hook to receive the ordered `ranks_[i]->device` values at the post-split anchor and compare `rad::dev_pci_address(actual_rank_devices[i])` to the selected ordered BDFs; Config rank splits remain untouched. In the session-owned v3 path, engine resident MiB now derives from the validated `resident_other_bytes` sidecar using rad0011's ceil/minimum-one rule and the log must agree; post-measure engine identity is recomputed against the newly resolved runtime home, and final typed home/build plus P1 resident bytes are checked. Fixed the serving-key assertion: the v3 manifest serving-contract cache key and operational owner key have distinct domains; strict serving-role read validates the former while the final plan, model/build identity, owner attempt and measuring-manifest digest are checked independently. The native v3 fixture now requires exactly one attempt/cache entry, the actual pinned measuring-manifest SHA in the owner receipt, matching attempt directory, resident rounding, strict serving read, measured plan/model/device identities, and P1 resource-input binding. Focused generated-v3 `bc_native_v3_session_test` passed (1/1); default-v2 `bc_serving_test` and `bc_session_stored_test` passed (2/2); `TestPlanMemoryPatch` passed (9/9) with executable TMPDIR under `/work/bigcherry`; freshly composed exact-pin ABI44 native-v3 engine target built and CTest passed (1/1). Materialization used all 33 patch IDs explicitly listed by source.radiance at upstream 3deabda; the standard resolver currently refuses the recipe because selected rad0046 remains state `untested` while the patch-set requires `validated`. This state was left unchanged and the test composition is not a claim that normal resolver validation passes. Hardware physical-BDF ordering is still unproven because no GPU was used. Production admission, stored/offline native replay and release gates remain closed. RV4315's code and test findings are incorporated; retain the hardware ordering and recipe-state limitations as open gates.

C60 (2026-11-20): incorporated RV4316's runtime-home identity finding. Freeze `fs::canonical(home)` before native inspection, use that canonical target for `RADIANCE_HOME`, inspector input and engine build identity, and compare both the post-measure resolved path and final manifest path to the frozen target. Extend `bc_native_v3_session_test` to retarget the selected home symlink after inspection to a byte-identical directory and prove refusal before owner-cache/measured-manifest publication. The fake sidecar now uses a valid resident byte count one KiB below an MiB boundary and its matching rounded-up log value, exercising rad0011's ceiling rule. After these changes, generated-v3 `bc_native_v3_session_test` passes (1/1), and default-v2 `bc_serving_test` plus `bc_session_stored_test` pass (2/2). Physical PCI BDF order still needs hardware evidence; recipe resolver still refuses selected rad0046's `untested` state, which was intentionally left unchanged. RV4316 findings are incorporated; no production/replay gate opens.

C63–C65 verify the owner-held cwd/environment, engine-only native v3 fresh startup, and named plus explicit stored replay through the sealed owner path. See docs/evidence/radiance-latest-bump/README.md. The exact-byte manifest, S8 sidecar, owner receipt, current build/root, and fresh host room checks were exercised. Remaining work includes the existing offline KL caller using the same snapshot, helper-on integration, managed reload disposition, production generation selection, and deleting the duplicated native fields once their callers migrate.

The tracked native-serving build selects wire edition 3. Engine, bc-serve and the MoE worker projects generated byte-identical codec sources and passed fingerprint verification. The real bc-serve CLI now accepts --native-config YAML --model-root DIR for named fresh starts and uses embedded YAML for exact replay. Host-only contract and v3 wire tests passed 2/2; recipe, component coupling and build identity/forwarding checks passed 23 tests and 10 subtests. Radiance 2.2 parsed the Flash-Next YAML and refused a server port placed under a model. Remaining: full matched live launch on GPU, drift and quality checks on the real model, shipped profile selection, and removal of duplicated legacy native projections.

Checkpoint 4a03a8172 reuses the full model digest within one cold measurement when the complete file stamp still matches. It logs the initial full-content scan and avoids copying the hashing buffer. Existing digest bytes and drift checks are retained; v3 owner and stored cache tests passed 2/2.

The first live RUM09 attempt reached measurement and refused a missing YAML host-pool value because it differed from the admitted per-rank pool. Checkpoint 1d1ce5be7 uses the admitted pool when the YAML omits it and rejects an explicitly conflicting pool. Exact 2.2 ABI44 fixture passed measuring, serving and offline preparation. A separate incremental runtime is retrying the original YAML under the GPU lease.

The isolated GPU retry passed native pool preparation and wrote the live engine plan and rank receipt. The final measured writer returned exit 11 because the 9216 MiB trial pool differed from its 2247 MiB derived requirement. No serving or replay occurred. Add bounded native remeasurement for omitted YAML pools with a distinct effective input key per attempt and expose explicit pool conflicts; preserve helper rollback and exact receipts.

Exact b0c256877 ABI44 integrated Flash-Next v3 with three physical helpers passed fresh and named starts, then explicit stored replay after the authored YAML was removed and VRAM had recovered from shutdown. All three starts report the same manifest digest and measurement attempt. Immediate explicit replay refused live VRAM after stop as designed. The authored YAML reached the upstream parser and the recorded TP1 engine BDF and three disjoint helper roles were admitted. Finite KL and shipped configuration/install checks remain part of the integrated proof.

59ab547ce ships the portable flashnext-4gpu.yaml and resolves it for verified installed v3 named start/serve without an explicit --native-config. The exact v2.2 engine sealed inspector accepted model/rank order, the libr4d/libr3moe/libref order, loopback 36080, 262144 context and 8 sessions. Build-tree explicit YAML and installed variants without a bundled YAML remain supported through the explicit flag. Remaining migration boundary: owner resolver and writer still mirror native placement, n-gram, cache dtype, speculation, RoPE/prefix, batch and checkpoint settings; RUM04 needs to derive required admission/receipt values from the typed upstream inspection and remove duplicate v3 defaults. Preserve v2 compatibility and replay proof, then run the installed named start on the verified release path.

Audit of d92592f9: the sealed manifest declared 12288 MiB host KV and 4705 MiB reserve, but the native engine bridge explicitly supplied only the pool, leaving omitted YAML knobs at upstream 0 and 4096. 6fe47b998 binds typed host KV and reserve in measurement and serving, binds the measured slot count only in serving, preserves the offline scorer’s allocation, and refuses draft/final KV or reserve divergence. Exact fixture replay passed. Rebuild owner/engine together, reuse the matched ENG05 plugins, and prove both resources in engine logs before comparing performance. Remaining authority work: use one typed projection from the matched upstream inspector inside the existing sealed writer FD for draft, P1, measured manifest and stored replay. Required receipts are listener/limits, kernel order, batch/chunk, placement, n-gram location, KV format, fixed speculation, RoPE and prefix policy. Keep BigCherry physical placement, helper transport, retention objective and measured resources as owner controls. Preserve v2 and the distinct named static, owner attempt and wire plan keys. Explicitly cover native auto and omitted values or refuse ambiguous release values before admission.

b8d41e31d admits native listener, limits, kernel order, batching, checkpoint interval, KV format, placement, speculation, RoPE and prefix policy into the resolver only from the matched engine’s sealed native inspection. The same receipt reaches the draft writer, measured P1 and final writer, and named/explicit replay. Verified installed explicit replay uses bundle configs and authenticated YAML resolution root when omitted. Focused native owner and exact upstream parser tests passed. Remaining qualification includes actual v3 bundle install/replay and the supported explicit handling of native auto defaults.

2026-11-20 owner decision: a serving config is one seamless native YAML file per deployment with a closed, typed top-level x-bigcherry section; the RSC12 TOML catalogue, layout files and their readers are deleted in this change. This replaces the step 'use existing BC profile selection first; add only a narrow x-bigcherry selector if needed'. Design and full key mapping (153 resolver keys plus file-level TOML keys): docs/design/radiance-native-config-x-bigcherry.md. The interim YAML-beside-slimmed-TOML commits d99323272..c04c69128 are reworked to that design (their YAML files, preflight and inspector tuning report are kept).

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
- 2026-11-19T15:15:51.751252+00:00 (updated-by): Updated: section:notes
- 2026-11-19T15:21:41.245162+00:00 (updated-by): Updated: section:notes
- 2026-11-19T15:30:30.513934+00:00 (updated-by): Updated: section:notes
- 2026-11-19T15:32:53.989125+00:00 (updated-by): Updated: section:notes
- 2026-11-19T15:43:06.516924+00:00 (updated-by): Updated: section:notes
- 2026-11-19T16:11:28.537296+00:00 (updated-by): Updated: section:notes
- 2026-11-19T16:24:38.328581+00:00 (updated-by): Updated: section:notes
- 2026-11-19T16:33:33.967028+00:00 (updated-by): Updated: section:notes
- 2026-11-19T16:38:19.396059+00:00 (updated-by): Updated: section:notes
- 2026-11-19T16:59:55.010146+00:00 (updated-by): Updated: section:notes
- 2026-11-19T17:20:24.644470+00:00 (updated-by): Updated: section:notes
- 2026-11-19T17:30:20.876923+00:00 (updated-by): Updated: section:notes
- 2026-11-19T17:47:58.678450+00:00 (updated-by): Updated: section:notes
- 2026-11-19T18:02:50.675833+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4314
- 2026-11-19T18:12:45.949208+00:00 (updated-by): Updated: section:notes
- 2026-11-19T19:13:03.651665+00:00 (updated-by): Updated: section:notes
- 2026-11-19T19:18:49.935154+00:00 (updated-by): Updated: section:notes
- 2026-11-19T19:21:04.806627+00:00 (updated-by): Updated: section:notes
- 2026-11-19T19:35:09.573526+00:00 (updated-by): Updated: section:notes
- 2026-11-19T19:43:56.206142+00:00 (updated-by): Updated: section:notes
- 2026-11-19T19:46:03.317466+00:00 (updated-by): Updated: section:notes
- RV4315
- 2026-11-19T20:23:36.088097+00:00 (updated-by): Updated: section:notes
- RV4316
- 2026-11-19T20:33:39.682508+00:00 (updated-by): Updated: section:notes
- 2026-11-19T22:03:18.923273+00:00 (updated-by): Updated: section:notes
- 2026-11-19T22:58:09.531280+00:00 (updated-by): Updated: section:notes
- 2026-11-19T23:20:19.446517+00:00 (updated-by): Updated: section:notes
- 2026-11-19T23:49:40.807569+00:00 (updated-by): Updated: section:notes
- 2026-11-19T23:59:22.141784+00:00 (updated-by): Updated: section:notes
- 2026-11-20T03:03:46.253523+00:00 (updated-by): Updated: section:notes
- 2026-11-20T03:22:59.663552+00:00 (updated-by): Updated: section:notes
- 2026-11-20T03:36:45.051737+00:00 (updated-by): Updated: section:notes
- 2026-11-20T03:53:19.772616+00:00 (updated-by): Updated: section:notes
- 2026-11-20T10:44:09.358513+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
