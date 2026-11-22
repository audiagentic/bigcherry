---
id: RUM12
order: 0
plan: patching-radiance-upstream
state: completed
created-at: '2026-11-21T23:41:27.558357+00:00'
breadth: ''
skill: advanced
created-by: claude
work: M
priority: P1
---

# Named start reuses the last measured plan; stat-based model identity

## Description

Starts must be quick and reuse what is already measured: a named serve/start reuses the most recent good measured manifest for its config instead of measuring again, and never hashes a whole model.

## Steps

- Move the named-config index from the run dir to a work-root store keyed on the resolved settings, ordered cards and VRAM, power profile, config inputs and model path/size/mtime; drop the engine, helper and bc-serve builds and the installed runtime path from the key.
- Identify the model by canonical path, size and mtime; never read it.
- A stored plan the replay checks refuse is measured once afresh and replaces the entry; an engine that fails to come up on a stored plan drops the entry.
- Measure the second named start of flashnext-r9700 on cards against the replay time.

## Detailed Solution & Technical Design

Store: the named-config index moves from the run dir to <work-root>/cache/named-configs/<sha256(config)>.json (named_store_index). Key (static_resolution_key v4): resolved effective values minus engine.radiance_home, handoff.shm and the derived memory.host_pool_mib/host_kv_mib; operator --set values except layout.name, handoff.shm and engine.radiance_home; ordered cards with arch/ordinal/VRAM; power profile; config input file digests; model identity. The producer, engine, helper and bc-serve builds, native YAML hash and installed runtime path are provenance, not key. Model identity (model_stat_identity): canonical path, size and mtime; the model is never read.
Lookup (session.cpp native path): stored_manifest_for(index, key) returns a miss on any damage. A hit is replayed (replay_stored_manifest; power admission is its last step). When replay refuses (a different engine or helper build, tampered bytes), load_carried_measurement reads the old attempt's single plan-cache entry (plan, rank demand, resident helper/engine MiB; its owner digest must equal the index receipt), pre-validates it with check_plan_for_configuration and decode_engine_rank_demand, and seals it again under this build through the draft and measured writers (no helper or measuring engine); provenance is plan_carried_from plus carried-from.json. Anything that cannot be carried measures once afresh and replaces the entry. An engine that fails to come up on a reused plan removes the index (EngineStart.reused_store). Slice 3 (2026-11-22): engine.radiance_home, which bc-serve adds as a --set on every installed start, was part of the key and defeated cross-build reuse; excluded.

## Code Samples & Guidance



## Files

engines/radiance/components/serving/stored_manifest_cache.{h,cpp} (key v4, model_stat_identity, named_store_index, StoredManifestIndex.directory, stored_manifest_for); session.cpp (named_cache, native lookup, load_carried_measurement, carried sealing, owner invalidation); session.h (reused_store, work_root); main.cpp (usage, work-root help); tests stored_manifest_cache_test.cpp, serving_test.cpp, session_test.cpp; docs serving README, SERVING_MANIFEST.md, BUMP_COUPLING.md, PIN_BUMP.md, bump-radiance skill.

## Validation

Hardware-free: stored_manifest_cache_test (key, stat identity, store), serving_test (cross-run-dir reuse without measuring, fall-through to one fresh measurement on a different engine build), session_test (entry dropped when the engine fails on a stored plan). On cards: the second named start of flashnext-r9700 about the replay time (44 s with the work root), no measuring, no fingerprint.

## Effort & Risk

M. Risk: a reused plan was measured under other conditions; contained by admission, which runs on every start. Relaxed checks and what still protects each: (1) full model fingerprint -> path/size/mtime: an in-place replacement keeping size and mtime is not detected by the key, but the engine checks the container it loads and admission refuses a plan that does not fit. (2) engine/helper/bc-serve build digests in the key -> provenance: a build the old plan does not suit is refused by the replay checks, then carried and re-sealed (the writers verify the new build) or measured; the engine refuses a plan that does not fit. (3) native YAML hash and installed runtime path in the key -> the YAML bytes are still sealed and inspected on every start, and config input digests stay in the key. (4) per-run-dir store -> work-root store: an entry is bound by key and owner receipt digest, and damage is a miss. (5) replaying a plan from another build -> carry re-seals under the current build, pre-validated against the current configuration and rank demand. (6) power admission and the live room check are unchanged on every start, reuse included. (7) an engine failure on a reused plan drops the entry, so the next start measures.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries (docs/standards/ARCHITECTURE_STANDARDS.md): config over code, fail closed on damaged stored state (a miss, never a guess), no per-model constants.

## Acceptance Criteria

A second named start with unchanged inputs reuses the stored manifest from any run dir and any build, runs no measuring child and reads no model bytes; a stored plan that cannot be reused measures once and replaces it; each relaxed check and what still protects it is listed (Effort & Risk). Cards (bigcherry-1, 2026-11-22): same-build second start ran no measuring attempts (flashnext-r9700 69 s to 54 s, qwen27b-tp3 35 s to 26 s); the cross-build carry re-run is pending after the radiance_home key fix.

## Notes

2026-11-22 (bigcherry-2): slice 2, cross-build reuse. A stored manifest is bound to the build that wrote it (engine identity, runtime home), its measurement is not. When replay of the stored manifest is refused (a different engine or helper build), the start reads the stored measurement back from the old attempt's owner cache (plan, rank demand, resident host memory; the entry's owner digest must equal the index's receipt) and seals it again under the current build: the draft and measured writers run, no helper or measuring engine does, the new attempt gets its own owner receipt, S8 evidence and index, and the old attempt and engine identity are kept as plan_carried_from (selection and carried-from.json). A stored measurement that cannot be carried (damaged, or its plan no longer fits the configuration) is measured afresh. If anything fails after carrying, the stored entry is dropped. Tests: serving_test (different engine build: no measuring engine, one measured writer, provenance recorded, store replaced; unusable store: measures). Remaining on cards: the second named start of flashnext-r9700 and qwen27b-tp3 on the same build (replay) and on a newer installed build (carry).

2026-11-22 completion (bigcherry-1, card evidence on the installed releases, own short gpu lease per run, work root shared by the arms of one config). Same-build reuse, bc-radiance 1170.38.0: second named start flashnext-r9700 69 s -> 54 s with 0 measuring attempts, qwen27b-tp3 35 s -> 26 s, 0 attempts, canaries correct. The first cross-build attempt (1170.38.0 store, 2020.0.0) did not engage: the store key still carried the owner-injected engine.radiance_home --set, so different installed prefixes missed (diagnosed by bigcherry-2, fixed in #1286). After #1286, bc-radiance 2020.0.1 installed under two prefixes: first start measures once (the key changed once: flashnext-r9700 73 s, 2 measuring attempts; qwen27b-tp3 39 s), the start under the second prefix logs 'the stored manifest is not replayable as it is (replay: the engine, helper and runtime home must belong to this bc-serve build); reusing its measurement under this build', writes carried-from.json and no measuring engine (flashnext-r9700 55 s, qwen27b-tp3 34 s) and the canary is correct. That is the same code path a different build takes (strict replay refused, measurement carried, plan re-sealed with the current writers), so the carry is shown on cards for a changed prefix; a genuinely different engine build will exercise it on the next release and is recorded by plan_carried_from. Numbers: reuse takes engine-start time only (about 25-35% less wall time than the first start on these configs; the first start itself is dominated by the measurement and a 62 GiB host-pool plan on flashnext-r9700). Residual limits, as listed in the acceptance: model replacement keeping path, size and mtime is outside stat identity; same-config-different-site-file changes miss by design.

## Change Log

- 2026-11-21T23:41:27.558357+00:00 (created-by): Created by claude

- 2026-11-21T23:41:35.527696+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance
- 2026-11-21T23:57:41.696211+00:00 (updated-by): Updated: section:notes
- 2026-11-22T01:35:02.089724+00:00 (state-transition): State: pending → in_progress
- 2026-11-22T01:35:14.769977+00:00 (updated-by): Updated: section:detailed_solution, section:files, section:effort_risk, section:standards, section:acceptance_criteria
- 2026-11-22T02:25:14.761613+00:00 (updated-by): Updated: section:notes
- 2026-11-22T02:25:20.319798+00:00 (state-transition): State: in_progress → completed
