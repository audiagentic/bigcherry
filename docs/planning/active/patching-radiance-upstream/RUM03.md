---
id: RUM03
order: 2
plan: patching-radiance-upstream
state: in_progress
created-at: '2026-11-18T01:28:07.390588+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P0
work: M
---

# Port build dependencies and plugin ABI 43 with a consistent runtime SDK

## Description

Decouple BigCherry nlohmann dependency from removed upstream vendor header. Rebuild local plugins against one ABI/layout fingerprint; update Registry shared-set row/device eligibility without losing chain_fenced.

## Steps

- Replace the serving CMake requirement for ${RADIANCE_SRC}/vendor/nlohmann/json.hpp with one separately pinned dependency target supplied by the build lane. Share it with helper and contract generation builds.
- Declare the JSON package version/hash and any new owned source directory as build identity inputs; use the existing build plan rather than a second cache mechanism.
- Port rad0016 registry changes from code_device_/rows_/plugins_ to the upstream shared set. Preserve row.chain and row.chain_fenced when assigning per-rank code eligibility.
- Extend abi/rad_abi_layout_rows.h from the owning patch whenever it adds a RadBuildCtx field. Currently add tp_ownership in rad0018 and pipelined_prefill plus pipelined_prefill_min_rows in rad0020, with exact-prefix idempotent edits.
- Rebuild libr3, libr2, moe-remote, ngram-helper and bc-gfx12 against the same migrated headers as the engine. Update all local ABI layout rows and audit exported fingerprints.
- Inspect package/runtime home to exclude stale plugin .so files. Reuse the existing component coupling, radiance-bump and release/SDK verification checks; do not create a parallel ABI verifier.

## Detailed Solution & Technical Design

Actual blockers: serving/CMakeLists.txt explicitly checks vendor/nlohmann/json.hpp; bc_serving_v2.cmake and moe-remote/CMakeLists.txt inherit the upstream vendor include. Retain nlohmann for the initial bump because porting all strict JSON, JCS and schema-codec code to yyjson does not prove a better transition. Use a normal version-pinned target and propagate its include directories through target_link_libraries, replacing repeated raw include path assumptions.

Upstream Registry::commit_ now fills set_->rows and records chain_fenced from rad_kernel_chain==2. The rad0016 patch's old singleton fields and loader insertion anchors fail. Reapply per-device masks after the new row is fully classified, without discarding upstream fields or altering chain semantics. Keep one plugin eligibility decision shared by loader/planner/runtime.

The selected transition is ABI 35 -> ABI 43 at upstream commit 3deabdaeca17b58bad90d332f4b204cf7bb8a10e. This is not solved by updating a numeric constant. The layout fingerprint covers RadBatch, RadBuildCtx and probe fields. Rebuild all consumers; keep custom field rows in the fingerprint source. Do not copy installed old plugins into the target radiance_home.

## Code Samples & Guidance

Proposed CMake shape (names are illustrative; reuse the build lane's actual pinned target):
```cmake
# One configured target owns the JSON include path and immutable version.
target_link_libraries(bc_serving_v2 PUBLIC bc_json_dependency)
target_link_libraries(bc_serving PUBLIC bc_serving_v2)
```

Registry port sketch:
```cpp
// Preserve upstream row.chain and row.chain_fenced as assigned by commit_.
row.rank_code_mask = eligibility_for_resolved_devices(staged_plugin, devices);
set_->rows.push_back(row); // exactly one insertion, using target's storage
```
The rank_code_mask/helper names are proposed local fields/functions, not existing upstream APIs.

### Deep scan implementation refinement

libr3/CMakeLists.txt invokes r3_rewrite.py on target libr4d and exposes r3_rewrite_check. Treat the rewrite generator as a caller of all changed kernel interfaces and a hashed build input. Never patch generated backend outputs independently. Keep target chain_fenced fields when modifying set_->rows eligibility.

Source evidence and removal conditions: docs/reviews/radiance-upstream-2026-11-18/CONSOLIDATION.md. Use patch-footprint.json to inspect every overlapping patch before changing an attachment site.

## Files

- engines/radiance/components/serving/CMakeLists.txt
- engines/radiance/components/serving/bc_serving_v2.cmake
- engines/radiance/components/moe-remote/CMakeLists.txt
- engines/radiance/patches/rad0016_vtp14_rank_code_selection/patch.py
- engines/radiance/patches/rad0018_vtp14_n2_tp_ownership/patch.py
- engines/radiance/patches/rad0020_vtp14_tp3_pipelined_prefill/patch.py
- tools/tests/patch/test_vtp14_n2_tp_ownership.py
- tools/tests/patch/test_vtp14_tp3_pipelined_prefill.py
- tools/bigcherry/engine_adapters/radiance_bump.py
- tools/bigcherry/campaign/lane.py
- engines/radiance/engine.toml
- engines/radiance/recipes.toml
- tools/bigcherry/release/runtime_assets.py

## Validation

Apply and replay the ABI43 VTP14 patch prefix to the exact candidate; assert every local RadBuildCtx field has a matching ABI_FIELD row. Run the focused rad0018/rad0020 patch tests. Run the existing upstream plugin_test wrong-ABI and wrong-layout cases and the existing per-device rank eligibility fixture. At the recipe migration checkpoint, run the existing component-coupling and radiance-bump ABI/layout checks, build each packaged plugin against one generated SDK, and inspect the staged runtime for stale shared objects. Do not run all GPU kernel selftests here; RUM07/RUM09 own activated-path checks.

## Effort & Risk

M: stale shared objects or missing build identity inputs can masquerade as successful compilation. One SDK generation, explicit dependencies and a deliberate mismatch refusal are sufficient initial proof.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

No component reads the removed upstream JSON include location; dependency and local source bytes affect build identity. Engine and all packaged plugins agree on ABI/layout. The local RadBuildCtx fields tp_ownership, pipelined_prefill and pipelined_prefill_min_rows each have ABI_FIELD coverage in the layout fingerprint source. Chain-fenced row semantics survive mixed-device filtering. A stale ABI or layout plugin is refused with an actionable error.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM01. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisite RUM01; can proceed alongside RUM02 after design approval. Keep dependency cleanup limited to the actual removed header and duplicated include plumbing.

Astra source review ASTRA-CONSOLIDATION.md: execute after RUM01 and before native-config slice RUM04. Build only the minimum rank eligibility required by the selected lane; keep SDK/fingerprint inputs. No strict JSON-to-YAML rewrite.

RCS15–18 now own native libr3d replacement. A matched comparison build retains libr3 until native qualification; this does not change the initial bump's acceptance. See docs/reviews/radiance-native-libr3d/INTEGRATION.md.

Candidate changed from reviewed ABI 41 to exact upstream 3deabdaeca17b58bad90d332f4b204cf7bb8a10e (ABI 43). C1 dependency and C2 native parser compile checkpoint recorded in docs/evidence/radiance-latest-bump.

## Change Log

- 2026-11-18T01:28:07.390588+00:00 (created-by): Created by codex
- 2026-11-18T01:34:00.281140+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:42:56.897731+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:56:37.647939+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:19:44.169437+00:00 (updated-by): Updated: order=2, section:notes
- 2026-11-18T02:22:32.511127+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:25:34.495094+00:00 (updated-by): Updated: section:notes
- 2026-11-18T08:49:24.938096+00:00 (updated-by): Updated: section:notes
- 2026-11-19T00:54:06.118974+00:00 (updated-by): Updated: section:notes
- 2026-11-19T00:54:15.725512+00:00 (state-transition): State: pending → in_progress

## Reviews

- RV4234
- 2026-11-19T02:09:24.234699+00:00 (updated-by): Updated: section:title, section:detailed_solution
- RV4235
- 2026-11-19T02:15:39.669409+00:00 (updated-by): Updated: section:steps, section:files, section:validation, section:acceptance_criteria
