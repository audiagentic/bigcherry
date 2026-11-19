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

# Port build dependencies and compose local ABI44 from upstream ABI43

## Description

Decouple BigCherry nlohmann dependency from removed upstream vendor header. Rebuild local plugins against one ABI/layout fingerprint; update Registry shared-set row/device eligibility without losing chain_fenced.

## Steps

- Replace the serving CMake requirement for upstream's removed vendor JSON header with one separately pinned dependency target, shared with helper and contract-generation builds.
- Declare the JSON package version/hash and owned source bytes as build identity inputs; retain the existing build plan rather than adding another cache layer.
- Port rad0016 registry changes to the upstream shared set. Preserve row.chain and row.chain_fenced while assigning per-rank code eligibility.
- Use upstream ABI43 as the frozen source baseline. Compose existing local tp_ownership, pipelined_prefill and pipelined_prefill_min_rows fields with RUM07's kv_layout fields in one local ABI44 extension. Append fields, bump RAD_ABI_VERSION once to 44, and add every field in abi/rad_abi_layout_rows.h; never change exported struct shape while retaining version 43.
- Rebuild libr3, libr2, moe-remote, ngram-helper, bc-gfx12 and every shipped plugin against the same ABI44 headers and SDK. Update component build identity and verify all staged shared objects; do not copy stale plugins from a prior installation.
- Keep the frozen ABI43 loader refusal fixture as an upstream-baseline check, then add local ABI44 checks proving ABI43 plugins are refused and matching ABI44/layout plugins load. Reuse existing component-coupling and release/SDK verification.

## Detailed Solution & Technical Design

Actual blockers: serving/CMakeLists.txt explicitly checks vendor/nlohmann/json.hpp; bc_serving_v2.cmake and moe-remote/CMakeLists.txt inherit the upstream vendor include. Retain nlohmann for the initial bump because porting all strict JSON, JCS and schema-codec code to yyjson does not prove a better transition. Use a normal version-pinned target and propagate its include directories through target_link_libraries, replacing repeated raw include path assumptions.

Upstream Registry::commit_ now fills set_->rows and records chain_fenced from rad_kernel_chain==2. The rad0016 patch's old singleton fields and loader insertion anchors fail. Reapply per-device masks after the new row is fully classified, without discarding upstream fields or altering chain semantics. Keep one plugin eligibility decision shared by loader/planner/runtime.

The selected transition is ABI 35 -> ABI 43 at upstream commit 3deabdaeca17b58bad90d332f4b204cf7bb8a10e. This is not solved by updating a numeric constant. The layout fingerprint covers RadBatch, RadBuildCtx and probe fields. Rebuild all consumers; keep custom field rows in the fingerprint source. Do not copy installed old plugins into the target radiance_home.

Local ABI extension: ABI43 is the exact upstream source baseline. Append kv_layout to RadBuildCtx and RadKVGroupDecl plus a per-format capability mask to RadArchProbe. Combine those fields with the existing local tp_ownership, pipelined_prefill and pipelined_prefill_min_rows additions and bump RAD_ABI_VERSION to 44 once. Add each new field to rad_abi_layout_rows.h; the loader fingerprint covers version, structure size, offsets and sizes. Do not mutate the exported struct shape while still reporting ABI43.

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
- engines/radiance/patches/rad0040_kv_layout_table/patch.py
- engines/radiance/engine.toml
- engines/radiance/recipes.toml
- abi/rad_abi.h
- abi/rad_builder.h
- abi/rad_abi_layout_rows.h
- tools/tests/patch/test_vtp14_n2_tp_ownership.py
- tools/tests/patch/test_vtp14_tp3_pipelined_prefill.py
- tools/bigcherry/engine_adapters/radiance_bump.py
- tools/bigcherry/campaign/lane.py
- tools/bigcherry/release/runtime_assets.py

## Validation

- Apply and replay the ABI43 VTP14 source patch prefix against the exact frozen commit; verify its upstream baseline and existing edits before the local ABI extension.
- Always-on ABI fixture proves ABI44 and the layout fingerprint include tp_ownership, pipelined_prefill, pipelined_prefill_min_rows, KV layout IDs/capabilities, and all new exported fields in exact order.
- Loader tests refuse an ABI43 plugin against local ABI44 and refuse a same-version foreign fingerprint or missing fingerprint; matching ABI44 plugins pass. Preserve the existing upstream wrong-ABI and wrong-layout checks.
- Run focused rad0018/rad0020 tests and existing rank eligibility tests. At recipe migration, build every shipped plugin and engine against one SDK, verify all runtime assets report ABI44 and the expected fingerprint, and inspect staged roots for stale plugin binaries.
- RUM07/RUM09 own activated cache-kernel and GPU correctness gates; do not expand this item into a full GPU selftest run.

## Effort & Risk

M: stale shared objects or missing build identity inputs can masquerade as successful compilation. One SDK generation, explicit dependencies and a deliberate mismatch refusal are sufficient initial proof.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

The frozen upstream source remains pinned to ABI43, while every local exported-struct extension is composed once into ABI44. RAD_ABI_VERSION and rad_abi_layout_rows.h agree for all added fields; stale ABI43 or foreign-layout plugins are refused. The engine and every shipped plugin are rebuilt against one SDK and report ABI44 with the same layout fingerprint. No stale plugin binary ships. The nlohmann dependency is pinned and included in build identity, registry rank/device eligibility preserves chain semantics, and the shared source list is compiled consistently.

## Notes

Local ABI boundary correction (2026-11-19): ABI43 remains the exact frozen upstream input and the C9 wrong-ABI/layout check proves only that upstream loader baseline. RUM07 must append local KV layout fields to RadBuildCtx, RadKVGroupDecl and RadArchProbe. Those join the existing rad0018/rad0020 struct additions in one ABI44 bump; never mutate the exported shape while still reporting 43. All shipped plugin kinds must rebuild with ABI44 before the candidate can be packaged or released. This increases the final same-SDK build gate but avoids a second conflicting ABI bump.

RV4242 resolution (2026-11-19): preserve rad0018/rad0020 as their established ABI43 field extensions. Candidate-only rad0048_abi44_kv_contract runs after rad0020, advances ABI43 to ABI44 once, fingerprints ownership/pipeline fields plus KV layout, capability and prefill-mode fields, and includes an always-on Release CPU fixture. It remains untested and absent from recipe composition until RUM07 runtime wiring and the full same-SDK plugin/loader gate pass. Focused test: tools/tests/patch/test_radiance_rad0048.py.

User reaffirmation (2026-11-19): B16F8 and B16I8 are required legacy cache features for this bump. ABI44's canonical descriptor must preserve IDs, widths, offsets, and capability bits for both formats; it does not itself port or validate runtime support. Keep rad0048 as sole ABI/descriptor owner and make rad0040 a consumer, then rebuild every plugin against the one ABI44 fingerprint. RUM07 carries the complete B16 runtime and focused correctness gates.

Source-pin correction (2026-11-19): the v2.1.0 tag resolves to ABI40 commit 2a2b756371c26f2768b8d129a098df275ace2c73, but migration artifacts were composed from upstream main commit 3deabdaeca17b58bad90d332f4b204cf7bb8a10e (ABI43). The local upstream mirror confirms that exact commit is refs/heads/main. Pin the recipe and all component built-against identities to the full SHA; keep pinned-version = 2.1.0 for BigCherry release numbering. All six coupling abi-version declarations are 43 (unpatched source); the local composed plugin ABI44 gate remains separate. Focused recipe, engine-recipe, pin-bump and pin-triage checks pass (121 passed, 5 subtests). Canonical ordered apply/replay, ABI44 build and refusal fixtures still need a rerun under the corrected pin; GPU performance gates remain separate.

RUM03 targeted validation after the shared JSON dependency migration (2026-11-19): focused recipe/pin suite passes 121 tests with 5 subtests, including the updated full-SHA expectation. The serving component reconfigures successfully. Focused CPU-only target builds succeeded for bc_v2_p1_live_test, bc-release and bc_fake_engine; bc_v2_p1_live_test ran successfully. A combined multi-target build invocation exited nonzero after reporting 11/13 steps; rerunning bc_fake_engine alone succeeded, so treat that combined invocation as inconclusive rather than a code failure. Full component build, full CTest, canonical ordered patch replay and ABI44 loader fixture gates remain pending.

Follow-up targeted validation (2026-11-19): component coupling suite passes with updated ABI43 expectation (9 passed, 1 skipped, 10 subtests). ABI44 package replay/release-fixture tests pass 2/2 against the exact 3deabdaeca17b58bad90d332f4b204cf7bb8a10e checkout. Runtime-asset metadata tests pass 26/26 with 15 subtests. The ABI43 prefix/ABI44 extension fixture is no longer skipped when RADIANCE_SRC points to the frozen composition checkout.

Required helper-row compatibility finding (RV4247, 2026-11-19): the pinned ABI43 source omits had_quant_act_fp8 and moe_gemm_fp8a8 registry rows that components/moe-remote/moe_remote_gpu.h resolves by exact name. rad0046_retained_moe_rows restores both rows and keeps wide block-E4M3 tiling restricted to class/part boundaries. Added rad0046 to serving-core immediately before the ABI44 contract extension. Its exact-source apply/replay test passes 1/1. Recipe and ABI44 prefix tests currently refuse composition because package state is untested while production requires validated; this is the intended gate, not a reason to weaken recipe state checks. Build libr4d/moe-remote on ABI44, assert both rows resolve and run retained E4M3 guard fixture, then mark the patch validated and rerun the ordered composition/replay gates. GPU activation/numerical parity still belongs to RUM09.



C28 (2026-11-19): exact rad0046 ABI43 apply/replay passed; an isolated `R3_ONLY` libr3 plugin build with the restored quantizer and grouped-MoE HIP units succeeded, and a no-GPU dlopen probe confirmed both helper rows survive real `r3_registry` filtering with live launch callbacks. The existing `r3_rad46_rows_policy_test` is a separate dispatch policy test, not this patch's boundary guard. Current build source is ABI43, so rad0046 remains `untested` until the same build/row check runs against composed ABI44. Hardware/numerical activation is RUM09. Evidence C28; Ledger event `chg_20261119_122726_verified-that-both-restored-he_3927`.



C29 (2026-11-19): extracted the actual block-E4M3 width decision into a dependency-free constexpr helper used by the HIP launcher and exact-source C++ fixture. The fixture proves `{512,256}` at parts=2 selects wide, `{512,384}` falls back because one class splits at 192, and non-block-E4M3 formats preserve legacy selection. The 27-package production prefix through rad0048 composed from pinned SHA 3deabdaeca17b58bad90d332f4b204cf7bb8a10 at ABI44. The ABI44 KV/fingerprint fixture passed; narrow ABI44 libr3 plugin built and its row probe found both callbacks; native libr4d row/MoE objects compiled for gfx1201; ABI44 moe-remote worker and mailbox plugin built on the same composition/SDK. Package remains untested pending formal activation/correctness evidence and review; GPU/numerical qualification is RUM09. Evidence C29; Ledger event `chg_20261119_124601_made-the-e4m3-scale-boundary-r_8302`.

Codeberg main advanced by 40 commits from the frozen 3deabda candidate to Radiance 2.2.0 at b6b8751cccb0e834a13ae9e5afca824ee9a487aa. The new upstream header still declares ABI43. A separate upstream review ref holds that exact source. Recheck ordered apply/replay and semantic overlaps before changing the candidate pin or carrying forward compatibility evidence. Keep the existing ABI44 candidate available for comparison.

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
- 2026-11-19T02:48:21.302296+00:00 (updated-by): Updated: section:notes
- 2026-11-19T04:02:01.528340+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-19T04:20:06.571659+00:00 (updated-by): Updated: section:title
- RV4242
- 2026-11-19T04:26:05.437126+00:00 (updated-by): Updated: section:notes
- 2026-11-19T04:33:41.862009+00:00 (updated-by): Updated: section:notes
- RV4246
- 2026-11-19T07:57:07.207210+00:00 (updated-by): Updated: section:notes
- 2026-11-19T08:01:30.731011+00:00 (updated-by): Updated: section:notes
- 2026-11-19T08:06:37.521336+00:00 (updated-by): Updated: section:notes
- RV4247
- 2026-11-19T08:34:34.698229+00:00 (updated-by): Updated: section:notes
- 2026-11-19T12:28:01.492650+00:00 (updated-by): Updated: section:notes
- 2026-11-19T12:46:32.419552+00:00 (updated-by): Updated: section:notes
- 2026-11-19T21:53:59.253973+00:00 (updated-by): Updated: section:notes
