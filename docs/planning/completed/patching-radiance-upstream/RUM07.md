---
id: RUM07
order: 7
plan: patching-radiance-upstream
state: completed
created-at: '2026-11-18T01:28:34.138312+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P1
work: L
---

# Consolidate KV layouts on upstream dtype row sizing and retain only supported local formats

## Description

Compare local bf16_i8 with upstream I8 numerics, reuse matching kernels and keep justified hybrid formats in one descriptor table. Port rad0037/38/40/41 and required selftests without restoring removed structs or duplicating layout switches.

## Steps

- Stage 1 — Restore the deployed plain E4M3 H256 Flash-Next reader for Qwen4exp GQA12/page4 and GQA6/page4. Keep store, descriptor sizing, dense/QSA reader, split merge and shared MTP group together; do not alias it to native I8. Candidate rad0049 exact replay and registry constraints pass; its integrated architecture/row paths compile in the rad0051/rad0052 composition. Carry forward the existing plain-E4M3 behavior evidence; the final short regression comparison is RUM09.
- Stage 2 — Make rad0048 the sole local ABI44 owner after rad0020. It appends/fingerprints the layout and prefill-mode fields and defines the canonical format IDs/descriptor. Do not add a competing descriptor table in rad0040 or retain ABI43 struct shapes with local extensions.
- Stage 3 — Port B16 runtime plumbing as a consumer of rad0048: config/YAML, capability admission, typed RadBuildCtx/RadKVGroupDecl flow, descriptor-derived planner and allocation bytes, and binder shape. Candidate rad0050 passes exact replay and focused NDEBUG host tests, including 6144/6272 bytes per H256 page4 layer with two local heads.
- Stage 4 — Port B16F8 and B16I8 together: standalone/fused stores, prefill/decode and selected dense/QSA readers, split merge, and MTP. Use a typed compile-time layout selector; reuse existing byte V staging where semantically identical. Keep the DFlash2 drafter's native E4M3 layout separate. Candidate rad0051/rad0052 exact replay, host fixtures, required rows and selected gfx1201 objects pass. Reuse the established B16 behavior evidence.
- Stage 5 — Complete supported geometry and eligibility: Qwen4exp H256/GQA12/page4 and GQA6/page4, plus Qwen35 H256/GQA6/page16. Account for actual post-shard KV heads; reject unsupported tuples before allocation. Packed layouts without a writer warn and stay resident. Keep QSA block-key state separate and trunk/MTP on the same kv_attn group.
- Stage 6 — Restore remaining advertised formats fp8_v4 and k6_v4 through the same canonical descriptor where their historical paths are supported; preserve the independent attn_prefill_fp8 off/QK/PV/both control only for historical format/geometry rows. Reject combinations without rows. Do not silently remove options or change semantics.
- Stage 7 — Use existing BigCherry device selftest/A-B records as the feature-behavior baseline. For new-base differences, require exact apply/reapply, always-on host checks for bytes/codecs/geometry/rows, and selected gfx1201 plus historically supported gfx1100 object compiles. Run a narrow device selftest only when hardware is available; do not repeat long quality/performance qualification here.
- Stage 8 — After all retained formats/geometries pass applicable new-base compatibility gates, ABI44 plugin checks and plan review, mark packages validated using existing behavior evidence plus new-base port evidence, and add them to the bumped production recipe/serving-core composition in the same promotion change. Do not leave carried-forward BigCherry features candidate-only or in a permanently separate opt-in set.

## Detailed Solution & Technical Design



## Code Samples & Guidance

Use descriptor-derived names, row16, scale group and capability checks at the config, builder and plugin boundaries. Kernel instantiations take a typed compile-time KV layout tag whose default is native; do not reuse legacy B16 option bit values because one aliases ABI43 O_I8. For the runtime extent, select rad_kv_group_row_elems(layout, native_dtype, logical_head_dim); native groups delegate to rad_kv_row_elems. The hybrid descriptor supplies exact padded U8 row bytes and row strides to both the planner and libr4d operand shapes. Example invariants at H256: B16F8 row16=48 and 768 bytes/head; B16I8 row16=49 and 784 bytes/head with scale offset 768; native E4M3 is 512 bytes/head. Unsupported geometry and B16 streaming return explicit errors before pool allocation.

## Files

- engines/radiance/patches/rad0037_bf16_fp8_kv/patch.py
- engines/radiance/patches/rad0038_bf16_i8_kv/patch.py
- engines/radiance/patches/rad0040_kv_layout_table/patch.py
- engines/radiance/patches/rad0041_kv_b16_flashnext/patch.py
- engines/radiance/patches/rad0042_kv_b16_selftest/patch.py
- engines/radiance/components/serving/contract/serving-v2.toml
- abi/rad_builder.h
- abi/rad_abi_layout_rows.h
- core/build/rad_builder.cpp
- core/mem/kv.cpp
- core/runtime/kvbind.cpp
- core/config.cpp
- core/engine_bringup.cpp
- arch/qwen35_fp8/qwen35_fp8.cpp
- arch/qwen4exp_fp8/qwen4exp_fp8.cpp
- libr4d/r4d_shapes.cpp
- libr4d/r4d_rows.cpp
- libr4d/r4d_plugin.h
- libr4d/r4d_attn_prefill_h256_gqa6.hip
- libr4d/r4d_attn_decode_h256_gqa6.hip
- libr4d/r4d_attn_paged_h256_gqa6.hip
- libr4d/r4d_attn_paged_gqa8.hip
- libr4d/r4d_model_bf16.hip
- libr4d/r4d_qk_norm_rope_gate.hip
- libr4d/r4d_selftest.cpp

## Validation

- Always-on CPU checks cover descriptor IDs/names/row16/offsets, supported geometry tuples, post-shard refusal, logical head dimension, local-head/page byte formulas, planner/binder first+last pages and MTP-layer binding, and packed stream-request warning plus resident fallback. No assert-only correctness gate.
- ABI44 fixture proves added KV fields and existing ownership/pipelining fields are in the version and fingerprint. ABI43 plugin artifacts refuse against ABI44; every shipped plugin is rebuilt and passes the exact ABI44 layout check under the shared RUM03 SDK gate.
- Exact-ref patch apply and byte-identical reapply pass for the selected candidate composition; r3_rewrite.py --check passes for supported gfx1100 rows.
- Compile selected gfx1201 device objects and historically supported gfx1100 rewrite/build rows. These checks establish source/template/resource compatibility only, not numerical device execution.
- Reuse historical BigCherry selftest, codec, KLD, serving and performance records under rad0037/38/40/41/42 for behavior already proven on the old base. Compare the new typed/tagged attachment to those semantics. Add always-on host vectors where the port changes codec or staging code. Run the narrow existing device selftest only when the target hardware is available; do not require repeating long A/B/KLD/performance runs to establish a source-port claim.
- At H256 assert row sizes V4=400, K6V4=352, B16F8=768, B16I8=784 bytes/head; native E4M3=512, native I8=516. For two local heads and page4 assert layer payloads 3200, 2816, 6144, 6272 and 4096 bytes respectively. Prove no custom option falls through to native I8. Until a format-aware streamed writer exists, require warning and resident fallback. RUM09 owns the final short paired regression/performance gate.

- Candidate `rad0053_v4_kv_layouts` restores resident `fp8_v4` and `k6_v4` for Qwen4exp H256/GQA12/page4 and H256/GQA6/page4 and Qwen35 H256/GQA6/page16. The 63 exact edits applied and reapplied byte-identically over `3deabdaeca17b58bad90d332f4b204cf7bb8a10e` and its dependencies. The existing Release `bc_b16_kv_runtime_test` fixture covers CLI/YAML, descriptor-derived 400/352-byte rows and 3200/2816 bytes per layer for two local heads and page4, and resident allocation/binding. Release architecture fixtures passed Qwen4exp GQA6 (46 checks), Qwen4exp GQA12 and shared MTP/QSA (216), native Qwen4exp E4M3 (170), and Qwen35 with isolated native E4M3 DFlash2 (100). The selected six `gfx1201` objects compiled over the completed package chain.
- The device codec body comparison against historical v1.17.0 commit `19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d` matched the tested FP16 scale rounding, clamps, nonfinite treatment, and K6 low/high bits after descriptor and typed-layout substitution. The packed K6 bit positions matched both writers and both reader sites. A host check reconstructed all 64 six-bit codes in every H256 lane. The exact-replay fixture pins the historical standalone reciprocal in double before rounding to float and the fused reciprocal in float after rounding the scale.
- Candidate `rad0054_attn_prefill_fp8` restores independent `off`, `qk`, `pv`, and `both`. It enables lossy prefill only for the historical H256/GQA6/page16 Qwen35 E4M3, V4 and K6V4 cache rows and leaves decode and drafter attention exact. The 24 edits applied and reapplied byte-identically after tightening the config insertion to the typed option-catalog row. Release fixtures passed CLI modes (15 checks), native YAML parsing, and Qwen35 supported mode and cache declarations. The combined package and registry suite passed 57 tests, including all nine lossy row constraints. The `gfx1201` page16 attention and registration objects compiled. These are new-base compatibility checks. Historical characterization remains the feature evidence. Integrated new-base startup and serving qualification and production recipe selection remain pending.

- The final Release Qwen35 precision fixture passed 99 checks, including refusal before KV group declaration for H128 and refusal of B16I8 with lossy prefill. The native E4M3 DFlash2 isolation fixture passed 100 checks in the same composition.

- `r3_rewrite.py --check` against the exact frozen `3deabdaeca17b58bad90d332f4b204cf7bb8a10e` composition through rad0054 passed with the global rewrite rules (`failures: {}`, exit 0). The default `R3_NATIVE=ON` path failed with the component's native blocks (exit 1). In `engines/radiance/components/libr3/native/r4d_attn_paged_h256_gqa6.hip.rw`, blocks 2 and 4 match old attention template signatures and already fail through rad0052, before the V4/prefill port. Block 3 additionally misses the rad0054 `CAN_SPLIT` guard that keeps lossy FP8 prefill unsplit. The default gfx1100 rewrite gate remains open until the native blocks forward the typed layout argument and preserve this split guard. No GPU job ran.

- The default libr3 native rewrite gate is resolved for the frozen ABI44 composition through rad0054. Common and `--native` `r3_rewrite.py --check` returned exit 0 and empty failures through rad0052 and rad0054. The regression verifies all four native/fallback, unsplit/split launches retain `0, Layout` and the QK/PV guard excludes lossy products from splitting. The changed generated H256/GQA6/page16 attention unit compiled for `gfx1100`. Exact v1.17 production `serving-core` replay applied 474 edits with zero failed files. Both native generations with the original production component and with only the new attention rule substituted emitted a byte-identical unit (SHA-256 `e293add9107f46684ae9f57808ae08146bdaa572464c38826729c8a8ba76cda2`). The 57-test focused patch and registry suite passed. No GPU job ran.

## Effort & Risk

L: equal row bytes do not imply equivalent scale/rounding semantics. Compare a small supported-format set, retain one descriptor source and refuse unqualified shapes rather than broadening the kernel matrix.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

The bumped production source includes plain E4M3 and every retained BigCherry custom cache format (fp8_v4, k6_v4, bf16_fp8, bf16_i8), plus the independent attn_prefill_fp8 QK/PV/combined control. Their local ABI44 fields and fingerprint are consistent; all plugin kinds rebuild against the same version/layout. One descriptor drives config, model eligibility, planning/allocation/transfer/binding, row constraints, store, and dense/QSA reader. Historical bytes and numerics match for Qwen35 H256/GQA6/page16 and Qwen4exp H256/GQA12/page4 plus GQA6/page4. Unsupported tuples refuse before allocation; packed stream requests preserve warning-plus-resident fallback. QSA storage remains separate, trunk/MTP share kv_attn, and Qwen35 drafter cache remains independent. Always-on host checks, exact replay and selected target compiles pass; existing BigCherry device/quality/performance evidence is carried forward for unchanged semantics, and any newly changed codec has focused available coverage. Every carried-forward feature package reaches validated state and is selected by the bumped production source; no feature is silently aliased, dropped, left candidate-only, or advertised without coherent supported behavior.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM03, RUM06. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM03/RUM06. The 166 failed-edit review surface is partly cascading; don't repair anchors one by one before choosing the unified KV contract.

Astra source review ASTRA-CONSOLIDATION.md: evaluate this slice early and independently where possible, before porting obsolete hunks. Same byte count does not establish equivalent offsets/scales/rounding. Integrated qualification depends on selected ABI/placement/runtime.

Production-format implementation blocker: target's native FP8 opt-in is absent on Qwen4exp and native i8 has different scale bytes; a copied Flash-Next YAML is not sufficient. Resolve local E4M3 retention or explicitly qualify a replacement before RUM09 real production canary. Refer to docs/reviews/radiance-upstream-2026-11-18/IMPLEMENTATION-HANDOFF.md.

ABI43 port assessment (2026-11-19): keep the current appended rad0040/rad0042 descriptor-only edits out of the candidate composition. Their arithmetic is correct (head_dim 256: B16F8=768 B/head, B16I8=784 B/head; plain E4M3=512 B/head, native I8=516 B/head), but the fixture is assert-only under NDEBUG and no runtime callers connect the table to allocation or kernels. The first integrated candidate slice is resident head_dim=256, GQA=12, page=4. Carry the hybrid format distinctly through native config advice/parser and argv/engine admission; don't map it silently to BF16/I8 or change ABI sizes unnoticed. Keep logical head_dim 256 separate from physical page row bytes in planning, group/page allocation and binding. Port standalone store with prefill/decode readers as one slice, retain MTP shared KV group, use existing unfused resident attention where valid, and reject streaming until mirror/ring transfer is ported. QSA block-key storage remains separate. Add always-on refusal and byte/shape checks, then compare store bytes plus dense, QSA, MTP draft/verify output against reference before adding GQA6 or page16. Plain fp8 parity remains separate.

ABI43 scope correction after RV4238: the earlier resident B16F8/B16I8 slice is not the first production slice. The selected Flash-Next RUM09 path resolves plain E4M3 by default and cannot be replaced with native I8. Implement and qualify the exact H256/GQA/page E4M3 reader plus store, planning, binding and MTP sharing first. The B16 hybrids remain deferred optional formats; do not use the current descriptor-only scratch as accepted runtime work. The exact source triage is anchored to 3deabdaeca17b58bad90d332f4b204cf7bb8a10e, and all row/probe claims must be checked against that commit.

2026-11-19 user direction supersedes prior deferral language: B16F8 and B16I8 are BigCherry cache features and must be migrated in this bump. Deliver plain E4M3 first for the selected deployment, then finish both hybrid cache formats as an integrated, separately gated stage before RUM07 completion. Preserve the dirty rad0040/rad0042 work and inspect/reuse its useful descriptor/test structure without treating it as runtime implementation. Sol is reviewing the ABI43 format contract and exact migration order.

User direction makes B16F8 and B16I8 required before RUM07 completes. Implement plain E4M3 for the selected deployment first, then complete both historical resident hybrid formats before this item can close. Sol ABI43 source review: supported B16 tuples are Qwen35 H256/GQA6/page16 and Qwen4exp H256/GQA12/page4 plus GQA6/page4; B16 streaming is historically refused. The prior TP1-only descriptor scratch is not the final geometry policy. Keep the dirty rad0040/rad0042 work intact until its useful arithmetic and fixtures are integrated and replaced by always-on/runtime coverage. Candidate source is frozen at ABI43 commit 3deabdaeca17b58bad90d332f4b204cf7bb8a10e. RUM03/RUM06 remain prerequisites; RUM09 owns the short integrated quality/performance gate. Do not restore removed structs or introduce a second global descriptor registry.

User direction requires all BigCherry cache features to migrate in this bump. In addition to required B16F8/B16I8, serving config still advertises fp8_v4 and k6_v4 although ABI43's config parser rejects them; keep them through the descriptor/kernel path or obtain explicit retirement approval before removal. Implement deployed plain E4M3 first, then all advertised packed layouts before RUM07 completes. Sol ABI43 source review confirms resident tuples: Qwen35 H256/GQA6/page16 and Qwen4exp H256/GQA12/page4 plus GQA6/page4. Qwen4exp GQA12 has 2 local KV heads at TP1, 1 at TP2. B16 streaming is historically refused. The prior TP1-only descriptor scratch is not final geometry policy. Preserve dirty rad0040/rad0042 until useful arithmetic/fixtures are integrated and replaced with always-on/runtime coverage. Candidate is frozen at ABI43 commit 3deabdaeca17b58bad90d332f4b204cf7bb8a10e. RUM03/RUM06 remain prerequisites; RUM09 owns short paired quality/performance qualification. No new global registry or restored removed structs.

User direction requires every BigCherry cache feature still advertised to migrate in this bump. In addition to required B16F8/B16I8, serving config still advertises fp8_v4 and k6_v4 although ABI43's parser rejects them; retain all four through the descriptor/kernel path or obtain explicit approval to retire. The independent attn_prefill_fp8 QK/PV control is also advertised and must remain separate. Implement deployed plain E4M3 first, then all advertised packed layouts and prefill controls before RUM07 completes. Sol and GPT reviews agree on resident tuples: Qwen35 H256/GQA6/page16 and Qwen4exp H256/GQA12/page4 plus GQA6/page4; GQA6 page sizes differ. Qwen4exp GQA12 has 2 local KV heads at TP1 and 1 at TP2. Preserve legacy packed-stream warning and resident fallback until an actual custom writer exists. The prior TP1-only descriptor scratch is not final geometry policy. Preserve dirty rad0040/rad0042 until useful arithmetic/fixtures are integrated and replaced with always-on/runtime coverage. Candidate is frozen at ABI43 commit 3deabdaeca17b58bad90d332f4b204cf7bb8a10e. RUM03/RUM06 remain prerequisites; RUM09 owns short paired quality/performance qualification. No new global registry or restored removed structs.

The frozen upstream candidate stays ABI43. Any local exported-field changes for KV layout, capability, ownership or pipelining are composed into one ABI44 bump under RUM03, including exact version and fingerprint checks plus same-SDK rebuild of all plugins. Do not run/cache or package ABI43 plugins against the extended layout.

RV4241 correction (2026-11-19): attn_prefill_fp8 is a required BigCherry feature. The v1.17.0 baseline accepts off/qk/pv/both; the ABI43 candidate dropped its config control while retaining the prefill_fp8 row input. Restore and route the exact modes. Historical lossy rows are only H256/GQA6/page16 for plain E4M3, fp8_v4 and k6_v4 (nine combinations); page4 and B16 combinations have no historical lossy row and must refuse before allocation. SERVING-OPTION-INVENTORY.md records this correction.

ABI44 contract scaffold (2026-11-19): rad0048_abi44_kv_contract owns the single post-rad0020 ABI bump and carries independent KV layout plus attn_prefill_fp8 ABI fields. Its descriptor covers all four packed IDs and H256 row/offset formulas; exact-source replay and always-on Release fixture pass. It does not implement runtime config, capability, planning, binding, store/read or attention dispatch. Keep the package unselected until those RUM07 stages and RUM03 same-SDK loader checks pass.

User reaffirmation (2026-11-19): B16F8 (`bf16_fp8`) and B16I8 (`bf16_i8`) are mandatory carried-forward features, not candidates for retirement. Preserve both alongside every other advertised BigCherry cache mode until their complete runtime paths and focused gates pass. The current candidate work has only ABI44 descriptor scaffolding; old rad0037/rad0038/rad0040/rad0041/rad0042 packages are not evidence of successful migration because the exact candidate composition still has replay failures.

Descriptor ownership clarification: candidate-only `rad0048_abi44_kv_contract` owns the ABI44 KV format IDs, canonical descriptor, row/offset/byte formulas, capability fields and ABI fixture. `rad0040_kv_layout_table` must consume that descriptor and own runtime config/admission/planning/binding/resident policy; it must not define a second `RAD_KV_LAYOUTS` table. Then rebase rad0037/rad0038 codecs and typed kernel tags on the ABI44 interfaces, followed by rad0041 Qwen4exp geometry/rows and rad0042 focused contract/device tests. Update recipe membership and package dependencies as one change only after every package exact-applies and re-applies in order. B16 acceptance covers Qwen35 H256/GQA6/page16 and Qwen4exp H256/GQA12/page4 plus GQA6/page4, exact standalone/fused storage bytes, dense and QSA reads, MTP draft/verify, unsupported-geometry refusal, and packed-cache resident fallback for streaming requests. Keep the staged small test gates; RUM09 owns deployment quality/performance.

Sol implementation sequencing (2026-11-19): candidate graph is rad0018 → rad0020 → rad0048 ABI44 → rad0049 plain E4M3 → B16 core plumbing → paired B16 store/read plus Qwen35/Qwen4exp routes → focused selftests and release qualification. Keep the initial core plumbing capability bits disabled until a coherent matching-layout store and selected reader/fused route are present. The smallest CPU integration gate checks native CLI/YAML selection, model admission, RadBuildCtx→kv_attn declaration propagation, descriptor-derived bytes through planner/fallback/binder, invalid tuple refusal, and unchanged native FP8/I8. The first working vertical slice is resident B16F8 and B16I8 for H256/GQA12/page4 with MTP; then cover Qwen4exp GQA6/page4 and Qwen35 GQA6/page16, both output epilogues, and focused device behavior. Keep the old production recipe untouched until candidate exact replay and gates pass.

Sol pre-port overlap findings (2026-11-19): rad0049 already changes the composed candidate's paged GQA8 wrapper, QK-norm/rope/gate store dispatch, plugin declarations/rows, and Qwen4exp geometry/probe/tests. Any B16 dispatch patch must require rad0049 and rad0050 and anchor on their composed output, not the frozen ABI43 snippets. In each trunk group declaration, set `RadKVGroupDecl.kv_layout` from `RadBuildCtx.kv_layout`, preserve the descriptor option spelling in geometry, and derive opaque U8 storage extent in `r4d_shapes.cpp`. Thread a typed layout selector through prefill/decode fetch and staging plus split/chained wrappers; do not reuse the legacy B16 bit. Qwen35's DFlash2 drafter is a separate cache: pass a local context with layout NONE and preserve its native E4M3 contract. ABI43 removed relevant H128 DFlash2 E4M3 reader rows, so restore/verify those narrowly before claiming Qwen35 B16 support; never silently convert the drafter to BF16/I8. Keep B16 probe capability clear until these group, geometry, store, read and row paths are all present.

Sol kernel-port hazards (2026-11-19): shared `rad_block_attn_gated_fp8.h` can declare streamed fused/store ops because schemas exist even when the selected cache is resident, leaving a required device row hole. Packed B16 trunk declarations must omit both streamed ops and resident-only QSA resolve declarations unless an exact matching row is added; preserve unrelated E4M3/native modes. `libref` generic attention/store code treats RAD_U8 as numeric elements, so explicitly refuse unsupported B16 host-reference dispatch until a real packed reference is implemented. Qwen35 DFlash2's qualified tuple is exactly H128/GQA4/page16 with width≤16, noncausal window path; do not broaden it to generic H128 windows.

Reuse refinement from Sol's ABI43 kernel inspection (2026-11-19): ABI43 already has byte-code V staging and paired FP16 widening for native I8 in `fetchV4/storeV4`. For B16, add a separate typed compile-time `KI` selector; use `V4 || KI` for the existing stage, `I8 || KI` for byte-code fetch, and select B16's per-group scale only in the `KI` branch. Keep native-I8 K handling unchanged and use BF16 K handling for B16. This reuses the existing staged-byte path without copying loops or aliasing native flag bits.

Implementation decisions and status (2026-11-19)

- This work depends on RUM03 (one ABI44 extension) and RUM06 (safe runtime ownership). The frozen upstream source remains ABI43 at 3deabdaeca17b58bad90d332f4b204cf7bb8a10.
- BigCherry cache features are required migration scope. Preserve plain E4M3, fp8_v4, k6_v4, bf16_fp8 (B16F8), bf16_i8 (B16I8), and the independent attn_prefill_fp8 off/qk/pv/both control. No advertised option may silently alias native I8/BF16 or disappear; unsupported format/geometry/product combinations must refuse before allocation.
- Candidate order: rad0018 → rad0020 → rad0048 ABI44 descriptor/ABI owner → rad0049 native E4M3 slice → rad0050 B16 runtime plumbing → paired B16 kernel/model dispatch → focused selftests. Keep all new candidate packages out of recipes until ordered apply and byte-identical reapply, targeted gates, and same-SDK plugin checks pass. Do not change the old production recipe during candidate work.
- rad0048 is the sole owner of ABI44 fields, packed format IDs, canonical descriptor, byte/offset formulas and layout fingerprint. rad0050 consumes those fields for CLI/YAML, admission, resident planning/allocation/binding, and stream fallback. Do not define a second RAD_KV_LAYOUTS table. rad0049 is candidate-only and restores Flash-Next native E4M3 H256/GQA12/page4 and GQA6/page4 rows/stores; exact apply/reapply and registry constraints passed, while host executable linking, device compilation and numeric qualification remain open.
- rad0050 is candidate-only. Exact apply/reapply passed; focused tests passed 2/2 and its NDEBUG host CTest passed 1/1. It verified 6144/6272 bytes per H256 page4 layer with two local KV heads, invalid-geometry refusal, unchanged native E4M3/I8 sizing and warning-plus-resident fallback. B16 capability bits remain clear pending matching kernels.
- Required B16 tuples: Qwen4exp H256/GQA12/page4 and GQA6/page4; Qwen35 H256/GQA6/page16. Qwen4exp's trunk and MTP share kv_attn; its QSA key cache stays separate. Qwen35's DFlash2 cache is separate and stays native E4M3: isolate it with layout NONE and restore only its qualified H128/GQA4/page16, width≤16 noncausal reader if required. Do not inherit U8/B16 or switch its dtype silently.
- Port the standalone and fused B16 codecs together with their actual selected dense/QSA readers, split merge, and draft/verify paths. Use a distinct typed compile-time layout tag; legacy O_KB16 bits conflict with ABI43 flags. Reuse ABI43 byte-code V staging and FP16 widening where exact; preserve B16I8 group-32 scales and rounding. Skip streamed writer declarations for packed layouts and retain warning-plus-resident fallback. libref generic numeric U8 paths must refuse until matching packed reference behavior exists.
- Keep small gates staged: config/YAML and descriptor contract; model/probe and geometry refusal; planner/fallback/binder bytes; row and symbol constraints; targeted gfx1201 compilation/device cases; then RUM09's short quality/performance qualification. Do not broaden unqualified geometries or run a broad performance suite during the port.

Execution authority: the Implementation decisions and status block immediately above supersedes every earlier planning note in this file wherever they conflict. Older notes are retained for traceability; use the current Steps and Detailed Solution sections as the active implementation contract.

rad0051 first vertical gate (2026-11-19): exact apply/reapply and required-row host tests pass; the Qwen4exp H256/GQA12/page4 model fixture confirms B16F8/B16I8 declarations, separate BF16 QSA cache, shared MTP `kv_attn`, and no unsupported stream/QSA-resolve ops. Release gfx1201 compilation passed for selected GQA12 wrapper, standalone/fused writers, registry and shapes. Shape callback fixture confirms row extents native E4M3=512, native I8=516, B16F8=768, B16I8=784. This is compile and host/source evidence, not device execution or numerical parity. GQA6/page4 and Qwen35 GQA6/page16 plus H128 DFlash2 native E4M3 isolation are still required. rad0051 remains candidate-only and unselected.

User correction (2026-11-19): treat B16F8/B16I8, V4/K6 and prefill-FP8 as tested, existing BigCherry features on the old Radiance base. This bump's task is to port their behavior to the new APIs, not to requalify or drop them. Reuse the historical device selftest, byte-codec, KLD and serving evidence owned by rad0037/38/40/41/42; test only the changed integration/attachment surface needed to establish compatibility on the new base (exact patch replay, ABI/layout fingerprint, host declarations/sizing/rows, selected target object compile, and focused parity vectors for changed codec/staging code). The final full performance comparison remains RUM09 as already planned. Once those port gates pass, promote every retained feature package into the bumped serving-core production recipe together; do not leave it permanently candidate-only or imply the old feature is untested just because the new-base adaptation is being checked.

New-base migration completed (2026-11-19): the task branch now targets Radiance v2.1.0 / ABI44 source commit 3deabdaeca17b58bad90d332f4b204cf7bb8a10. The ordered production recipe carries rad0048 through rad0054 together, all seven packages are `validated`, and the old rad0037/rad0038/rad0040/rad0041/rad0042 experiment-only chain and empty MTP replay set are no longer selected. Historical old-base behavior evidence remains the feature evidence; new-base compatibility is supported by exact ordered apply/replay, focused runtime/geometry/codec/prefill fixtures, selected object builds, full component build, ABI44 checks, and component tests. Production closure resolves to 32 validated packages; patch-lint passes. Latest `origin/main` through 31bd5b9e9 is merged, and the new stored-manifest/runtime-receipt tests pass 3/3. No GPU numerical or performance run was performed; RUM09 owns the final paired comparison and runtime gates.

## Ownership and order

Use the selected source commit 3deabdaeca17b58bad90d332f4b204cf7bb8a10 as frozen ABI43 input. Order candidate work as rad0018 → rad0020 → rad0048 → rad0049 → rad0050 → rad0051 → focused selftests. rad0048 is the only owner of the ABI44 append and fingerprint plus the packed format IDs/descriptor; rad0050 consumes it for runtime plumbing. Do not make rad0040 a second descriptor owner. Keep all candidate packages out of production recipes until ordered apply/reapply, focused gates and same-SDK plugin checks pass.

## Layout data and eligibility

Keep plain native E4M3 separate from packed formats. The canonical packed IDs are fp8_v4=1 (row16 25; 400 bytes/head at H256), k6_v4=2 (row16 22; 352 bytes/head), bf16_fp8=3 (row16 48; 768 bytes/head), bf16_i8=4 (row16 49; scale group 32; 784 bytes/head). Native E4M3 is 512 bytes/head; native I8 is 516. Custom physical storage is RAD_U8 while logical head_dim remains 256. Use one descriptor-derived row extent in grouping, planner sizing, fallback allocation, binding and HIP shapes; do not duplicate byte arithmetic. For H256, two local KV heads and page4, payloads are E4M3=4096, V4=3200, K6V4=2816, B16F8=6144, B16I8=6272 bytes per layer before alignment.

Only support measured tuples: Qwen35 FP8 H256/GQA6/page16; Qwen4exp H256/GQA12/page4 and GQA6/page4. Use post-shard GQA and actual local KV-head count. Qwen4exp GQA12 has two local KV heads at TP1 and one at TP2. Reject unsupported layout/model/head/page combinations before pool allocation. Packed layouts have no streamed writer: warn and keep the group resident. Keep Qwen4exp QSA block-key storage separate, and preserve one shared trunk/MTP kv_attn group.

## B16 kernels and model routing

Port B16F8 and B16I8 as a pair. B16F8 stores 512 BF16 K bytes followed by 256 E4M3 V bytes at unit scale with finite saturation at ±448. B16I8 stores the same K, then 256 signed V bytes and eight FP16 scales at offset 768. Each scale covers 32 values, uses upward-representable absmax/127, floor 2^-14 for nonzero groups, cap 515.5, rintf, and clamp [-127,127]; zero groups store zero scales and codes. Preserve the old nonfinite policy and byte exactness between standalone and fused stores. Do not equate equal row sizes with equal numerics.

Use a distinct typed compile-time layout tag; legacy O_KB16 option bits collide with current ABI flags. ABI43 already has byte V staging and FP16 widening used by I8; reuse it where exact, with B16-specific group scale selection and a separate BF16 K branch. Thread the layout through prefill/decode, fused gated-store and split/chained wrappers without per-element runtime dispatch. Register only rows matching layout, dtype, geometry and page. Add capability bits only after that architecture has its complete declaration, store, reader and row set.

Qwen4exp supports H256/GQA12/page4 and GQA6/page4, with MTP using trunk kv_attn and QSA remaining separate. Qwen35 supports H256/GQA6/page16 and its DFlash2 drafter remains a separate native E4M3 cache. The qualified drafter tuple is H128/GQA4/page16, width≤16 and noncausal/window-capable; isolate the drafter context with kv_layout NONE and restore only the required native E4M3 reader row if absent. Do not let the trunk B16 layout leak into the drafter or silently change the drafter dtype.

Shared gated-attention schemas may declare streamed ops even when the packed mode cannot run them; omit packed stream and QSA-resolve declarations unless a matching implementation/row exists, preventing device holes. Generic libref numeric U8 paths are not packed codecs; refuse B16 host-reference dispatch until matching reference semantics exist.

## Precision modes and gates

Restore attn_prefill_fp8 independently from cache selection. Preserve off, qk, pv and both and only accept historical rows for the exact layout and geometry. Historical lossy rows exist only at H256/GQA6/page16 for plain E4M3, fp8_v4 and k6_v4; page4 and B16 product combinations have no historical lossy row and must refuse before allocation.

Use always-on CPU checks for descriptor fields, names, row widths/offsets, page bytes, geometry refusal, YAML/CLI roundtrip, model admission, planner/fallback/binder and native-layout regression. Exact replay must pass on frozen source; verify required registry symbols and r3_rewrite coverage. Targeted device checks cover standalone/fused bytes, zero/tiny/tie/saturation values, dense and QSA attention, split merge, checkpoint spans, MTP draft/verify, required launch rows, and selected gfx1201/gfx1100 objects. RUM09 owns the short model-quality/performance comparison. Avoid a broad performance suite during porting.

## Change Log

- 2026-11-18T01:28:34.138312+00:00 (created-by): Created by codex
- 2026-11-18T01:39:53.057270+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:26.892039+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:57:05.367589+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:19:59.088702+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-11-18T02:23:39.101657+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:26:37.220559+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:28:24.920139+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T03:41:48.679183+00:00 (updated-by): Updated: section:notes
- 2026-11-19T01:28:36.142111+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4238
- 2026-11-19T02:22:01.408878+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:validation, section:acceptance_criteria
- 2026-11-19T02:23:43.500330+00:00 (updated-by): Updated: section:notes
- 2026-11-19T03:23:02.448475+00:00 (updated-by): Updated: section:steps
- 2026-11-19T03:23:19.121543+00:00 (updated-by): Updated: section:notes
- 2026-11-19T03:23:35.587274+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-11-19T03:36:20.630843+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-19T03:37:18.503106+00:00 (updated-by): Updated: section:detailed_solution, section:validation
- 2026-11-19T03:38:29.355058+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:validation, section:acceptance_criteria, section:notes
- 2026-11-19T03:39:12.923328+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria
- 2026-11-19T03:56:06.476909+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:validation, section:acceptance_criteria, section:notes
- 2026-11-19T04:02:36.192409+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
- RV4241
- 2026-11-19T04:14:09.066477+00:00 (updated-by): Updated: section:notes
- 2026-11-19T04:25:59.053482+00:00 (updated-by): Updated: section:notes
- 2026-11-19T04:33:16.952350+00:00 (updated-by): Updated: section:notes
- 2026-11-19T04:35:18.466372+00:00 (updated-by): Updated: section:notes
- 2026-11-19T04:38:18.111270+00:00 (state-transition): State: pending → in_progress
- 2026-11-19T04:42:54.218462+00:00 (updated-by): Updated: section:notes
- 2026-11-19T04:44:47.078673+00:00 (updated-by): Updated: section:notes
- 2026-11-19T04:45:36.699443+00:00 (updated-by): Updated: section:notes
- 2026-11-19T04:46:24.867580+00:00 (updated-by): Updated: section:validation
- 2026-11-19T04:48:33.461984+00:00 (updated-by): Updated: section:notes
- 2026-11-19T04:49:46.417860+00:00 (updated-by): Updated: section:steps
- 2026-11-19T04:50:28.355200+00:00 (updated-by): Updated: section:detailed_solution
- 2026-11-19T04:52:18.466432+00:00 (updated-by): Updated: section:notes
- 2026-11-19T04:59:38.248102+00:00 (updated-by): Updated: section:notes
- 2026-11-19T05:05:51.011370+00:00 (updated-by): Updated: section:steps, section:acceptance_criteria
- 2026-11-19T05:10:47.319390+00:00 (updated-by): Updated: section:validation
- 2026-11-19T05:11:08.739583+00:00 (updated-by): Updated: section:notes
- 2026-11-19T05:11:58.494782+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria
- 2026-11-19T05:12:58.524411+00:00 (updated-by): Updated: section:steps
- 2026-11-19T05:36:49.852678+00:00 (updated-by): Updated: section:validation
- 2026-11-19T05:37:58.145044+00:00 (updated-by): Updated: section:validation
- 2026-11-19T05:53:54.253241+00:00 (updated-by): Updated: section:validation
- RV4244
- 2026-11-19T05:59:50.084969+00:00 (updated-by): Updated: section:validation
- 2026-11-19T07:31:11.696666+00:00 (updated-by): Updated: section:notes
- 2026-11-19T07:35:34.813970+00:00 (state-transition): State: in_progress → completed
