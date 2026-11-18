---
id: VTP44
order: 0
plan: run-dense-27b-virtual-tp
state: in_progress
created-at: '2026-11-16T07:45:37.732759+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Generic KV layout descriptors and geometry rows; Flash-Next as second model for bf16_fp8/bf16_i8

## Description

rad0037/rad0038 (VTP43) added the bf16_fp8 and bf16_i8 KV caches for one geometry (head_dim 256, gqa 6, block 16) by hand-writing rows, per-plugin name/row-size lists and libr3 rewrite blocks for that geometry. Flash-Next (qwen4exp_fp8: gqa 12 at one and two ranks, gqa 6 with block 4 at four ranks, and fused gate+quantiser decode rows) needs the same layouts at other geometries and is the second validation model for promoting rad0037/38. Owner rule: the engine stays generic, no model-specific constants or per-model copies, enhancements reusable for newer models. So first make the layout and the rows data-driven, then add Flash-Next's geometry as data plus its own kernel instantiations.

## Steps

1. Patch A, the refactor (no new geometry; its rad id is taken from the next free one when created, since rad0039 is the heat observer, #942): a KV-layout descriptor table in abi/rad_builder.h (name, exact storage rules: K/V element kinds, offsets, scale placement, group, row bytes as an integral-checked function of head_dim, alignment) read by the core, both arch plugins, probe, bringup, streaming refusal and the serving enum generator; per-plugin name and row-size lists and the `gqa == 6` guards removed; row declarations for the existing gqa6 rows generated from an explicit list of (op, layout, head_dim, gqa, block, tuning) tuples. Behaviour byte-identical: row-table dump diff, option-bit diff, selftests, 27B KLD equal.
2. Patch B: gqa12 rows (prefill, decode, scratch) and the attn_paged_gate_quant rows for b16f8/b16i8 as new tuples with their own tuning, qwen4exp_fp8 plumbing, libr3 rewrites at gqa12 and block-4 as per-template rules with count checks plus a post-rewrite instantiation inventory.
3. libr3: replace per-geometry rewrite blocks with per-template rules where the build allows, else keep exact-count blocks; add an inventory check that expected and produced gfx1100 instantiations match.
4. Flash-Next validation: flashnext-tp2xtx-r9700 (two 7900 XTX TP2, R9700 remote-MoE helper) with kv_cache_dtype bf16_i8 / bf16_fp8 vs its bf16 reference; KLD, speed A/B, identity check (default cache identical with and without rad0037/38 and patches A and B).
5. Lightweight promotion (QFP18 tier, profile evidence) of rad0037/38 (and patches A and B if they stand alone) with Qwen3.8-27B and Flash-Next as the two named models.

## Detailed Solution & Technical Design

GPT design review 1 (docs/evidence/2026-11-16-vtp44-generic-kv-layouts/design-review-1.md). Accepted: (1) The descriptor defines exact storage semantics (offsets, alignment, scale placement, packing order, padding) as declarative data and the table builder rejects any supported geometry whose row is not integral and aligned; no function pointers in the ABI table. (2) Geometry-specific tuning stays explicit: a generated row takes its option bits and instantiation from a per-tuple entry, so a gqa12 row never inherits gqa6 tile size or LDS skew; only row-construction logic is shared. (3) libr3: generate or rewrite per template with exact counts and a mandatory post-rewrite audit comparing expected and produced instantiations, because a broad rule can compile while a row is silently absent; the `@@ find N or 0` pair form is audited for the same reason. (4) The fused gate+quantiser is its own operation kind (attn_paged_gate_quant) consuming a layout descriptor with its own tuple set; only valid operation/layout/geometry combinations are generated, never a full cross product. (5) The manifest selects a layout and states limits but cannot make an unsupported (head_dim, gqa, block, op) tuple appear supported: startup refuses with the layout, tuple, op and arch named, never falling back to another layout's row. (6) Two patches: refactor first with a proof, then new geometry. (7) Out of scope now: MLA, sliding window, quantised K and any descriptor field no kernel needs yet (head_dim 128 must fit by arithmetic alone and is the test of genericity). (8) Proof of no change: deterministic row-table dump (constraints, function identities, option bits) before and after, same compiler inputs recorded, selftests and fixed-input KLD equal; selftest expected values derived independently of the table so a bad descriptor is caught.

## Code Samples & Guidance

Patch A as built, rad0040_kv_layout_table (2026-11-16): `RAD_KV_LAYOUTS(X)` in `abi/rad_builder.h`, `X(id, name, row16, group)`, one line per packed layout. The `RAD_KVQ_*` enum (values unchanged; `RAD_KVQ_COUNT` appended), `rad_kvq_name`, `rad_kvq_from_name`, `rad_kvq_row16` and the `RAD_KV_LAYOUT_NAMES` string are generated from it; the flag parser (accepted names, refusal text, help line), `bc_kv_quant_of` and the qwen35_fp8 plugin read them. libr4d's bf16_fp8 / bf16_i8 attention rows come from one tuple list, `R4D_B16_ATTN_TUPLES` in `r4d_rows.cpp`: `X(tag, id, kv, kvdesc, head_dim, gqa, block, qlmax, geom, prefill_fn, prefill_scratch_fn, decode_fn, decode_scratch_fn)` generates the constraint arrays and the prefill and decode rows, and a `static_assert` ties each tuple's kv name to the table. Patch B adds gqa12 and block-4 as further tuples.

Deviations from step 1 above, decided while building it:
- Only the rows rad0037/38 own are generated. The upstream and fp8_v4 / k6_v4 rows are not rewritten: a textual patch cannot rewrite 30 upstream rows without making the patch as large as the file, and the proof of no change is cleaner for rows the patch already owns. They move onto the list when a change touches them.
- The `head_dim 256, gqa 6` check in the qwen35_fp8 plugin stays. Patch B adds the second plugin (qwen4exp_fp8) that needs the same check; the shared helper is written then, from two real uses, not one.
- The descriptor holds sizes and names only (id, name, row bytes per 16 dims, scale group). K/V element kinds and offsets stay documented in the ABI header comment and in the kernels: no kernel or plugin reads them as data yet, so a field would be speculative.

GPT code review of patch A (code-review-a.md): no blocking finding. The tuple list is for the `attn_paged` prefill and decode pair; patch B models the fused gate+quantiser as its own op kind with its own tuple list. Macro-valued tuple fields expand before stringify and paste.

Proof used for patch A: `libr4d/r4d_rows.cpp` compiles on the host alone in about a second, so the before and after objects are compared with `objdump -s` (all section bytes) and `objdump -r` (all relocations): identical. The device baseline waits for the corrected pinned selftest run (BASELINE-STATUS.md in the evidence folder): an unpinned `r4d_selftest` picks a gfx1100 card for gfx1201-only code and segfaults inside ROCm 7.2.4, on every row.

Patch B as built, rad0041_kv_b16_flashnext (2026-11-16): `RAD_KV_GEOMETRIES(X)` (`X(id, head_dim, gqa)`) and `rad_kvq_geometry_ok` in `abi/rad_builder.h`; the gqa8-body units (gqa12, gqa6 with 4-token pages) gain `b16f8` / `b16i8` prefill, decode and fused gate+quantiser entry points behind `R4D_ATTN_B16`; a second tuple list `R4D_B16_QSA_TUPLES` in `r4d_rows.cpp` generates their twelve rows (decode causal or not, the fused row its own op kind), with `static_assert`s that a tuple's kv name is the table's and its geometry is listed; the qwen4exp_fp8 plugin reads the layout and geometry tables and its probe offers every layout the table lists for the model's geometry. Deviations: the Qwen3.5 plugin keeps its own `gqa 6` check (one geometry; the shared list would admit untested ones); libr3 needed no rewrite change (the gqa8 body includes the gqa6 templates whose rad0037/38 rewrites exist), and the gfx1100 compile plus `r3_rewrite.py --check` stand in for the planned instantiation inventory; the selftest cases for the new rows are not written yet. gfx1201 resource usage is in `device-compile-b.md`: `bf16_i8` spills at most 22 VGPRs (existing worst case 25), `bf16_fp8` single-wave decode spills 45 (existing single-wave variants at most 20), so Flash-Next decode needs a speed A/B.

Patch B2, rad0042_kv_b16_selftest (2026-11-17): the device cases for rad0041's rows, in `libr4d/r4d_selftest.cpp`. The Flash-Next QSA numeric cases take `bf16_fp8` and `bf16_i8` (gqa12 and gqa6 block-4: prefill 64 and 100 rows, decode at every wave count), the split-KV prefill and mixed-step tables gain both geometries, and the fused gate+quantiser byte-identity case builds a bf16-K cache with finite V bytes and f16 group scales (row sizes from `RAD_KVQ_*_ROW16`). It compiles and passes libr3's rewrite check; the device run is queued on the `gpu` lease (patched `r4d_selftest` against the corrected rad0038 baseline: 0 failures, 246 skipped, none in the b16 or gqa12 rows), so the new cases' first run is their evidence.

Device result (2026-11-16, DEVICE-AB.md): r4d_selftest on the R9700 with rad0040/41/42 applied: 0 failures, 248 skipped (baseline 246, the two extra are the gqa6 block-4 fused rows with no smoke case, as for fp8v4 / k6v4); every baseline `ok` line still `ok`; 274 new `ok` lines for the new rows including the gqa12 fused byte-identity cases. Not yet: 27B KLD equality for patch A, libr3 gfx1100 device run, Flash-Next KLD / speed / identity.

Flash-Next KLD (2026-11-17, FLASHNEXT-KLD.md): short corpus bf16_i8 0.0446, bf16_fp8 0.0495, fp8 0.0539 against a bit-identical bf16 repeat; but re-chunking the plain bf16 cache alone gives 0.037 on the same documents, the band RAD39 recorded. Placement was ruled out (pinned run identical). The 27B bar does not apply to this model.

Flash-Next speed and capacity (2026-11-17, FLASHNEXT-AB.md, two XTX, no helper, no speculation): decode and TTFT within about 1% of bf16 for bf16_i8, bf16_fp8 +6.7% on one-stream decode and 5.9% slower 48K TTFT; capacity 1.283x (bf16_i8) and 1.307x (bf16_fp8) against bf16, below 1.3x for bf16_i8 because the QSA block-key group is not quantised; needle probe 8 to 9 of 9 for the packed caches. Open: per-token expert-id dump (generic engine option, own patch) to confirm routing flips as the KLD floor; the helper arm; a quantised QSA block-key layout if 1.3x matters for bf16_i8.

Default-cache identity (2026-11-17, IDENTITY.md): Flash-Next default bf16 and fp8 caches match the stock build at the KLD float floor (mean 6.4e-11 and 3.3e-11, max 2.2e-8, top-1 100%, 4,161 positions, placement pinned). Not bit-exact by construction (two binaries); greedy-stream comparison and the 27B default caches are not run.

27B equality (2026-11-17, EQUALITY-27B.md): default bf16 and fp8 caches match the stock build at the float floor (short and long); packed caches on a same-build reference: long bf16_i8 0.002685 (bar 0.0027 met), bf16_fp8 0.003228, fp8 0.004947, reproducing VTP43; short bf16_i8 0.00141, bf16_fp8 0.00192, fp8 0.00224 (VTP43's short packed figures used an older build's reference and were too low). The libr3 gfx1100 selftest for patch B was not run by decision; Flash-Next serving on the two 7900 XTX exercised those rows.

Status 2026-11-17 (end of this slice): rad0037/38/40/41 are validated in serving-core (#1033); `experiment.kv-bf16-packed` now lists only rad0042. Open, in order of value:
- Routing changes under any perturbation (FLASHNEXT-ROUTING.md, 2026-11-17, from the existing `--expert-stats` tallies): about 1.1% of Flash-Next's routed selections move whenever the arithmetic changes, equally for a chunk-size change, bf16_i8, bf16_fp8 and fp8, and every arm differs from every other by the same ~22,000; the control pair is exact. This fits the KLD floor and shows KLD cannot rank KV caches on this model, but causation is still inferred. To make KLD usable on Flash-Next: record each step's chosen expert ids in the reference run and replay them in the measured runs (a per-token expert-id record plus a replay option, two manifest fields, an engine patch of its own). Until then Flash-Next quality for the packed caches rests on the needle probe only.
- Repeat the Flash-Next helper A/B (FLASHNEXT-HELPER-AB.md) for the reference bf16 arm and bf16_i8 one-stream decode, with the n-gram table in RAM once the host has the memory (it needs 89.7 GB; about 82 GB was available), and a heat file kept in the task folder (the run folder holding it was deleted by a workspace cleanup mid-run).
- A quantised QSA block-key layout: the unquantised block-key group holds bf16_i8 to 1.283x bf16 on Flash-Next, under the 1.3x bar.
- A greedy-token stream comparison across builds (IDENTITY.md checks KLD only); the libr3 gfx1100 device selftest for patch B's rows (not run by decision; Flash-Next serving exercised them); the allowlist reduction and the GPT code review of patch B.

Status 2026-11-18: rad0043 (`--expert-stats` counts and labels every expert, protected ones included) and rad0044 (`--expert-route-record FILE`, manifest field `expert_route_record`) are merged (#1047, #1076). rad0045 (`--expert-route-replay FILE`, manifest field `expert_route_replay`, new op `router_topk_replay` in libr4d) is in review (#1089): on Flash-Next TP2 every replayed run takes the recorded route (expert counts equal the record's in all 48 layers) and a self-replay gives KLD exactly 0. With the route fixed the KV caches separate (FLASHNEXT-REPLAY.md, 4,163 tokens, one run per arm): chunk-128 noise reference 0.0076, bf16_i8 0.0074 (not distinguishable from it), bf16_fp8 0.0084, fp8 0.0097, against 0.037 to 0.040 for all of them without replay. Open, in order of value:
- A second corpus slice (and a repeat) for the replayed KLD, so the ordering is not one document pair's; score the R9700-engine layouts the same way.
- Layout smoke (background): flashnext-2xtx and flashnext-tp2xtx start and serve with bf16_fp8 and bf16_i8; flashnext-4gpu bf16 at the layout's sessions (8 x 262144 tokens, KV about 35 GB) does not fit the R9700 and the run retries with four sessions; helper-free flashnext-r9700 cannot be driven by the launcher and its engine-only start needs io_uring, which this container's seccomp profile refuses.
- Repeat the Flash-Next helper A/B (FLASHNEXT-HELPER-AB.md) for the reference bf16 arm and bf16_i8 one-stream decode, with the n-gram table in RAM and more than one run per arm.
- A quantised QSA block-key layout: the unquantised block-key group holds bf16_i8 to 1.283x bf16 on Flash-Next, under the 1.3x bar.
- A greedy-token stream comparison across builds (IDENTITY.md checks KLD only); the libr3 gfx1100 device selftest for patch B's rows.
- Remove the `RADIANCE_DEBUG_ROUTING` environment hooks (rad_block_moe_fp8.h, bc_moe_remote_block.h, ctx.cpp): the record supersedes them and they break the manifest-fields rule.
- Promote rad0043 (stats labelling) after qualification: `--expert-profile` placement reads these counts.

## Files

- `abi/rad_builder.h`, `core/config.cpp`, `core/rad_core.h`, `core/engine_bringup.cpp`, `core/kvstream/bringup.cpp`, `arch/qwen35_fp8/qwen35_fp8.cpp`, `libr4d/r4d_rows.cpp`, `tests/config_test.cpp` (through patch rad0040).
- `engines/radiance/patches/rad0040_kv_layout_table/`, `engines/radiance/patches/rad0041_kv_b16_flashnext/`, `engines/radiance/patches/rad0042_kv_b16_selftest/` (`libr4d/r4d_selftest.cpp`; rad0041 also `arch/qwen4exp_fp8/qwen4exp_fp8.cpp`, `libr4d/r4d_attn_paged_gqa8.hip` and its two wrappers, `libr4d/r4d_plugin.h`, `tests/arch_test.cpp`), `engines/radiance/recipes.toml` (experiment `kv-bf16-packed`).
- `docs/evidence/2026-11-16-vtp44-generic-kv-layouts/` (design review, baseline catalogues, BASELINE-STATUS.md).

## Validation

Refactor (patch A): row-table dump identical before/after; instantiation inventory identical; r4d_selftest (gfx1201) and r3_selftest (gfx1100) pass with the same case counts; 27B long and short KLD equal to the VTP43 numbers (bf16_i8 mean 0.00269) from a same-build reference; no new spills. New geometry (patch B): selftest numeric, split, mixed, store and fused-store cases at gqa12 and block 4 against an independent oracle with negative controls; spill counts per instantiation; Flash-Next KLD and speed on flashnext-tp2xtx-r9700; refusal tests for an unlisted tuple naming layout, tuple, op and arch; model-constant guard allowlist shrinks.

## Effort & Risk

L. Risks: option bits and tile shapes at gqa12 differ (spills near the 256-VGPR bound), libr3 rewrites can silently omit an instantiation, the patch system's unique-guard and earlier-hunk pitfalls across a large multi-file refactor, and rad0037/38 text being refactored while still untested. Mitigation: two patches, the dump-diff proof, and the instantiation inventory check.

## Standards



## Acceptance Criteria

- [x] GPT design review 1 findings resolved in this plan before implementation.
- [x] A single KV-layout descriptor table drives names, row sizes, probe, bringup, streaming refusal and the serving enum; no per-plugin lists remain. DONE: rad0040 (#1012); `RAD_KV_LAYOUTS` and `RAD_KV_GEOMETRIES` in `abi/rad_builder.h`.
- [x] The refactor of the gqa6 rows is proven unchanged (row-table dump, selftests, 27B KLD). DONE: row-table dump and instantiation inventory identical, r4d_selftest (gfx1201) 0 failures with every baseline row still passing, 27B default caches identical to the stock build and the packed caches reproducing VTP43 (EQUALITY-27B.md).
- [x] Flash-Next serves bf16_i8 and bf16_fp8 on the two 7900 XTX with the R9700 helper; KLD and speed recorded against its bf16 cache. KLD done 2026-11-17 (FLASHNEXT-KLD.md): it sits inside Flash-Next's arithmetic-noise band, so it cannot judge KV error. Speed, capacity and a 9-question 48K needle probe done on the two XTX without the helper (FLASHNEXT-AB.md); the helper arm is done too (FLASHNEXT-HELPER-AB.md, 2026-11-17): production layout with the R9700 helper, n-gram table on disk because host RAM was short; capacity per token 1.283x and 1.308x bf16, no loss beyond a run-to-run spread that was large (one bf16_fp8 outlier repeated). bf16_i8's one-stream decode (19% under bf16) was run once and is not resolved.
- [x] An unsupported (layout, head_dim, gqa, block, op) refuses at startup naming the tuple. DONE for (layout, head_dim, gqa): the plugin refuses naming them (`rad_kvq_geometry_ok`, arch_test); (block, op) mismatches are caught by row resolution.
- [ ] libr3 rewrites are per template with counts and an instantiation inventory check; model-constant allowlist entries added by VTP43 are removed or reduced.
- [x] rad0037/38 promoted through the lightweight tier with two named models, or the gap recorded. DONE 2026-11-17: rad0037, 38, 40 and 41 validated in #1033 (tag fix #1032) with 27B and Flash-Next as the named models; rad0042 stays an experiment overlay (it only edits the selftest).
- [ ] GPT code review done and findings resolved; PR(s) merged with evidence.

## Notes

Created 2026-11-16 from the VTP43 follow-up. Owner direction: move away from hardcoded configs and model-specific paths; reuse and design for newer models. Flash-Next topology to test: two 7900 XTX (gfx1100, libr3) TP2 as the dense engine, R9700 (gfx1201) as the remote-MoE helper (config flashnext-tp2xtx-r9700); the earlier Flash-Next arrangement had the R9700 as engine with XTX and RX 6900 XT helpers.

## Change Log

- 2026-11-16T07:45:37.732759+00:00 (created-by): Created by agent
- 2026-11-16T07:46:39.525612+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-16T08:21:18.350003+00:00 (state-transition): State: pending → in_progress
