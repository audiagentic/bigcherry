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

## Files

- `abi/rad_builder.h`, `core/config.cpp`, `core/rad_core.h`, `core/engine_bringup.cpp`, `core/kvstream/bringup.cpp`, `arch/qwen35_fp8/qwen35_fp8.cpp`, `libr4d/r4d_rows.cpp`, `tests/config_test.cpp` (through patch rad0040).
- `engines/radiance/patches/rad0040_kv_layout_table/`, `engines/radiance/patches/rad0041_kv_b16_flashnext/` (also `arch/qwen4exp_fp8/qwen4exp_fp8.cpp`, `libr4d/r4d_attn_paged_gqa8.hip` and its two wrappers, `libr4d/r4d_plugin.h`, `tests/arch_test.cpp`), `engines/radiance/recipes.toml` (experiment `kv-bf16-packed`).
- `docs/evidence/2026-11-16-vtp44-generic-kv-layouts/` (design review, baseline catalogues, BASELINE-STATUS.md).

## Validation

Refactor (patch A): row-table dump identical before/after; instantiation inventory identical; r4d_selftest (gfx1201) and r3_selftest (gfx1100) pass with the same case counts; 27B long and short KLD equal to the VTP43 numbers (bf16_i8 mean 0.00269) from a same-build reference; no new spills. New geometry (patch B): selftest numeric, split, mixed, store and fused-store cases at gqa12 and block 4 against an independent oracle with negative controls; spill counts per instantiation; Flash-Next KLD and speed on flashnext-tp2xtx-r9700; refusal tests for an unlisted tuple naming layout, tuple, op and arch; model-constant guard allowlist shrinks.

## Effort & Risk

L. Risks: option bits and tile shapes at gqa12 differ (spills near the 256-VGPR bound), libr3 rewrites can silently omit an instantiation, the patch system's unique-guard and earlier-hunk pitfalls across a large multi-file refactor, and rad0037/38 text being refactored while still untested. Mitigation: two patches, the dump-diff proof, and the instantiation inventory check.

## Standards



## Acceptance Criteria

- [x] GPT design review 1 findings resolved in this plan before implementation.
- [ ] A single KV-layout descriptor table drives names, row sizes, probe, bringup, streaming refusal and the serving enum; no per-plugin lists remain.
- [ ] The refactor of the gqa6 rows is proven unchanged (row-table dump, selftests, 27B KLD).
- [ ] Flash-Next serves bf16_i8 and bf16_fp8 on the two 7900 XTX with the R9700 helper; KLD and speed recorded against its bf16 cache.
- [ ] An unsupported (layout, head_dim, gqa, block, op) refuses at startup naming the tuple.
- [ ] libr3 rewrites are per template with counts and an instantiation inventory check; model-constant allowlist entries added by VTP43 are removed or reduced.
- [ ] rad0037/38 promoted through the lightweight tier with two named models, or the gap recorded.
- [ ] GPT code review done and findings resolved; PR(s) merged with evidence.

## Notes

Created 2026-11-16 from the VTP43 follow-up. Owner direction: move away from hardcoded configs and model-specific paths; reuse and design for newer models. Flash-Next topology to test: two 7900 XTX (gfx1100, libr3) TP2 as the dense engine, R9700 (gfx1201) as the remote-MoE helper (config flashnext-tp2xtx-r9700); the earlier Flash-Next arrangement had the R9700 as engine with XTX and RX 6900 XT helpers.

## Change Log

- 2026-11-16T07:45:37.732759+00:00 (created-by): Created by agent
- 2026-11-16T07:46:39.525612+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-16T08:21:18.350003+00:00 (state-transition): State: pending → in_progress
