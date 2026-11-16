---
id: VTP44
order: 0
plan: run-dense-27b-virtual-tp
state: pending
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

1. rad0039 (refactor only, no new geometry): a KV-layout descriptor table in abi/rad_builder.h (name, exact storage rules: K/V element kinds, offsets, scale placement, group, row bytes as an integral-checked function of head_dim, alignment) read by the core, both arch plugins, probe, bringup, streaming refusal and the serving enum generator; per-plugin name and row-size lists and the `gqa == 6` guards removed; row declarations for the existing gqa6 rows generated from an explicit list of (op, layout, head_dim, gqa, block, tuning) tuples. Behaviour byte-identical: row-table dump diff, option-bit diff, selftests, 27B KLD equal.
2. rad0040: gqa12 rows (prefill, decode, scratch) and the attn_paged_gate_quant rows for b16f8/b16i8 as new tuples with their own tuning, qwen4exp_fp8 plumbing, libr3 rewrites at gqa12 and block-4 as per-template rules with count checks plus a post-rewrite instantiation inventory.
3. libr3: replace per-geometry rewrite blocks with per-template rules where the build allows, else keep exact-count blocks; add an inventory check that expected and produced gfx1100 instantiations match.
4. Flash-Next validation: flashnext-tp2xtx-r9700 (two 7900 XTX TP2, R9700 remote-MoE helper) with kv_cache_dtype bf16_i8 / bf16_fp8 vs its bf16 reference; KLD, speed A/B, identity check (default cache identical with and without rad0037-40).
5. Lightweight promotion (QFP18 tier, profile evidence) of rad0037/38 (and rad0039/40 if they stand alone) with Qwen3.8-27B and Flash-Next as the two named models.

## Detailed Solution & Technical Design

GPT design review 1 (docs/evidence/2026-11-16-vtp44-generic-kv-layouts/design-review-1.md). Accepted: (1) The descriptor defines exact storage semantics (offsets, alignment, scale placement, packing order, padding) as declarative data and the table builder rejects any supported geometry whose row is not integral and aligned; no function pointers in the ABI table. (2) Geometry-specific tuning stays explicit: a generated row takes its option bits and instantiation from a per-tuple entry, so a gqa12 row never inherits gqa6 tile size or LDS skew; only row-construction logic is shared. (3) libr3: generate or rewrite per template with exact counts and a mandatory post-rewrite audit comparing expected and produced instantiations, because a broad rule can compile while a row is silently absent; the `@@ find N or 0` pair form is audited for the same reason. (4) The fused gate+quantiser is its own operation kind (attn_paged_gate_quant) consuming a layout descriptor with its own tuple set; only valid operation/layout/geometry combinations are generated, never a full cross product. (5) The manifest selects a layout and states limits but cannot make an unsupported (head_dim, gqa, block, op) tuple appear supported: startup refuses with the layout, tuple, op and arch named, never falling back to another layout's row. (6) Two patches: refactor first with a proof, then new geometry. (7) Out of scope now: MLA, sliding window, quantised K and any descriptor field no kernel needs yet (head_dim 128 must fit by arithmetic alone and is the test of genericity). (8) Proof of no change: deterministic row-table dump (constraints, function identities, option bits) before and after, same compiler inputs recorded, selftests and fixed-input KLD equal; selftest expected values derived independently of the table so a bad descriptor is caught.

## Code Samples & Guidance



## Files



## Validation

Refactor (rad0039): row-table dump identical before/after; instantiation inventory identical; r4d_selftest (gfx1201) and r3_selftest (gfx1100) pass with the same case counts; 27B long and short KLD equal to the VTP43 numbers (bf16_i8 mean 0.00269) from a same-build reference; no new spills. New geometry (rad0040): selftest numeric, split, mixed, store and fused-store cases at gqa12 and block 4 against an independent oracle with negative controls; spill counts per instantiation; Flash-Next KLD and speed on flashnext-tp2xtx-r9700; refusal tests for an unlisted tuple naming layout, tuple, op and arch; model-constant guard allowlist shrinks.

## Effort & Risk

L. Risks: option bits and tile shapes at gqa12 differ (spills near the 256-VGPR bound), libr3 rewrites can silently omit an instantiation, the patch system's unique-guard and earlier-hunk pitfalls across a large multi-file refactor, and rad0037/38 text being refactored while still untested. Mitigation: two patches, the dump-diff proof, and the instantiation inventory check.

## Standards



## Acceptance Criteria

- [ ] GPT design review 1 findings resolved in this plan before implementation.
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
