---
id: RUM07
order: 7
plan: patching-radiance-upstream
state: pending
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

- Make a compact comparison of upstream RAD_I8 per-token/head K/V scale/row format and local bf16_i8/bf16_fp8/b16 formats: dtype, row bytes, scale placement, quantization/rounding and supported head/GQA forms.
- Reuse upstream implementations where the numeric/layout contract truly matches. Decide which local hybrid formats remain valuable; don't port every historical variant by default.
- Decide native equivalence before porting each format hunk. If no non-native production format survives, delete the local table and call native sizing/dispatch directly. Only for retained extra formats extend the existing rad0040 table; do not introduce a global descriptor registry.
- Port store and attention readers together; preserve d6109d7e causal split-KV merge behavior and MTP shared KV group semantics. Update model declarations and exact row strides, not only option names.
- Update rad0037/rad0038/rad0040/rad0041 hook definitions and the necessary rad0042 selftests. Remove duplicate switches/row-size literals for migrated formats in the same slice.
- Migrate authored names and contract enum/probe mappings explicitly, refusing obsolete formats with a migration message. Do not keep aliases whose semantics differ invisibly.
- Select only currently shipped target geometries for the initial selftest slice; advertise no untested combination.

## Detailed Solution & Technical Design

Target abi/rad_builder.h defines rad_kv_row_elems(dtype,head_dim)=2*head_dim+(dtype==RAD_I8?4:0). That helper counts elements for the declared dtype; compute bytes with the correct dtype width, not by assuming all returns are byte counts. Upstream I8 stores two int8 head vectors plus two f16 scales per token/head. Local mixed layouts may retain bf16 K or use different scale groupings, so names/byte-size similarity do not establish equivalence.

Prefer extending the existing rad0040 layout-table approach to target semantics rather than restoring removed kv_quant or old prefill_fp8 structs. Keep descriptors in owned source and use typed format IDs at kernel boundaries. Dispatch is resolved by model declaration/row selection; no virtual callback per attention element.

New target model families add attention shapes, but migrating the production Flash-Next layout does not authorize every head_dim/GQA permutation. Unsupported probes must refuse before allocation. Preserve exact-default behavior for the current accepted default and measure intentionally approximate modes against the existing quality/noise gate.

## Code Samples & Guidance

Native format: call rad_kv_row_elems and multiply by dtype storage width. Compare exact offsets/scales/rounding and causal read/write, not just names or byte size. If all supported formats are native, delete local layout/dispatch duplication. Only retained non-native formats extend existing rad0040 RAD_KV_LAYOUTS/rad_kvq_from_name; derive their planner/transfer/store/read bytes together and include libr3 r3_rewrite.py. No new global descriptor registry or per-element virtual dispatch. Source/removal proof: ASTRA-CONSOLIDATION.md and patch-footprint.json.

## Files

- engines/radiance/patches/rad0037_bf16_fp8_kv/patch.py
- engines/radiance/patches/rad0038_bf16_i8_kv/patch.py
- engines/radiance/patches/rad0040_kv_layout_table/patch.py
- engines/radiance/patches/rad0041_kv_b16_flashnext/patch.py
- engines/radiance/patches/rad0042_kv_b16_selftest/patch.py
- engines/radiance/components/serving/contract/serving-v2.toml

## Validation

For each retained production format: one store/read roundtrip, one causal split-merge case at the actual head/GQA shape, an allocation/transfer stride assertion and unsupported-shape refusal. Reuse existing r4d selftest cases rather than create a broad shape fuzzer. RUM09 adds one short model-quality canary; expand only for a real mismatch.

## Effort & Risk

L: equal row bytes do not imply equivalent scale/rounding semantics. Compare a small supported-format set, retain one descriptor source and refuse unqualified shapes rather than broadening the kernel matrix.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

Planning, transfer and kernels agree on one format descriptor; target native I8 is reused only where semantics match. Retained local formats have supported-path proof and correct allocation. No obsolete duplicated row-size switch or silent incompatible alias remains in the migrated callers.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM03, RUM06. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisites RUM03/RUM06. The 166 failed-edit review surface is partly cascading; don't repair anchors one by one before choosing the unified KV contract.

Astra source review ASTRA-CONSOLIDATION.md: evaluate this slice early and independently where possible, before porting obsolete hunks. Same byte count does not establish equivalent offsets/scales/rounding. Integrated qualification depends on selected ABI/placement/runtime.

## Change Log

- 2026-11-18T01:28:34.138312+00:00 (created-by): Created by codex
- 2026-11-18T01:39:53.057270+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:26.892039+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:57:05.367589+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:19:59.088702+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-11-18T02:23:39.101657+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:26:37.220559+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:28:24.920139+00:00 (updated-by): Updated: section:code_samples
