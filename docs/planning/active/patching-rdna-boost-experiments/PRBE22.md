---
id: PRBE22
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:56.585436+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-MMQ-001: RDNA MMQ tile-width reduction for LDS occupancy

## Description

TODO. Investigate RDNA3 (gfx1100) MMQ tile-width (mmq_x=J)/LDS-occupancy redesign; original upstream PR #32 diff is invalid against the current table-driven mmq-config-rdna3.cuh architecture. Relevance: still meaningful (gfx1100 is a primary hardware target). Critically, this project already has the exact infrastructure this item needs: tools/bigcherry/tuning/catalog.py's enumerate_mmq() and read_mmq_config_table() parse the CURRENT architecture-specific CASE-row tables directly out of the real header (not a stale diff), and mmq.cuh's `ggml_cuda_mmq_get_config(type, J, fallback, cc)` dispatches per architecture (rdna3 has its own function, separate from rdna3_5/gfx1151) -- so PRBE22 is a candidate-definition + campaign-run task against existing infrastructure, not new kernel authoring.

## Steps

1. Read tools/bigcherry/tuning/catalog.py's `enumerate_mmq()`, `read_mmq_config_table()`, and `MMQConfigRow` to confirm how it already parses mmq-config-rdna3.cuh's CASE rows (mirrors the rdna3_5 table format confirmed this batch: `CASE(type, mmq_x, nwarps, dim, dim2, LAYOUT, ITER_K, bool, bool)`).
2. Define high-cost dense/MoE tensor-shape signatures for Qwen3.6-27B (dense) and Qwen3.6-35B-A3B (MoE) at gfx1100/gfx1201/gfx1151-family-separated architecture scope (gfx1151 itself is out of scope for PRBE22 -- see PRBE27).
3. Use the catalog to enumerate alternate mmq_x candidate rows swept around the native winner for physical M/ubatch 64..4096, rather than hand-editing CASE macros.
4. Run tools/bigcherry/patch/validation_campaign.py's record stage for backend-op parity + temp-0/bit identity on every candidate before any timing measurement (fail-closed: a candidate that doesn't match upstream numerically is dropped before benchmarking).
5. Run the tune stage; capture LDS/VGPR/waves/occupancy via rocprofv3 alongside pp128/512/1024/4096 median/MAD kernel timing.
6. Promote only a candidate meeting >3% repeatable kernel gain or >1% E2E gain with <=1% non-target regression, gated to the exact (architecture, type, shape-signature) it was measured on -- never a global default. Otherwise retain the redesign findings as negative evidence.

## Detailed Solution & Technical Design

The catalog's own design note (catalog.py module docstring) states MMQ candidates are deliberately read OUT of the current upstream table rather than invented, specifically because inventing arbitrary J values would generate candidates that abort on launch -- this directly rules out porting the obsolete PR #32 diff's J values without re-deriving them against the current table. PRBE22's actual new work is: (a) defining the high-cost signature set (which (type, shape) pairs matter for the two target models), (b) choosing which additional J rows to add to the catalog (if the native table is missing a promising width) vs. which existing rows to just re-measure at the new signature set, (c) running the campaign and interpreting occupancy data. No new CUDA kernel code is required unless a genuinely new J value needs a new CASE row (in which case that row must satisfy mmq.cuh's existing config-struct shape, not invent a new field).

## Code Samples & Guidance

Real b11126 anchors (verified): ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh row format (CASE macro, rdna3_5 table -- rdna3's own table, mmq-config-rdna3.cuh, is structurally the same per catalog.py's ARCH_CONFIG_HEADERS mapping and was not read directly this batch, must be confirmed by implementer); mmq.cuh:230 `ggml_cuda_mmq_get_config(const ggml_type type, const int J, const bool fallback, const int cc)` dispatching via `cc` to `ggml_cuda_mmq_get_config_rdna3` for gfx1100. Patch package sketch (only needed if a new CASE row is added, not for a pure catalog/campaign re-scope): patches/12xx_rd28_mmq_tile_width_rdna3/patch.toml (state=untested, plan-item=RD28, kind=enhancement) with an Edit() anchored on the existing CASE-row block in mmq-config-rdna3.cuh, mode="insert_after" the last row of the relevant type, adding one new mmq_x candidate row; guard on `cc == GGML_CUDA_CC_RDNA3`.

## Files

ggml/src/ggml-cuda/mmq-config-rdna3.cuh (RDNA3 table, needs direct read to confirm exact row format before editing); tools/bigcherry/tuning/catalog.py (candidate enumeration, no change needed unless adding a genuinely new J); campaign config for the Qwen3.6-27B/35B-A3B high-cost signature set; rocprofv3 occupancy capture (tools/bigcherry/analysis/resource_report.py already exists for blacklist/resource accounting, confirm it also emits LDS/VGPR/occupancy or extend it).

## Validation

1. Catalog dry-run / patch-lint if a new CASE row is added. 2. test-backend-ops MUL_MAT/MUL_MAT_ID parity + temp-0 determinism for every candidate before timing. 3. pp128/512/1024/4096 kernel median/MAD, call-weighted E2E ranking, LDS/VGPR/waves/occupancy via rocprofv3 (Brutus, not run here). 4. Architecture non-target controls (gfx1201/gfx1030 must show no regression from a gfx1100-gated candidate).

## Effort & Risk

M: mostly campaign/measurement work reusing validated framework; risk is in correctly defining high-cost signatures and interpreting occupancy trade-offs, not in kernel-authoring correctness.

## Standards

Needs-redesign disposition; architecture-specific evidence; no upstream-diff assumption.

## Acceptance Criteria

A current table-driven design is proven or a new selector is authored from current seams; obsolete anchors are not ported; promotion meets the explicit gain/regression threshold.

## Notes

Supersedes: RD28
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd28

2026-09-24 relevance at b11126: TODO. Confirmed the project's own tools/bigcherry/tuning/catalog.py already reads MMQ candidates directly out of the current architecture tables (not a stale PR diff) -- this substantially de-scopes PRBE22 from 'design a new kernel' to 'define signatures and run the existing campaign'. mmq-config-rdna3.cuh itself was not read this batch (only rdna3_5's structurally-identical table was); implementer must confirm the exact CASE row shape before adding any new row. GPT design request for PRBE22+27 hit a queue-saturated gateway (8 queued/2 running) and was not obtained in-session; plan authored directly from verified catalog.py + mmq.cuh source.

## Change Log

- 2026-09-09T10:54:56.585436+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:11.248783+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.227051+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.949508+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:52:05.324105+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025218_rdna-successors-prbe2022-now_5714
- 2026-09-10T02:52:18.433749+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:30:47.787525+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
