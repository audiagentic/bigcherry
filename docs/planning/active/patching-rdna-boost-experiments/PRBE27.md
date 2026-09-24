---
id: PRBE27
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:19.101435+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-MMQ-002: Dedicated RDNA3.5 MMQ device table

## Description

TODO, hardware-blocked. Same redesign-against-current-table approach as PRBE22, scoped to gfx1151 (RDNA3.5, Strix Halo integrated) which already has its OWN dedicated table function `ggml_cuda_mmq_get_config_rdna3_5` (confirmed at b11126, mmq-config-rdna3-5.cuh) -- so 'a dedicated RDNA3.5 MMQ device table' already EXISTS upstream; PRBE27's real scope is tuning/redesigning that existing table's candidate rows, not creating a new dedicated table from scratch. This project has NO gfx1151 hardware (project hardware is gfx1100/gfx1201/gfx1030 per rdna-plan-brief.md and CLAUDE.md/memory) -- this item cannot be validated until such hardware is available. Original upstream PR #25 diff is obsolete against the current table and must not be ported verbatim.

## Steps

1. Use the same catalog-driven approach as PRBE22 (tools/bigcherry/tuning/catalog.py enumerate_mmq, reading mmq-config-rdna3-5.cuh directly) to define gfx1151-only candidate table rows for dense/MoE Qwen corpus at exact Q4/Q6/Q8 shapes.
2. Build/non-selection controls on gfx1100/gfx1201 (candidates gated `cc` to RDNA3.5's specific value must not be selected on RDNA3/RDNA4 -- verify via build-clean + candidate-selection unit test, no hardware needed for this check).
3. Author the candidate rows and campaign config now (design-only deliverable).
4. DO NOT run hardware validation -- retain explicit 'hardware-blocked redesign' disposition until gfx1151 hardware is available in this project.
5. When hardware becomes available: run the same parity/temp-0/PP/TG/resource-stat campaign described in PRBE22's validation section, scoped to gfx1151.

## Detailed Solution & Technical Design

Because gfx1151 already has its own dedicated config-table function upstream (unlike gfx1100 which shares the plain rdna3 table), PRBE27 is narrower than it sounds: it is a candidate-row tuning exercise on an existing table, using the same catalog/campaign infrastructure identified for PRBE22, gated to `GGML_CUDA_CC_RDNA3_5` (an existing `cc` dispatch value in mmq.cuh:242 `ggml_cuda_mmq_get_config_rdna3_5(type, J, fallback)`). No new dispatch branch or new dedicated table is required -- the 'dedicated table' acceptance criterion in the item's own title is already satisfied by upstream; only new/re-tuned rows within it are new work.

## Code Samples & Guidance

Real b11126 anchor (verified): ggml/src/ggml-cuda/mmq.cuh:242 `return ggml_cuda_mmq_get_config_rdna3_5(type, J, fallback);` inside the architecture dispatch chain (alongside gcn/cdna/rdna4/rdna3/rdna2/blackwell/ampere/pascal branches at lines 233-278). Candidate rows follow the same CASE-macro format already verified in mmq-config-rdna3-5.cuh this batch. No patch package should be authored with STATE other than a design-only artifact until gfx1151 hardware exists; when it does, sketch as patches/12xx_rd34_mmq_rdna3_5_retune/patch.toml, state=untested, plan-item=RD34.

## Files

ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh (candidate rows, design-only for now); tools/bigcherry/tuning/catalog.py (candidate enumeration); no patch package committed until hardware exists.

## Validation

Now (no hardware): build-clean on gfx1100-only and multi-arch builds with gfx1151 candidate rows present but non-selected (cc-gated); a candidate-selection unit test proving gfx1100/gfx1201 never resolve to the new rdna3_5-only rows. Later (when gfx1151 hardware exists): full PPL/temp-0 parity, pp128/512/1024/4096, TG, resource stats, decode-neutrality check -- same shape as PRBE22's campaign, scoped to gfx1151.

## Effort & Risk

S now (design/candidate-definition only), M later when hardware validation becomes possible. Risk of doing real work that can never be promoted is explicitly accepted by the item's own 'retain hardware-blocked redesign disposition' acceptance criterion.

## Standards

Needs-redesign; hardware-scoped; no unsupported extrapolation.

## Acceptance Criteria

A current table-driven gfx1151 design is proven and correct with positive hardware evidence, or no implementation is promoted; obsolete PR #25 diff is never applied.

## Notes

Supersedes: RD34
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd34

2026-09-24 relevance at b11126: TODO, hardware-blocked. Confirmed gfx1151 already has its own dedicated ggml_cuda_mmq_get_config_rdna3_5 table function upstream -- PRBE27's title goal ('dedicated RDNA3.5 table') is already met; remaining work is candidate-row tuning within it, blocked on gfx1151 hardware this project does not have. GPT design request for PRBE22+27 hit a queue-saturated gateway and was not obtained in-session; plan authored directly from verified mmq.cuh source.

## Change Log

- 2026-09-09T10:55:19.101435+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:31.554843+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.249087+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.981923+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:55:23.253944+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025542_rdna-successors-prbe2628-now_5552
- 2026-09-10T02:55:42.894059+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:31:00.635833+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
