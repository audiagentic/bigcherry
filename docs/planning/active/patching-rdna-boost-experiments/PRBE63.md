---
id: PRBE63
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:54.429485+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# VK-DRV-003: Cooperative-matrix bypass on AMD proprietary Vulkan

## Description

Compare conventional versus cooperative-matrix Vulkan routes by exact driver/ICD, architecture, operation, quant type, and signature; do not introduce global coopmat policy.

## Steps

Build the comparison matrix across proprietary AMDVLK/RADV, gfx1100/gfx115x/gfx120x, dense/MoE, MUL_MAT/MUL_MAT_ID, supported/excluded CM1 quant types and exact recorded shapes; fingerprint device/ICD; include PR #27952 as evidence; record route/fallback reason, parity, kernel/PP/TG and ISA where available; keep qualification separate from PRVP01-RO22.

## Detailed Solution & Technical Design

Coopmat choice is workload/type/driver dependent. Measure conventional and CM1 routes per signature; preserve dense RDNA4 exclusions as controls and selected MoE/quantized wins where observed. No architecture-only or global coopmat-off policy.

## Code Samples & Guidance



## Files

Vulkan route selector/telemetry; driver-signature comparison matrix; parity tests; kernel/PP/TG/ISA evidence.

## Validation

Parity for every route; performance split by driver/ICD, architecture, operation, quant, and signature; include RDNA4 dense and MoE controls.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

No global coopmat policy; conclusions scoped to fingerprinted driver/architecture/operation/quant/signature; independent of PRVP01-RO22 qualification.

## Notes

Supersedes: RD80
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd80

## Change Log

- 2026-09-09T10:57:54.429485+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:03.686191+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.411603+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.233203+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:16:32.506067+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031644_repaired-four-more-active-succ_8062
- 2026-09-10T03:16:44.985999+00:00 (updated-by): Updated: section:ledger-events
