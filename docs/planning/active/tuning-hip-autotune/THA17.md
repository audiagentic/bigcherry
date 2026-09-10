---
id: THA17
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:06.337896+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# BLAS-family kernel tuning: authorize and build candidate search for the rocBLAS/Tensile path HI17 made observable

## Description

Authorize and build BLAS-family candidate search over rocBLAS/Tensile vendor solutions using existing promotion/correctness/behavioral gates.

## Steps

Implement separate vendor_selection identity layer; ingest rocblas-gemm-tune CSV/override artifacts; restrict to signatures with explicit solution APIs and observe effective backend; resolve/apply exact solution with vendor artifact identity invalidation; measure workspace source/lifetime; run seam parity and full MTP graph lifecycle then positive E2E promotion gates.

## Detailed Solution & Technical Design

Candidate is vendor algorithm/solution selection, not hand-written kernel. Keep HI17 seven-field semantic BLAS plan separate from selector identity. Fail closed on artifact mismatch and use standard HI67/HI143/HTR01 positive performance promotion; HI17 parity is seam-only.

## Code Samples & Guidance



## Files

BLAS selector/schema/runtime seam; rocblas-gemm-tune ingestion; workspace telemetry; candidate tests and Brutus evidence.

## Validation

Specified tests, real rocblas-gemm-tune on RD87 shapes, native seam parity, artifact invalidation, MTP capture/replay and per-solution determinism/workspace, positive E2E.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only exact vendor-artifact-compatible solutions with correctness, graph lifecycle, workspace and positive E2E benefit; no interpolation or bare version reuse.

## Notes

Supersedes: HI173
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi173

## Change Log

- 2026-09-09T10:50:06.337896+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:17.605955+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.929508+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.498621+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:30:17.383046+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033046_repaired-four-more-tuning-succ_3978
- 2026-09-10T03:30:47.014113+00:00 (updated-by): Updated: section:ledger-events
