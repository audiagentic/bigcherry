---
id: THA23
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:30.558840+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P3
---

# Softmax/GLU-activation candidate-search: authorize and build tuning for remaining elementwise op classes

## Description

After THA18's pilot, authorize softmax and standalone GLU/activation candidate search with simple op identities.

## Steps

Wait for THA18; inspect softmax.cu and glu.cu variants outside BLAS fusion; define stable resolved-config identity and hook; add native telemetry then candidate search/HI67/HI143/HTR01 gates; keep fused GLU signature observation separate.

## Detailed Solution & Technical Design

Treat softmax/activation as own dispatch targets, not the glu_op request field already observed in BLAS fusion. Preserve native fallback and avoid broad op-class bundling.

## Code Samples & Guidance



## Files

softmax.cu/glu.cu-family dispatch, candidate identity/registry, telemetry and tests.

## Validation

Native/candidate correctness, standalone-vs-fused boundary, record telemetry, behavioral and E2E performance.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Remain blocked until THA18; promote only real variants with exact identity and full correctness/behavioral/E2E evidence, otherwise native remains default.

## Notes

Supersedes: HI179
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi179

## Change Log

- 2026-09-09T10:50:30.558840+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:47.437594+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.959783+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.542615+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:32:20.933489+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033234_repaired-four-non-matmul-tunin_4268
- 2026-09-10T03:32:34.863179+00:00 (updated-by): Updated: section:ledger-events
