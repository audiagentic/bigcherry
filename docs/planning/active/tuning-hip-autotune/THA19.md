---
id: THA19
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:14.474006+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# RoPE candidate-search: authorize and build tuning for RoPE dispatch once HI174 proves the pattern

## Description

After THA18 establishes a viable pilot pattern, authorize RoPE candidate search only if real internal variants exist.

## Steps

Wait for THA18's decomposition-versus-authoring result; map rope.cu variants and stable resolved-config identity; use THA18 hook and HI67/HI143/HTR01 evidence pipeline; add candidate search/correctness/behavioral gating only for proven variants, otherwise file a separate kernel-authoring item.

## Detailed Solution & Technical Design

RoPE remains blocked until the foundational non-matmul hook and identity extension are accepted. Do not invent variants or silently widen scope.

## Code Samples & Guidance



## Files

rope.cu dispatch, candidate schema/registry, correctness/behavioral evidence and tests.

## Validation

Native/candidate correctness and real record telemetry; candidate performance only after foundation and positive gate.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

No implementation before THA18; promote only real variants with exact identity and full correctness/behavioral/E2E evidence, otherwise retain native.

## Notes

Supersedes: HI175
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi175

## Change Log

- 2026-09-09T10:50:14.474006+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:26.865897+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.939547+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.514013+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:30:32.574143+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033046_repaired-four-more-tuning-succ_3978
- 2026-09-10T03:30:47.035703+00:00 (updated-by): Updated: section:ledger-events
