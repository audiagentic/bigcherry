---
id: THA18
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:09.823686+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# Extend the HIP autotune dispatch/candidate system beyond matmul: non-matmul op-class scope and foundational infra

## Description

Extend HIP autotune observability and candidate identity to one non-matmul pilot op without authorizing search or promotion.

## Steps

Select RoPE or rmsnorm pilot; inspect real dispatch variants and determine decomposition versus new kernel authoring; choose per-op versus generic hook architecture; define resolved-config candidate identity; extend hook/signature/replay only as needed; add native telemetry and offline tests; validate pilot on real gfx1100 while proving matmul unchanged.

## Detailed Solution & Technical Design

Infra-first foundation, not all-op tuning. Preserve HIP_AUTOTUNE_STANDARDS: resolved config in identity, request shape/type in signature, fail closed. Flash-attention remains later due complexity.

## Code Samples & Guidance



## Files

Pilot op dispatch and 0200 hook; signature/manifest/replay extension; record telemetry; tests.

## Validation

Pilot contract tests plus full suite; native correctness and record telemetry on gfx1100; matmul regression unchanged.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close only after pilot/architecture choice is justified, candidate identity and hook strategy are documented, telemetry works, and no candidate search/promotion is enabled.

## Notes

Supersedes: HI174
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi174

## Change Log

- 2026-09-09T10:50:09.823686+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:22.052698+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.934855+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.506371+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:30:27.100795+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033046_repaired-four-more-tuning-succ_3978
- 2026-09-10T03:30:47.025121+00:00 (updated-by): Updated: section:ledger-events
