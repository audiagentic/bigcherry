---
id: THA27
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:49.382139+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Make tuning falsifiable — predicted matmul saving, and the decode fraction it applies to

## Description

Make tuning falsifiable by joining call-weighted per-signature savings to measured matmul fraction and required A/B repetitions.

## Steps

Implement impact report from record calls + measurements with confidence intervals and family/signature breakdown; trace rocprofv3 matmul fraction by stable kernel mapping; compute expected E2E effect and repetitions; run A/B at required count or record below-resolution result. Use no-flush B2 execution contract.

## Detailed Solution & Technical Design

Separate precise local kernel evidence from noisy end-to-end throughput. Predicted effect = matmul_fraction × predicted_saving; never present point estimates as exact.

## Code Samples & Guidance



## Files

bigcherry impact tool, trace aggregation/mapping, TUNING-DETAIL report and evidence.

## Validation

Reproduce RV19 ~10.2% call-weighted result, interval propagation, fraction and repetition estimate, real A/B/no-resolution conclusion.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Every claimed tuning benefit has reproducible impact model, confidence bounds, measured decode fraction, and a resolved A/B or explicit below-resolution disposition.

## Notes

Supersedes: HI35
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi35

## Change Log

- 2026-09-09T10:50:49.382139+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:07:07.589758+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.979180+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T12:09:29.097375+00:00 (updated-by): Updated: section:title
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.573334+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:34:45.973983+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033527_repaired-five-more-tuning-succ_3062
- 2026-09-10T03:35:27.044252+00:00 (updated-by): Updated: section:ledger-events
