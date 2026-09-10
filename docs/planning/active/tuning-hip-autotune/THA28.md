---
id: THA28
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:53.238663+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# HI36b — generalised dispatch runtime: two-level lookup and generalised cache entries on replay v5

## Description

Implement replay-v5 generalized two-level lookup for mmq/mmvq using canonical masked digests and explicit match provenance.

## Steps

After v5 foundation, define C++/Python byte-identical generalized digest masks (mmq family/type/K/M; mmvq family/type/full ne0); export representative highest-call promoted winner with collision assertion; lookup exact→generalized→native; guard can_execute; separate counters/match_kind; run alignment-class experiment and holdouts.

## Detailed Solution & Technical Design

Generalized key is recommendation only; real target signature authorization remains can_execute. No generalized keys for mmvf/blas. Exact behavior unchanged when disabled.

## Code Samples & Guidance



## Files

replay v5 format/digest, exporter, C++/Python lookup, coverage and tests.

## Validation

Cross-language vectors, disabled byte-identical counters, can_execute failure native/miss, same-workload exact hits unchanged, 49/49 holdout prediction target.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only mmq/mmvq generalized entries with self-describing match_kind/provenance, digest equivalence, can_execute safety, and holdout evidence; no v4 marker or NO-GO family keys.

## Notes

Supersedes: HI75
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi75

## Change Log

- 2026-09-09T10:50:53.238663+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:07:13.269137+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.984179+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T12:09:31.481521+00:00 (updated-by): Updated: section:title
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.580210+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:34:52.915790+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033527_repaired-five-more-tuning-succ_3062
- 2026-09-10T03:35:27.061717+00:00 (updated-by): Updated: section:ledger-events
