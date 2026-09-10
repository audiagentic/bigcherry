---
id: THA20
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:18.646462+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# rmsnorm/norm candidate-search: authorize and build tuning for normalization kernels once HI174 proves the pattern

## Description

After THA18 proves the non-matmul hook, authorize rmsnorm/norm candidate search only after determining real internal variants versus new kernel authoring.

## Steps

Wait for THA18; inspect norm.cu variants and identity; if variants exist extend hook/telemetry and HI67/HI143/HTR01 pipeline, otherwise create separate kernel-authoring scope; validate native first and preserve matmul.

## Detailed Solution & Technical Design

Rmsnorm is a simple pilot alternative to RoPE. Do not invent candidate variants before THA18 determines decomposition versus authoring.

## Code Samples & Guidance



## Files

norm.cu dispatch, THA18 hook/identity, telemetry, tests/evidence.

## Validation

Native correctness/telemetry and candidate evidence only after foundation; full suite and gfx1100.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

No work before THA18; proceed only with justified real variants, stable identity, and full correctness/behavioral/E2E gates.

## Notes

Supersedes: HI176
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi176

## Change Log

- 2026-09-09T10:50:18.646462+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:31.666728+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.944455+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.521841+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:31:41.118746+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033234_repaired-four-non-matmul-tunin_4268
- 2026-09-10T03:32:34.816162+00:00 (updated-by): Updated: section:ledger-events
