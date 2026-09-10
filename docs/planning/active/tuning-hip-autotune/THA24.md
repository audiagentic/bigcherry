---
id: THA24
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:34.713754+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Custom gfx1100 Q8_0 × Q8_1 WMMA kernel for K=5120, M=6144, N=512 (cherrypips family)

## Description

Implement a bounded gfx1100 Q8_0×Q8_1 WMMA custom-kernel readiness/qualification path for K=5120,M=6144,N=512, without speculative enablement.

## Steps

Keep custom_kernel opt-in; validate architecture mask/stable identity, distinct correctness reference, SHA-256 architecture-matched resource report, and passed benchmark evidence; add offline regression tests and preserve native candidates; only then consider source/dispatch implementation.

## Detailed Solution & Technical Design

Readiness contract precedes any cherrypips kernel. Malformed custom candidates must fail before catalog/registry generation; no broad external RDNA candidates substitute for exact gfx1100 objective.

## Code Samples & Guidance



## Files

Custom candidate schema/validator, catalog/registry tests, future kernel package only after evidence.

## Validation

HI25 readiness tests/full suite; identity/mask/reference/resource/benchmark digest checks.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

No custom kernel reaches runtime/catalog without complete architecture, identity, correctness, resource, and benchmark evidence; native remains default.

## Notes

Supersedes: HI25
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi25

## Change Log

- 2026-09-09T10:50:34.713754+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:52.675805+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.965772+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T12:09:25.332102+00:00 (updated-by): Updated: section:title
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.550547+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:33:19.585135+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033345_repaired-three-more-tuning-suc_6484
- 2026-09-10T03:33:45.427767+00:00 (updated-by): Updated: section:ledger-events
