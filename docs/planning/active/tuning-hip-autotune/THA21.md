---
id: THA21
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:22.602748+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P3
---

# Flash-attention candidate-search: authorize and build tuning for attention kernels (largest, highest-risk op class)

## Description

After THA18 and a simpler pilot, investigate flash-attention candidate search with attention-specific identity and safety.

## Steps

Wait for THA18 plus a completed RoPE/rmsnorm pilot; inspect fattn/flash-attention-ext variants, KV-cache/mask/precision structure; choose hook/identity extension; add native telemetry first, then candidate search only with correctness/behavioral gates.

## Detailed Solution & Technical Design

Highest-risk non-matmul class; do not assume matmul identity transfers. Separate candidate dimensions for mask/cache/precision/sequence and preserve native fallback.

## Code Samples & Guidance



## Files

fattn*.cu, attention-specific hook/identity, telemetry, correctness and behavioral tests.

## Validation

Native and candidate reference/PPL correctness, cache/mask/precision coverage, real record telemetry and positive E2E.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Remain blocked until THA18 pilot proves foundation; promote only exact attention-safe variants with full correctness and E2E evidence.

## Notes

Supersedes: HI177
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi177

## Change Log

- 2026-09-09T10:50:22.602748+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:42.072905+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.950375+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.529403+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:32:05.001893+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033234_repaired-four-non-matmul-tunin_4268
- 2026-09-10T03:32:34.839312+00:00 (updated-by): Updated: section:ledger-events
