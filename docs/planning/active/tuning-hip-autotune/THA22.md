---
id: THA22
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:50:26.827128+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# Quantize/dequantize candidate-search: authorize and build tuning for quant/dequant kernels

## Description

After THA18, determine whether standalone quantize/dequantize dispatch is distinct from HI173 BLAS conversion routes before authorizing tuning.

## Steps

Audit quantize.cu and dequant paths versus HI17/THA17 conversion telemetry; define ownership and identity; if distinct, extend hook/telemetry and correctness/behavioral pipeline; otherwise record reuse/no-duplicate disposition.

## Detailed Solution & Technical Design

Do not duplicate BLAS conversion-route work. Scope only standalone operations outside BLAS and preserve native fallback.

## Code Samples & Guidance



## Files

quantize/dequantize dispatch, shared identity/telemetry, tests/evidence.

## Validation

Boundary/ownership audit, native correctness, telemetry and candidate evidence only after distinct surface proven.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

No candidate search until distinct dispatch ownership and THA18 pattern are proven; promote only with exact correctness and E2E gates.

## Notes

Supersedes: HI178
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi178

## Change Log

- 2026-09-09T10:50:26.827128+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:06:36.850833+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.954844+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.536503+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:32:14.398788+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033234_repaired-four-non-matmul-tunin_4268
- 2026-09-10T03:32:34.850756+00:00 (updated-by): Updated: section:ledger-events
