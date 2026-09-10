---
id: PRBE53
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:10.449932+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-FA-001: Mixed K/V FlashAttention dispatch for asymmetric KV types

## Description

Enable mixed K/V FlashAttention dispatch for asymmetric KV types only where MMA/tile kernels support both types and reference correctness is established.

## Steps

Recheck PR #27150; extend CUDA/HIP FA eligibility beyond early type-equality rejection for f16/q8_0, q8_0/q4_0, and q4_0/q8_0; verify actual MMA/tile route on XTX/R9700; test symmetric and unsupported controls across Q rows 1/3/128/512 and depths 0/32K/64K/128K/240K.

## Detailed Solution & Technical Design

Allow mixed K/V cache types to use tile FA when each type is independently supported, avoiding catastrophic vector fallback. Keep symmetric and unsupported mixed combinations on existing dispatch.

## Code Samples & Guidance



## Files

CUDA/HIP FA dispatcher eligibility; mixed-type reference/PPL tests; route telemetry; long-context replay evidence.

## Validation

Reference FA/PPL parity, actual kernel route and no fallback, PP/TG/VRAM and selected FA kernel. Acceptance only combinations with correctness coverage and real GPU FA path.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Require reference-quality mixed-K/V outputs, verified non-fallback MMA/tile execution, and no material PP/TG/VRAM regression across specified long-context controls.

## Notes

Supersedes: RD63
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd63

## Change Log

- 2026-09-09T10:57:10.449932+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:23.119898+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.367626+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.160338+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:11:52.617299+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031217_repaired-four-more-active-succ_7909
- 2026-09-10T03:12:17.949987+00:00 (updated-by): Updated: section:ledger-events
