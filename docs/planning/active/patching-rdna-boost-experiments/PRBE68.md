---
id: PRBE68
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:18.039731+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# VK-SUB-001: Architecture-aware Vulkan submission cap

## Description

Implement architecture-aware Vulkan submission caps that protect timeout-prone older AMD while avoiding the reported RDNA4 throughput regression.

## Steps

Recheck issue #26679; characterize submission work/FLOP thresholds on gfx1201/R9700, older AMD/GCN timeout-prone controls, other vendors; add architecture-aware ceiling and telemetry; run PP/TG, submission count, GPU fault/timeout and corruption tests across long runs.

## Detailed Solution & Technical Design

Replace generalized submission safety heuristic with architecture-sensitive caps. Select ceiling from device identity and work threshold, preserving conservative limits where timeout risk exists and allowing modern RDNA4 throughput. Keep fallback and observability.

## Code Samples & Guidance



## Files

Vulkan command submission batching/cap selector; architecture/work-threshold tests; PP/TG and fault telemetry evidence.

## Validation

No DeviceLost, timeout or corruption; PP/TG, submission count, GPU fault telemetry and long-run stability.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Require architecture-specific cap that prevents safety failures while avoiding material modern-GPU throughput regression; retain conservative fallback on unknown devices.

## Notes

Supersedes: RD85
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd85

## Change Log

- 2026-09-09T10:58:18.039731+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:29.747464+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.434952+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.268296+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:18:48.866865+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031907_repaired-the-vulkan-submission_2651
- 2026-09-10T03:19:07.580884+00:00 (updated-by): Updated: section:ledger-events
