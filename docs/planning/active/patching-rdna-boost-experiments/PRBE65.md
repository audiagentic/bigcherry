---
id: PRBE65
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:03.403872+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# VK-FA-001: Decouple Vulkan FA occupancy tuning from exact shared-memory capability equality

## Description

Decouple Vulkan FA occupancy tuning from the hard shared-memory legality check while ensuring no illegal shader launch.

## Steps

Identify the affected AMD driver capability reports; separate legal shared-memory limit checks from occupancy heuristic selection; test reported 32/64KiB limits, genuinely undersized devices, non-AMD controls and explicit overrides; measure FA PP/TG before/after.

## Detailed Solution & Technical Design

Use architecture/driver-aware occupancy tuning independent of exact maxComputeSharedMemorySize==64KiB, but retain a hard legality gate based on actual shader requirements. Never select a shader exceeding device limits.

## Code Samples & Guidance



## Files

Vulkan FA tuning selector and legality predicate; capability/override tests; AMD/non-AMD PP/TG evidence.

## Validation

No illegal launch; FA PP/TG before/after across capability reports and controls.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Separate heuristic from legality, with no shader over-limit and repeatable occupancy benefit on affected devices; fallback on uncertain capability.

## Notes

Supersedes: RD82
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd82

## Change Log

- 2026-09-09T10:58:03.403872+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:11.754779+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.420235+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.247665+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:17:38.400709+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031805_repaired-four-more-graph-and-v_2834
- 2026-09-10T03:18:05.338956+00:00 (updated-by): Updated: section:ledger-events
