---
id: PRBE09
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:07.484798+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# UP-VK-005: Vulkan tensor-parallel AllReduce -- upstream PR/issue tracking and BigCherry adoption path

## Description

Track the upstream Vulkan tensor-parallel AllReduce path and decide a BigCherry adoption route; implementation is not implied by tracking.

## Steps

- Identify the upstream PR/issue, current status, API/ABI and licensing/source commit.
- Map the proposal to RD68 and RD57 constraints, including tensor-parallel topology and Vulkan provider boundaries.
- Define a reproducible local control/treatment plan only if an upstream implementation is available and buildable.
- Measure correctness, synchronization, fallback and end-to-end throughput on representative single/dual topology.
- Record adopt, defer, or reject with explicit evidence and preserve the upstream reference.

## Detailed Solution & Technical Design

This item owns adoption analysis for Vulkan tensor-parallel AllReduce. It must not silently duplicate RD57 or claim a patch exists. Keep host/native fallback, provider identity and topology assumptions explicit.

## Code Samples & Guidance



## Files

Upstream PR/issue record; Vulkan AllReduce integration seam; RD68/RD57 cross-references; adoption decision and campaign evidence.

## Validation

Upstream status and source verification; build/license check; collective correctness and fallback; tensor-parallel topology matrix; balanced performance evidence.

## Effort & Risk



## Standards

Upstream provenance; provider/topology isolation; no implementation claim without source and evidence.

## Acceptance Criteria

An auditable upstream/adoption decision is recorded; any adopted code has exact source identity and passes correctness/fallback gates; otherwise the candidate remains deferred without production claims.

## Notes

Supersedes: RD104
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd104

## Change Log

- 2026-09-09T10:54:07.484798+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:13.948243+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.169761+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.860761+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:35:11.802321+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023529_three-more-rdna-successors-now_3176
- 2026-09-10T02:35:29.332235+00:00 (updated-by): Updated: section:ledger-events
