---
id: PRBE55
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:18.934230+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-VK-004: Small-N speculative execution avoids inappropriate MMVQ route

## Description

Route small-N speculative/MTP verification widths through decode-like Vulkan paths when MMVQ harms throughput or acceptance.

## Steps

Recheck PR #25666; characterize N=1..8 routing and acceptance at contexts 0/32K/128K; add shape/arithmetic selector for widths 2/3/4/5/8; compare normal TG N=1, prefill N>=128, and non-MTP small-batch controls; record kernel path, verify time, effective TG, and acceptance.

## Detailed Solution & Technical Design

Treat MTP verify widths 2-4 as decode-like when ordinary batch MMVQ is harmful, but prefer shape/arithmetic predicates over a workload string. Preserve normal TG and prefill routes.

## Code Samples & Guidance



## Files

Vulkan matmul routing selector; temp-0 identity/acceptance tests; MTP replay and route/performance evidence.

## Validation

Temp-0 identity; acceptance must not decrease; kernel path, verify time, effective TG and acceptance across widths/contexts.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Require temp-0 identity, no unexpected acceptance loss, and repeatable verify/TG improvement before selecting the small-N route; use shape-based gating and retain baseline otherwise.

## Notes

Supersedes: RD65
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd65

## Change Log

- 2026-09-09T10:57:18.934230+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:32.113186+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.376629+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.175315+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:12:04.636398+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031217_repaired-four-more-active-succ_7909
- 2026-09-10T03:12:17.975805+00:00 (updated-by): Updated: section:ledger-events
