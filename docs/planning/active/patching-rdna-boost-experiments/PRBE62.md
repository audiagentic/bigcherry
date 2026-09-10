---
id: PRBE62
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:49.294113+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# VK-DRV-002: AMD graphics queue for compute on proprietary Vulkan

## Description

Evaluate AMD proprietary Vulkan graphics-queue dispatch for MoE decode while retaining compute queue for dense workloads unless evidence supports a scoped selector.

## Steps

Recheck discussion #21043; compare graphics versus compute queue on proprietary AMD Vulkan for MoE decode, dense Qwen decode/prefill, and unsupported drivers; verify output parity, TG/PP and queue utilization; implement only a workload/driver-conditioned selector.

## Detailed Solution & Technical Design

Graphics queues may help dispatch-heavy MoE while hurting dense workloads. Make queue choice conditional on workload and driver, with compute fallback and no global policy.

## Code Samples & Guidance



## Files

Vulkan queue selection and workload/driver predicate; output tests; MoE/dense benchmark evidence.

## Validation

Output parity; TG/PP and queue utilization separated by workload/driver.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Never enable globally; retain only a proven workload/driver-specific selector with repeatable benefit and no dense regression.

## Notes

Supersedes: RD79
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd79

## Change Log

- 2026-09-09T10:57:49.294113+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:59.947699+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.407607+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.225741+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:16:25.617635+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031644_repaired-four-more-active-succ_8062
- 2026-09-10T03:16:44.974520+00:00 (updated-by): Updated: section:ledger-events
