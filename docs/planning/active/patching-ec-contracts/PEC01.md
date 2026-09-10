---
id: PEC01
order: 0
plan: patching-ec-contracts
state: pending
created-at: '2026-09-09T10:47:32.637015+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# VK-TUNE-002: Large physical ubatch for Vulkan MoE prefill -- campaign boundary-sweep dimension

## Description

Run and disposition the Vulkan MoE physical-ubatch boundary sweep. This is campaign infrastructure/recommendation work, not a runtime kernel patch.

## Steps

- Verify source discussion #21043 and freeze the exact Vulkan baseline/provider identity before porting or measuring.
- Sweep physical ubatch 256, 512, 1024, 2048 and 4096 where memory allows on Qwen MoE Vulkan prefill.
- Use dense Qwen and memory-constrained controls; keep model/workload, context and batch dimensions fixed across arms.
- Check output correctness and collect PP, VRAM and latency with balanced repeats and failure/OOM evidence.
- Record whether the result becomes an EC03 campaign boundary or a model/workload tuning recommendation; do not convert it into a generic runtime default.

## Detailed Solution & Technical Design

Hypothesis is that large physical ubatch improves MoE tile/occupancy while dense models may not benefit. The result belongs in EC01 boundary.dimensions/EC03 campaign expansion and must remain workload/provider scoped.

## Code Samples & Guidance



## Files

Vulkan MoE campaign recipe; ubatch boundary matrix; Qwen MoE and dense controls; correctness/PP/VRAM/latency artifacts; EC03 recommendation.

## Validation

Output unchanged; ubatch 256/512/1024/2048/4096 where feasible; MoE vs dense; memory-constrained control; PP, VRAM, latency and OOM evidence.

## Effort & Risk



## Standards

Campaign boundary, not kernel patch; causal controls; preserve memory safety and correctness.

## Acceptance Criteria

A reproducible sweep yields an explicit recommendation or rejection. If beneficial, record it as a scoped model/workload boundary for EC03; no unqualified runtime patch or default is claimed.

## Notes

Supersedes: EC14
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-ec-contracts-ec14

## Change Log

- 2026-09-09T10:47:32.637015+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:32.639007+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.749756+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.197048+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:36:20.969226+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023638_the-two-ec-contract-successors_4025
- 2026-09-10T02:36:38.452678+00:00 (updated-by): Updated: section:ledger-events
