---
id: PRBE46
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:41.470034+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-VK-001: MoE density-aware Vulkan MMV routing

## Description

Implement and qualify Vulkan MoE density-aware MMV routing for RADV RDNA3/3.5/4, replacing the fixed batch cutoff only where routing density and driver evidence support it.

## Steps

1. Recheck llama.cpp PR #27332 against the pinned Vulkan backend. 2. Add a density-aware MUL_MAT_ID routing heuristic and preserve explicit batch boundary behavior at B=8 and B=9. 3. Gate by expert density, batch/concurrency 1..64, driver, and supported architecture; retain existing path for unsupported cases. 4. Test Qwen MoE -np 8/9/16/32/64, B<=8 controls, PP512, and dense models. 5. Record path selection, TG aggregate, kernel timing, and per-request latency across target drivers.

## Detailed Solution & Technical Design

Route Vulkan MUL_MAT_ID to MMV based on expert density rather than a fixed batch<=8 cutoff, targeting the reported batch-9 cliff. Keep the heuristic explicit and driver-aware; do not assume the source threshold generalizes across RADV generations. Preserve old routing as fallback and make B=8/B=9 boundary behavior observable.

## Code Samples & Guidance

Trigger: Qwen MoE -np 8,9,16,32,64 on XTX/R9700 RADV. Controls: B<=8, PP512, dense models, and unsupported drivers/architectures. Boundary: batch/concurrency 1..64 and routing density.

## Files

Vulkan MUL_MAT_ID dispatch heuristic; density/path-selection tests; RADV architecture/driver matrix; Qwen MoE replay manifests and evidence for UP-VK-001.

## Validation

Correctness: output parity against existing Vulkan path. Performance: report selected path, TG aggregate, kernel timing, and per-request latency with interleaved controls, explicitly covering B=8 and B=9. Acceptance: promote only when benefit generalizes to target drivers/architectures without regression; retain fixed routing otherwise.

## Effort & Risk

M; routing thresholds may overfit a driver or workload. Require density controls and explicit boundary tests.

## Standards

Preserve Vulkan fallback, driver/architecture qualification, and campaign evidence provenance.

## Acceptance Criteria

Acceptance requires output parity, explicit B=8/B=9 boundary coverage, density/driver/architecture controls, and a repeatable target-driver latency/TG benefit without regressing other batches; otherwise retain the existing routing.

## Notes

Supersedes: RD55
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd55

Supersedes RD55; coordinate with reusable-build Vulkan scoping without duplicating ownership.

## Change Log

- 2026-09-09T10:56:41.470034+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:54.998308+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.337328+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.109221+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:08:34.398338+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-10T03:09:12.902174+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030930_repaired-two-more-active-patch_7368
- 2026-09-10T03:09:30.729645+00:00 (updated-by): Updated: section:ledger-events
