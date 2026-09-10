---
id: PRBE49
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:53.658140+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-VK-002: Vulkan FA MMQ FP32 quant-scale calculation

## Description

Verify and, if absent, implement Vulkan FA-MMQ FP32 quantization-scale calculation to prevent FP16 denormal/overflow corruption at long context.

## Steps

1. Recheck llama.cpp PR #27413 against the pinned Vulkan FA-MMQ path. 2. Reproduce tiny-scale Q4/Q8 KV cases with a reference FA implementation and inspect NaN/Inf behavior. 3. Implement FP32 scale/reciprocal computation only if the pin lacks an equivalent fix. 4. Add synthetic tiny-amplitude, normal-range, non-MMQ, and deep-context model tests. 5. Measure regression budget and retain the existing path if correctness is already covered.

## Detailed Solution & Technical Design

Compute FA-MMQ Q quantization scale and reciprocal in FP32 so denormals and overflow do not corrupt attention results. Scope the change to Vulkan MMQ; preserve non-MMQ paths and compare against reference outputs.

## Code Samples & Guidance



## Files

Vulkan FA-MMQ quantization shader/helper; tiny-scale and normal-range backend-op tests; deep-context Qwen replay and regression evidence.

## Validation

Correctness first: reference FA output, no NaN/Inf, model output/PPL for tiny-scale synthetic and deep-context real cases. Performance: quantify any regression against normal-scale controls. Acceptance: carry only if current pin lacks equivalent protection and the corruption case reproduces or the source fix is low-risk and verified.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Require exact/reference-quality FA output with no NaN/Inf on tiny-scale Q4/Q8 KV cases and no material normal-workload regression; otherwise retain or document existing equivalent protection.

## Notes

Supersedes: RD59
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd59

## Change Log

- 2026-09-09T10:56:53.658140+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:07.160220+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.349689+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.129637+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:10:31.384387+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031049_repaired-three-vulkanbackend_9010
- 2026-09-10T03:10:49.493059+00:00 (updated-by): Updated: section:ledger-events
