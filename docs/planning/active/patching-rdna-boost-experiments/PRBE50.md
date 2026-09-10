---
id: PRBE50
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:57.545335+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# UP-VK-003: Quantized GET_ROWS view-offset correctness

## Description

Verify and, if absent, implement Vulkan quantized GET_ROWS support for non-zero view offsets while keeping rows GPU-resident.

## Steps

1. Recheck llama.cpp PR #26854 and current Vulkan quantized GET_ROWS shader. 2. Add backend-op cases with offsets crossing quantization block boundaries for Q4/Q8. 3. Compare exact rows against reference and exercise offset-zero plus F16/F32/I32 controls. 4. Confirm Qwen VL/TTS view-based gather where available, no crash, and no CPU fallback. 5. Measure GPU residency and regression budget.

## Detailed Solution & Technical Design

Honor non-zero view offsets in the Vulkan quantized GET_ROWS shader by incorporating the byte/block offset into row addressing. Keep offset-zero and non-quantized paths unchanged, and fail safely rather than returning wrong rows or crashing.

## Code Samples & Guidance



## Files

Vulkan quantized GET_ROWS shader/addressing; Q4/Q8 offset tests across block boundaries; F16/F32/I32 controls; Qwen VL/TTS replay and GPU-residency evidence.

## Validation

Correctness: exact reference rows for offset-zero and non-zero offsets across block boundaries, no crash or wrong row, and no CPU fallback. Performance: confirm GPU residency and no material regression. Acceptance: correctness baseline must be present before promotion.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Require exact rows and stable GPU-resident execution for non-zero Q4/Q8 view offsets, including block-boundary cases, with no crash or material regression; otherwise retain fallback and keep item open.

## Notes

Supersedes: RD60
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd60

## Change Log

- 2026-09-09T10:56:57.545335+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:11.442219+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.355299+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.136955+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:10:37.505870+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031049_repaired-three-vulkanbackend_9010
- 2026-09-10T03:10:49.504719+00:00 (updated-by): Updated: section:ledger-events
