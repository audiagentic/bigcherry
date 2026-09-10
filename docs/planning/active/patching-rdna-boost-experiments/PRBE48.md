---
id: PRBE48
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:49.423040+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# UP-HRX-001: AMD-native HRX backend comparative lane

## Description

Scope an AMD-native HRX backend lane as a separate experimental backend, measured against matched HIP and Vulkan contracts rather than mixed into HIP patches.

## Steps

1. Recheck ggml-hrx PR #27218 and define the supported model/backend contract. 2. Decide the relationship to the proposed Vulkan lane before implementation; keep HRX isolated. 3. Build a minimal backend maturity matrix for Qwen3.6 dense/MoE Q8/Q4 on supported gfx targets. 4. Run backend correctness, PPL/temp-0, protocol, PP/TG/MTP, VRAM, load-time, kernel-coverage, and build-maturity checks against matched HIP/Vulkan controls. 5. Keep experimental unless feature parity and repeatable advantage are demonstrated.

## Detailed Solution & Technical Design

Treat HRX as a third backend lane with identical model/workload recipes and explicit ownership. Do not cherry-pick kernels into HIP; first establish compile/build maturity, operation coverage, protocol compatibility, and matched performance evidence. User decision on generalizing with Vulkan remains an explicit scope gate.

## Code Samples & Guidance



## Files

Separate ggml HRX backend tree and build integration; backend-op/correctness tests; matched Qwen workload recipes; HIP/Vulkan comparison reports and evidence.

## Validation

Correctness: backend tests, PPL/temp-0 output, and server protocol compatibility where applicable. Performance: PP/TG/MTP, VRAM, load time, kernel coverage, and compile maturity against HIP/Vulkan. Acceptance: experimental only until correctness and feature parity are adequate and a repeatable required-workload advantage is shown.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Keep HRX experimental unless matched backend correctness/feature parity and repeatable PP/TG/MTP advantage are demonstrated; do not merge backend-specific kernels into HIP without a separate decision.

## Notes

Supersedes: RD57
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd57

## Change Log

- 2026-09-09T10:56:49.423040+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:02.970413+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.345686+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.123012+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:10:25.411076+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031049_repaired-three-vulkanbackend_9010
- 2026-09-10T03:10:49.471004+00:00 (updated-by): Updated: section:ledger-events
