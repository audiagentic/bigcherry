---
id: PRBE38
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:05.203236+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-FUS-002: Fuse GEMV activation + elementwise MUL

## Description

Implement and qualify the AMD-FUS-002 HIP GEMV epilogue fusion for graphs matching GEMV -> SiLU/activation -> elementwise MUL. The live target is the exact supported fused layout; unsupported broadcast, shape, dtype, or alias forms must remain on the unfused path.

## Steps

1. Identify the GEMV epilogue matcher and enumerate activation-plus-MUL graphs from captured decode traces. 2. Add a fail-closed matcher for supported SiLU/activation and MUL broadcast/layout forms, preserving Q8_0 and native-BF16 guards. 3. Emit the fused epilogue only when tensor shapes, strides, dtype, and output ownership are proven compatible; otherwise dispatch the existing sequence. 4. Add focused positive, negative, and unsupported-layout tests. 5. Replay representative decode graphs and compare launch count, tensor-group (TG) timing, and HBM traffic against the unfused control.

## Detailed Solution & Technical Design

Fuse the post-activation multiply into the HIP GEMV epilogue so the intermediate activation is not written and reread. The matcher must prove the exact GEMV -> SiLU -> MUL topology, compatible broadcast semantics, contiguous/strided output constraints, and destination ownership. Keep a conservative fallback for every other MUL shape or aliasing arrangement. Preserve existing accumulation precision and backend guards; this is a dispatch-path optimization, not permission to change numerical semantics.

## Code Samples & Guidance

Trigger: GEMV -> SiLU -> MUL with the supported scalar/per-channel broadcast and output layout. Controls: alternate MUL broadcasting, non-contiguous or aliased outputs, unsupported activation/dtype, and graphs where the activation result is consumed elsewhere. Required negative result: no fusion and unchanged unfused dispatch.

## Files

HIP GEMV epilogue matcher/emitter and dispatch code; graph-pattern tests for positive and negative forms; replay manifest and evidence under docs/evidence/ or the campaign output for AMD-FUS-002.

## Validation

Correctness: elementwise output parity against the unfused reference across representative shapes, dtypes, activation values, and edge cases. Safety: prove no unsupported broadcast or alias form fuses. Performance: record dispatch/launch count, TG/kernel timing, and HBM traffic for matched and control graphs; require a repeatable improvement or no-regression decision. Acceptance: exact supported layout only; decline fusion on ambiguity.

## Effort & Risk

M; matcher and epilogue changes are localized but incorrect broadcast or alias assumptions can silently corrupt decode outputs. Fail-closed fallback and differential tests contain the risk.

## Standards

Preserve native-BF16/F32 accumulation guards, Q8_0 behavior, unsupported-hardware fallback, and the repository patch qualification/evidence rules.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: RD46
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd46

Supersedes RD46. Depends on PRBE37 (AMD-FUS-001). Related fusion work must not broaden this matcher without a separate acceptance decision.

## Change Log

- 2026-09-09T10:56:05.203236+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:18.726737+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.301853+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.057267+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:02:34.325623+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030318_repaired-three-patching-succes_9681
- 2026-09-10T03:03:18.963781+00:00 (updated-by): Updated: section:ledger-events
