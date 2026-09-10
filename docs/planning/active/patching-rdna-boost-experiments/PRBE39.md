---
id: PRBE39
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:09.423127+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-FUS-003: Fuse GEMV -> view -> residual ADD

## Description

Implement and qualify AMD-FUS-003 HIP fusion for GEMV -> view/reshape -> residual ADD in supported hybrid/GDN decode graphs. Preserve alias safety and use the existing unfused path whenever view semantics are not provably exact.

## Steps

1. Locate the GEMV epilogue matcher and graph representations for view/reshape followed by residual ADD. 2. Define the supported view strides, shapes, dtype, and residual ownership conditions from captured hybrid/GDN graphs. 3. Add a fail-closed fused path that writes the GEMV result directly into the residual-add destination only when the view is an exact legal mapping. 4. Add tests for positive contiguous and supported strided views, non-contiguous/aliasing rejection, and ordinary transformer controls. 5. Replay graphs with graph capture enabled and compare outputs, launches, and TG timing with the unfused control.

## Detailed Solution & Technical Design

Extend GEMV epilogue fusion through an exact view/reshape into residual ADD, eliminating the intermediate write and a follow-up kernel. The implementation must validate byte strides, shape equivalence, dtype, destination aliasing, and residual read/write ordering before selecting the fused kernel. Any ambiguous alias or unsupported layout must retain the existing sequence. Keep this distinct from PRBE12/RD13, whose mul_mat+add view fusion targets a different SSM shape.

## Code Samples & Guidance

Trigger: hybrid/GDN decode GEMV -> view/reshape -> residual ADD with a supported exact mapping. Controls: different view strides, non-contiguous aliasing, residual reused elsewhere, and standard transformer paths. Boundary: captured supported view shapes only; reject ambiguous mappings.

## Files

HIP GEMV epilogue/view matcher and residual-add emitter; graph and alias-analysis tests; graph-capture replay manifest and evidence for AMD-FUS-003.

## Validation

Correctness: exact output parity and aliasing safety against the unfused reference, including residual reuse and graph-capture replay. Negative tests must show no fusion for unsupported strides, non-contiguous aliasing, or unknown ownership. Performance: record launch count, TG/kernel timing, and memory traffic versus control. Acceptance: only supported view semantics fuse; any alias ambiguity falls back.

## Effort & Risk

M; the principal risk is an incorrect stride or alias proof that changes residual results. Conservative matcher gating and differential tests are mandatory.

## Standards

Preserve graph capture correctness, dtype/accumulation behavior, unsupported-layout fallback, and patch qualification/evidence provenance.

## Acceptance Criteria

Acceptance requires exact GEMV->view->residual mapping and output/alias parity, graph-capture replay success, rejection of unsupported strides or ambiguous aliases, and repeatable launch/TG improvement versus control; otherwise retain fallback.

## Notes

Supersedes: RD47
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd47

Supersedes RD47. Keep separate from PRBE12/RD13: that existing mul_mat+add view fusion is a different operation and SSM graph shape.

## Change Log

- 2026-09-09T10:56:09.423127+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:25.553774+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.306463+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.063889+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:02:57.572846+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030318_repaired-three-patching-succes_9681
- 2026-09-10T03:03:18.986477+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:05:36.573029+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030619_removed-migration-placeholder_7703
- 2026-09-10T03:06:19.301698+00:00 (updated-by): Updated: section:ledger-events
