---
id: PRBE41
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:18.783942+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# AMD-SSM-001: Channels-major SSM_CONV input mode

## Description

Implement and qualify channels-major SSM_CONV input support across CPU and HIP, preserving existing time-major callers and enabling DeltaNet/Qwen hybrid graphs without mandatory transposes.

## Steps

1. Recheck AMD PR #52 merge f6ea7bced112 against the pinned ggml SSM APIs. 2. Separate API/layout semantics from architecture performance and add the public SSM_CONV layout parameter. 3. Implement CPU reference support and HIP/CUDA short- and long-token paths, with clean fallback predicates for unsupported backends. 4. Switch only proven channels-major DeltaNet/Qwen graph callers; keep time-major callers bit-identical. 5. Validate composition with PRBE42 chunked GDN, PRBE21 SSM-conv concat, and any RD27 materialization. 6. Exercise tensor-parallel/meta split state in a real multi-device run. 7. Measure ubatch-sensitive prefill and verify decode neutrality.

## Detailed Solution & Technical Design

Add a backward-compatible channels-major input mode to ggml_ssm_conv so DeltaNet avoids a physical transpose. Implement both layouts in CPU and HIP/CUDA, propagate layout metadata through graph construction and backend meta splits, and use explicit support predicates with fallback for non-implementing backends. Keep the API/layout change separate from the chunked recurrence in PRBE42. Do not alter time-major callers.

## Code Samples & Guidance

Trigger: Qwen3.6/Qwen3.5 hybrid/GDN prefill with proven channels-major graph construction. Controls: existing time-major callers, unsupported backends, short/long token paths, and single versus multi-sequence. Boundary: ubatch 256/512/1024/2048/4096 and representative context lengths.

## Files

ggml public SSM_CONV API/op params; CPU and HIP/CUDA SSM_CONV implementations; backend support predicates; meta split-state handling; DeltaNet/Qwen graph callers; direct backend-op tests; multi-device replay manifest and evidence.

## Validation

Direct op: both layouts, d_conv 3/4/9, representative d_inner, short/long tokens, single/multi sequence; reproduce 90/90 layout parity where applicable. Model: qwen35 dense, qwen35moe/Qwen3.6, qwen3next deterministic/PPL parity. Composition: PRBE41 alone, PRBE42 alone, RD49+RD50, and relevant PRBE21/RD27. Tensor parallel/meta: real multi-device execution. Performance: report CONCAT/CONT/SSM_CONV/GDN times and total prefill at ubatch 256..4096; decode must remain neutral. Acceptance: backward-compatible API and correct CPU+HIP with fallback elsewhere.

## Effort & Risk

L; shared API and multiple graph/backend callers create broad regression risk. Preserve old layout and test all support predicates before claiming performance.

## Standards

Preserve bit-identical time-major behavior, CPU reference coverage, backend fallback, multi-device/meta correctness, and campaign evidence provenance.

## Acceptance Criteria

Acceptance requires backward-compatible channels-major and time-major SSM_CONV behavior, 90/90-class direct parity, CPU+HIP correctness with clean fallback, real multi-device/meta correctness, correct composition with PRBE42, and stable prefill evidence with decode neutral.

## Notes

Supersedes: RD49
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd49

Supersedes RD49. PRBE21 is a distinct SSM conv_input concat optimization; cross-reference but do not merge.

## Change Log

- 2026-09-09T10:56:18.783942+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:34.146919+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.314431+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.077374+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:04:24.733208+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030451_carried-forward-the-detailed-s_2071
- 2026-09-10T03:04:51.318883+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:05:49.720099+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030619_removed-migration-placeholder_7703
- 2026-09-10T03:06:19.327420+00:00 (updated-by): Updated: section:ledger-events
