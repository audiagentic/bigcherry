---
id: RHA04
order: 0
plan: run-hip-autotune
state: in_progress
created-at: '2026-09-09T10:49:44.318293+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Establish production native parity and measure tuned dispatch end to end

## Description

Runtime/parity qualification and hardware matrix remain in progress. The 2026-09-08 HI168 bundle completed all requested server-bench cells (four 9B GPUs plus dual-XTX 27B, stock/native/replay, balanced orders) with clean shutdown and useful exploratory metrics, but every cell has execution_evidence_status=missing and performance_admitted=false. Those numbers are direction-finding only; the decision-grade rerun must use the current server-capture contract with explicit expected_execution locators and diagnostics-off production arms, followed by the matched dual-XTX 27B run.

## Steps

1. Resolve model/build/topology from config/environment.toml and BC_* variables; record immutable build plans, source revisions, compiler/generated-input bindings, model identity, GPU identity, topology, runtime settings, dispatch mode, cache identity, and diagnostics state.
2. Use the maintained server-bench endpoint runner documented in docs/reference/testing/TEST.md; do not use llama-bench. Verify graceful shutdown and retain raw runner output plus machine-readable cell evidence.
3. Complete the 9B Q6_K matrix serially on each physical GPU (XTX GPU0, XTX GPU1, gfx1201, gfx1030 where available): llama.cpp stock/native, BC native, and validated BC replay/tuned. Keep diagnostics-off production timing separate from diagnostic activation evidence.
4. Run the fresh matched dual-XTX 27B Q8_0 matrix after the 9B matrix, with stock, BC native, and validated replay arms, balanced/interleaved ordering, uncontended GPUs, non-speculative settings first, and the same production-shaped prompt/decode workload.
5. Analyze pp and generation independently with per-cell samples, ordering/position effects, uncertainty, correctness, replay activation, final tuned-launch counts, cache compatibility, and work-equivalence. A missing proof is not a pass; diagnostic throughput is never a production performance claim.
6. If native regression is established, bisect the common production path and dispatch overhead before changing winner policy; repeat the matched contrast after each accepted fix. If tuned replay is neutral or slower, report that result and retain native fallback rather than forcing promotion.
7. Exercise the remaining BC build types using the same runner and standardized configuration/output schema, but classify diagnostic/framework observations separately from the three-arm production result.
8. Update reusable test/build documentation and preserve raw artifacts, provenance, limitations, and exact environment roles. Only after the complete matrix and validation gates pass may this successor be considered for completion.

## Detailed Solution & Technical Design

Capability owner: run.

Primary comparison: llama.cpp stock/native versus the same-release BC native build, with BC replay/tuned as a separate cache-on arm. The cache-off/on contrast must use the same production binary wherever possible so winner effects are not confused with compile or patch differences.

Required evidence model per cell: build identity and effective compiler/generated-copy digest; model/quantization; GPU and topology; runtime/dispatch mode; diagnostics flag; cache/header/manifest identity; request/workload settings; warmup and sample order; pp and tg metrics; correctness/work-equivalence verdict; graceful teardown result; replay coverage and final tuned-launch evidence where applicable.

The 9B per-GPU matrix answers architecture/device-specific behavior. The subsequent dual-XTX 27B run answers the production topology/model question. Neither result may be generalized beyond its recorded identity. Existing historical captures remain historical and are not silently promoted to current evidence.

## Code Samples & Guidance



## Files

docs/reference/testing/TEST.md
docs/reference/testing/MULTI_GPU_LARGE_MODEL_VALIDATION.md
tools/bigcherry/campaign/bench_runner.py
tools/bigcherry/campaign/benchmark.py
tools/bigcherry/tuning/server_runner.py
artifacts/<run-id>/ (raw evidence only)
successor-specs/run-hip-autotune-hi168.md

## Validation

The retained 2026-09-08 evidence bundle at docs/evidence/2026-09-08-HI168-e2e/ proves server-bench completion, balanced ordering, successful requests, and clean teardown, but explicitly records missing physical-device attestation. It cannot satisfy closure. Next validation is a fresh serial 9B matrix using required execution evidence and raw identity-bound receipts, then the matched 27B matrix; retain all invalid/missing-attestation cells and do not promote their throughput.

## Effort & Risk

Hardware execution is time-consuming and can be invalidated by concurrent workload, changed model/build, diagnostics leakage, cache incompatibility, or unequal MTP work. Preserve failed/invalid cells with their reason; do not retry selectively until a preferred result appears. Use environment roles and availability checks rather than committed host facts.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md
docs/reference/testing/TEST.md
docs/reference/testing/MULTI_GPU_LARGE_MODEL_VALIDATION.md
docs/standards/HIP_AUTOTUNE_STANDARDS.md
No llama-bench for server-bench qualification.

## Acceptance Criteria

- Complete, identity-bound 9B Q6_K per-GPU stock/native/replay matrix with raw server-bench evidence and explicit unsupported/invalid classifications.
- Complete fresh matched dual-XTX 27B Q8_0 stock/native/replay comparison after the 9B matrix.
- Diagnostics-off production timing is separated from diagnostic activation and framework/build observations.
- Replay activation, cache compatibility, final tuned-launches, correctness, work-equivalence, graceful teardown, balanced ordering, pp/tg uncertainty, and limitations are evidenced.
- Remaining BC build types are exercised with the same reusable runner or explicitly recorded as unavailable.
- Process/docs are reusable and environment-configurable; implementation changes are tested, ledger-recorded, committed, and pushed.

## Notes

Supersedes: HI168
Migration: capability-rebaseline-v3-2026-09
Successor key: run-hip-autotune-hi168

Supersedes: HI168\nMigration: capability-rebaseline-v3-2026-09\nSuccessor key: run-hip-autotune-hi168\n\nInherited HI168 bundle is exploratory only because physical execution attestation is missing. Rerun must use environment settings, explicit expected_execution locators, maintained server-bench, and diagnostics-off production timing before any parity or replay conclusion.

## Change Log

- 2026-09-09T10:49:44.318293+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:54.453513+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.903535+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:52:28.033769+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:standards, section:acceptance_criteria
- chg_20260909_135244_made-the-hi168-successor-concr_5314
- 2026-09-09T13:52:44.598004+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T13:53:18.300323+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260909_135334_corrected-the-rha04-plan-text_5379
- 2026-09-09T13:53:34.448039+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T14:03:31.913574+00:00 (state-transition): State: pending → in_progress
- chg_20260909_140341_rha04-is-now-actively-executin_4158
- 2026-09-09T14:03:41.316296+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T14:33:08.340751+00:00 (updated-by): Updated: section:description, section:validation, section:notes
- chg_20260909_143319_marked-the-existing-hi168-numb_5423
- 2026-09-09T14:33:19.052828+00:00 (updated-by): Updated: section:ledger-events
