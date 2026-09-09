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
docs/evidence/2026-09-08-HI168-e2e/
docs/evidence/2026-09-10-rha04-attestation-smoke/
docs/evidence/2026-09-10-rha04-gpu0-required-capture/
successor-specs/run-hip-autotune-hi168.md

- docs/evidence/2026-09-10-rha04-gpu0-attestation-preflight/
- tools/bigcherry/campaign/benchmark.py

- docs/evidence/2026-09-10-rha04-matrix-summary/README.md

## Validation

The retained 2026-09-08 HI168 evidence bundle at docs/evidence/2026-09-08-HI168-e2e/ proves server-bench completion, balanced ordering, successful requests, and clean teardown, but explicitly records missing physical-device attestation. The 2026-09-10 BC-native dual-XTX smoke at docs/evidence/2026-09-10-rha04-attestation-smoke/ never reached health during model materialization and is explicitly invalid. The fresh 2026-09-10 GPU0 required-attestation capture at docs/evidence/2026-09-10-rha04-gpu0-required-capture/ loaded and shut down cleanly but failed closed before timing with ATTESTATION_MISSING; no throughput result was produced. This confirms the maintained server-capture attestation gap. The capture path now has an identity-bound, untimed preflight using the same binary/model/common production arguments and relevant environment, with only a diagnostic verbosity delta; it persists a correlation ID and hashes and fails closed on missing/wrong/ambiguous identity. The timed production process is unchanged. Unit coverage passes for success, failure-before-timing, argument/environment drift, clean shutdown, and existing observe/required behavior. RHA04 remains open until the fresh per-GPU 9B and matched dual-XTX 27B matrices complete with replay/work-equivalence evidence.

The maintained capture now runs an untimed identity-bound attestation preflight with the same binary/model/common production arguments and relevant environment, a separate diagnostic verbosity delta, persisted hashes/correlation ID, and fail-closed missing/wrong/ambiguous identity checks. Timed production arguments remain unchanged. Unit coverage passes for success, failure-before-timing, argument/environment drift, clean shutdown, and existing observe/required behavior. Hardware matrices remain pending.

Real Brutus GPU0 preflight now succeeds on the current pushed implementation: expected ROCm/gfx1100/0000:03:00.0 was observed with matching binary/model hashes, common production arguments and HIP/ROCR visibility, and clean SIGINT teardown. Raw evidence is retained under docs/evidence/2026-09-10-rha04-gpu0-attestation-preflight/. An earlier HTTP attempt correctly failed closed because the binary returned 404 and was force-killed; no timing was recorded. The full 9B per-GPU and dual-XTX 27B server-bench matrices remain pending.

A real Brutus GPU0 native timed smoke now passes the new gate and maintained server-bench contract: pp512 1198.19, pp2048 2460.25, tg128 84.83, tg512 85.36 tokens/s for one repetition, with verified ROCm/gfx1100/0000:03:00.0 identity and clean SIGINT teardown. Evidence is retained under docs/evidence/2026-09-10-rha04-gpu0-native-cell/. This is exploratory and remains performance_admitted=false; balanced stock/native/replay 9B and dual-XTX 27B matrices remain pending.

The first complete current-source decision-shaped cell is now retained for GPU0: 18/18 stock/native/replay cells across six balanced permutation rounds completed with verified physical identity and clean SIGINT teardown. Medians: stock 1290.355/2512.050/84.300/84.985, native 1279.175/2520.325/84.265/85.030, replay 1268.820/2523.355/84.260/84.725 (pp512/pp2048/tg128/tg512). Direct replay-vs-native bootstrap effects: pp512 -2.420% (95% -5.316..+0.201), pp2048 +0.026% (-0.249..+0.214), tg512 -0.360% (-0.935..+0.385). Evidence: docs/evidence/2026-09-10-rha04-gpu0-full/. This is still unadmitted; other GPUs and dual-XTX 27B remain required.

The GPU1 full current-source 9B matrix is now complete and retained: 18/18 stock/native/replay cells across six balanced rounds, verified physical identity gfx1100/0000:06:00.0, and clean SIGINT teardown. Medians: stock 1303.420/2519.355/84.385/85.080, native 1296.765/2526.685/84.490/85.185, replay 1299.100/2529.420/83.835/84.755 (pp512/pp2048/tg128/tg512). Direct replay-vs-native bootstrap effects: pp512 +0.205% (-0.841..+1.195), pp2048 +0.063% (-0.165..+0.242), tg512 -0.464% (-0.587..-0.303). Evidence: docs/evidence/2026-09-10-rha04-gpu1-full/. Still exploratory/unadmitted; GPU2/GPU3 and dual-XTX 27B remain required.

GPU2 is complete: 18/18 current-source stock/native/replay cells across six balanced rounds, verified gfx1201/0000:09:00.0 identity, clean SIGINT teardown. Medians: stock 1240.345/2246.085/67.745/68.255, native 1388.285/2793.680/67.815/68.095, replay 1388.955/2791.335/67.825/68.310 (pp512/pp2048/tg128/tg512). Direct replay-vs-native bootstrap effects: pp512 +0.451% (-0.438..+1.496), pp2048 -0.717% (-2.500..+0.322), tg512 +0.409% (+0.169..+0.684). The stock pp anomaly is explicitly classified as an execution/build anomaly, not a tuning gain. Evidence: docs/evidence/2026-09-10-rha04-gpu2-full/. GPU3 and dual-XTX 27B remain required.

GPU3 is complete: 18/18 current-source stock/native/replay cells across six balanced rounds, verified gfx1030/0000:17:00.0 identity, clean SIGINT teardown. Medians: stock 905.130/1281.160/55.260/55.815, native 892.860/1279.880/55.290/55.700, replay 888.725/1281.450/55.250/55.785 (pp512/pp2048/tg128/tg512). Direct replay-vs-native bootstrap effects: pp512 -0.800% (-2.299..+0.482), pp2048 +0.018% (-0.210..+0.177), tg512 +0.033% (-0.251..+0.318). Evidence: docs/evidence/2026-09-10-rha04-gpu3-full/. All single-GPU 9B cells are now complete; matched dual-XTX 27B remains required.

The matched dual-XTX 27B workload was run in explicit observe mode after the required capture failed closed on missing ordered physical locators under -sm tensor/--fit off. All 18 observe cells completed with clean SIGINT teardown but all have execution_evidence_status=missing, so no decision-grade performance claim is made. Medians: stock 929.250/1287.315/34.000/34.190, native 929.815/1286.975/33.870/34.100, replay 929.065/1286.185/34.175/34.265 (pp512/pp2048/tg128/tg512). Direct replay-vs-native bootstrap effects: pp512 -0.056% (-0.377..+0.298), pp2048 -0.024% (-0.185..+0.131), tg512 +0.657% (+0.411..+0.990). Evidence: docs/evidence/2026-09-10-rha04-27b-dual-observe/. RHA04 remains open until a dual-device physical attestation path is implemented or explicitly accepted by review.

The current-source matrix summary now consolidates all four attested 9B topologies and the dual-XTX 27B observe-only result with full medians, paired replay/native effects, bootstrap intervals, and explicit admission limitations. The four 9B runs provide 72/72 physically attested cells; the dual-XTX run provides 18/18 engineering cells but remains blocked on ordered dual-device physical attestation.

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

Supersedes: HI168
Migration: capability-rebaseline-v3-2026-09
Successor key: run-hip-autotune-hi168

Inherited HI168 bundle is exploratory only because physical execution attestation is missing. The first fresh required-attestation GPU0 cell also failed closed before measurement because the production-shaped -sm none/--fit off launch emitted no physical locator evidence; see review RV157 and docs/evidence/2026-09-10-rha04-gpu0-required-capture/. Do not weaken required evidence to observe. The identity-bound preflight is implemented in tools/bigcherry/campaign/benchmark.py and deliberately does not promote KFD observation to authoritative identity or alter timed diagnostics/server arguments.

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
- 2026-09-09T14:46:18.213220+00:00 (updated-by): Updated: section:files, section:validation
- chg_20260909_144630_recorded-the-latest-bc-native_4039
- 2026-09-09T14:46:30.799268+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:02:13.405913+00:00 (updated-by): Updated: section:files, section:validation, section:notes
- chg_20260909_150304_the-first-required-attestation_2781
- 2026-09-09T15:03:04.213387+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:40:30.521435+00:00 (updated-by): Updated: section:validation
- chg_20260909_154042_hardened-production-benchmark_7289
- 2026-09-09T15:40:42.183106+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_155359_fixed-attestation-preflight-te_4726
- 2026-09-09T15:53:59.349146+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T15:55:53.136139+00:00 (updated-by): Updated: section:files, section:validation
- chg_20260909_155626_verified-the-new-physical-gpu_5643
- 2026-09-09T15:56:26.937731+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T16:00:33.021296+00:00 (updated-by): Updated: section:validation
- chg_20260909_160047_validated-that-the-new-physica_8483
- 2026-09-09T16:00:47.461671+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T16:17:02.558123+00:00 (updated-by): Updated: section:validation
- chg_20260909_161714_completed-and-retained-the-ful_7897
- 2026-09-09T16:17:14.271279+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T16:31:42.399424+00:00 (updated-by): Updated: section:validation
- chg_20260909_163156_completed-and-retained-the-ful_5078
- 2026-09-09T16:31:56.222599+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T16:47:05.627763+00:00 (updated-by): Updated: section:validation
- chg_20260909_164716_completed-and-retained-the-att_3548
- 2026-09-09T16:47:16.871836+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T17:03:43.561153+00:00 (updated-by): Updated: section:validation
- chg_20260909_170355_completed-and-retained-the-att_8516
- 2026-09-09T17:03:55.099040+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T17:24:26.558009+00:00 (updated-by): Updated: section:validation
- chg_20260909_172439_completed-the-dual-xtx-27b-eng_9789
- 2026-09-09T17:24:39.722856+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T17:25:45.277109+00:00 (updated-by): Updated: section:files, section:validation
- chg_20260909_172557_added-the-full-reproducible-ma_5709
- 2026-09-09T17:25:57.323123+00:00 (updated-by): Updated: section:ledger-events
