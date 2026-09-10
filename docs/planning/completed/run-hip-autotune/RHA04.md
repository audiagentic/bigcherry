---
id: RHA04
order: 0
plan: run-hip-autotune
state: completed
created-at: '2026-09-09T10:49:44.318293+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# Establish production native parity and measure tuned dispatch end to end

## Description

Runtime/parity qualification remains in progress. The four single-GPU 9B identity-bound stock/native/replay matrices are complete and provide exploratory per-topology evidence; the matched dual-XTX 27B run is observe-only because tensor-split logs lack ordered physical locators. Decision-grade dual-XTX parity and any production admission remain open. Reusable remaining BC build-type coverage is split to RHA08 by review RV160.

The identity-bound dual-XTX 27B Q8_0 matrix is now complete: six balanced permutations × stock/native/replay (18/18), all clean and verified. The four single-GPU 9B matrices remain complete. The capture establishes physical RCCL identity and parity evidence, while production admission remains open for source/work-equivalence provenance and replay activation/final-tuned proof. Reusable BC build-type coverage remains split to RHA08.

## Steps

1. Resolve model/build/topology from config/environment.toml and BC_* variables; record immutable build plans, source revisions, compiler/generated-input bindings, model identity, GPU identity, topology, runtime settings, dispatch mode, cache identity, and diagnostics state.
2. Use the maintained server-bench endpoint runner documented in docs/reference/testing/TEST.md; do not use llama-bench. Verify graceful shutdown and retain raw runner output plus machine-readable cell evidence.
3. Completed: the identity-bound 9B Q6_K matrix on each available physical GPU (stock/native/replay), with diagnostics-off production timing separated from diagnostic evidence.
4. Run a fresh matched dual-XTX 27B Q8_0 stock/native/replay matrix after the 9B matrix, with physical attestation, balanced/interleaved ordering, uncontended GPUs and the same production-shaped workload.
5. Analyze pp and generation independently with per-cell samples, ordering/position effects, uncertainty, correctness, replay activation, final tuned-launch counts, cache compatibility and work-equivalence. A missing proof is not a pass; diagnostic throughput is never a production performance claim.
6. If native regression is established, bisect the common production path and dispatch overhead before changing winner policy; repeat the matched contrast after each accepted fix. If tuned replay is neutral or slower, retain native fallback rather than forcing promotion.
7. Remaining reusable BC build-type coverage is tracked under RHA08 and must not be silently counted as RHA04 parity evidence.
8. Only after the complete identity-bound dual-XTX matrix and validation gates pass may RHA04 be completed.

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
docs/evidence/2026-09-10-rha04-matrix-summary/
docs/planning/completed/run-hip-autotune/reviews/RHA04/RV160.md
successor-specs/run-hip-autotune-hi168.md

docs/evidence/2026-09-10-rha04-required-six/README.md
docs/evidence/2026-09-10-rha04-required-six/run.json
docs/evidence/2026-09-10-rha04-required-six/pair-*/

## Validation

Required-six identity-bound dual-XTX matrix is complete and identity-verified with diagnostics isolated. RHA10/RHA11 admission evidence now proves source/work equivalence, corrected replay activation with 79 entries and zero misses/unavailable/rerun/incompatible rows, positive final tuned launches, clean diagnostics-off production timing, correctness, and graceful teardown. The completed admission record is docs/evidence/2026-09-10-rha10-admission-gate/replay-diagnostic-activation-rha11-corrected.json.

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

Supersedes HI168. Migration: capability-rebaseline-v3-2026-09. Four single-GPU 9B matrices are identity-attested and retained, but the dual-XTX 27B timing is observe-only because tensor-split --fit off emits no ordered physical locators. Do not weaken required attestation or infer parity from observe-only numbers. Review RV160 split reusable BC build-type coverage into RHA08; RHA04 now owns only decision-grade native/replay parity and dual-XTX attestation.

Supersedes HI168. Migration: capability-rebaseline-v3-2026-09. Four single-GPU 9B matrices are identity-attested and retained, but the dual-XTX 27B timing is observe-only because tensor-split --fit off emits no ordered physical locators. Do not weaken required attestation or infer parity from observe-only numbers. Review RV160 split reusable BC build-type coverage into RHA08; RHA04 now owns only decision-grade native/replay parity and dual-XTX attestation. Dev GPT review req_7944dc1be1fd4173 recommends the RCCL preflight composition and confirms selectors/KFD must not be promoted to identity. Implementation is present but unverified against the live Brutus RCCL log until the exact untimed probe runs.

Fresh required dual-XTX 27B matrix is complete and identity verified. RCCL preflight diagnostics are isolated from timed production environment. The results show native within roughly +/-0.7% of stock on these medians and replay within roughly +/-0.4%; pp512 stock includes one retained host-side outlier. Do not call this production admission until correctness/work-equivalence, source/build provenance, and replay activation/final tuned-launch evidence are independently satisfied.

Final production-admission proof is split to RHA10. RHA04 retains the completed identity-bound parity matrix and remains in_progress until RHA10 proves source/work equivalence and final tuned replay launches. RHA08 and RHA09 are completed evidence dependencies, not admission substitutes.

RHA11 completed correctness quarantine: two replay winners were replaced with native through a manifest-valid cache export. Corrected cache is canonical on Brutus with the original retained as dispatch.cache.pre-rha11. Corrected diagnostics activation and diagnostics-off timing are retained under docs/evidence/2026-09-10-rha11-correctness. RHA04 admission still follows RHA10's explicit miss-policy decision.

Supersedes: HI168\nMigration: capability-rebaseline-v3-2026-09\nSuccessor key: run-hip-autotune-hi168\n\nThe identity-bound 9B per-GPU matrices and required dual-XTX 27B stock/native/replay matrix are complete. RHA10 and RHA11 are now complete: the two divergent tuned winners were quarantined, 24 previously uncovered native signatures were explicitly seeded through the supported exporter, and the resulting 79-entry cache is canonical on Brutus. Final replay activation recorded 21,566/21,566 coverage, 57 exact matches, zero misses/unavailable/rerun/incompatible rows, and positive final tuned launches. Diagnostics-off production timing returned 0 at pp512 927.62, pp2048 1289.19, tg128 33.71, tg512 34.05 t/s. The corrected three-prompt work-equivalence corpus matched stock/native on all prompts. Reusable BC build-type evidence remains under RHA08 and durable raw lifecycle evidence under RHA09.

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
- 2026-09-09T18:21:27.940700+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:notes
- chg_20260909_182139_the-plan-now-separates-the-dua_5938
- 2026-09-09T18:21:39.235727+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T18:28:09.363414+00:00 (updated-by): Updated: section:validation, section:notes
- chg_20260909_182824_the-dual-xtx-attestation-gap-n_9053
- 2026-09-09T18:28:24.784168+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_183306_the-live-probe-confirmed-tenso_1229
- 2026-09-09T18:33:06.166444+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_183551_the-capture-now-waits-for-the_3770
- 2026-09-09T18:35:51.536858+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T19:12:33.995235+00:00 (updated-by): Updated: section:description, section:files, section:validation, section:notes
- chg_20260909_191246_the-fresh-dual-xtx-27b-run-is_8236
- 2026-09-09T19:12:46.413981+00:00 (updated-by): Updated: section:ledger-events
- chg_20260909_191424_timed-benchmark-arms-now-fail_8101
- 2026-09-09T19:14:24.974787+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T19:45:51.724390+00:00 (updated-by): Updated: section:validation, section:notes
- chg_20260909_194609_separated-completed-parity-cap_7282
- 2026-09-09T19:46:09.191178+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T20:45:19.134130+00:00 (updated-by): Updated: section:notes
- chg_20260909_204705_the-quarantined-cache-is-now-t_2378
- 2026-09-09T20:47:05.979876+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-09T21:00:40.650761+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-09T21:00:51.165160+00:00 (state-transition): State: in_progress → completed
- chg_20260909_210128_the-remaining-replay-misses-ar_2991
- 2026-09-09T21:01:28.031001+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.452931+00:00 (updated-by): Updated: section:ledger-events
