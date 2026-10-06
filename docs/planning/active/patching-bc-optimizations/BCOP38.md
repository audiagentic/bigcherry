---
id: BCOP38
order: 38
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-06T11:07:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Disposition WHIRL adaptive speculation through PRBE52

## Description

2026-10-06 optimisation-audit disposition for WHIRL v0.1.3 adaptive speculative decoding. PRBE52/1255/1268 remain the authoritative BigCherry front-draft-depth owner. FMTP05 remains a separate ahead-overlap economics owner and this item does not resume the paused FMTP02-FMTP07 pipeline.

## Provenance and ownership

The adaptive-depth lineage is:

`nasone32/llama.cpp-RDNA3-7900xtx-opt@10579a7365a3bc86c4f8e41aaab20e73e1571e5e -> BigCherry 1255 acceptance heuristic -> 1268 runtime integration -> candidate WHIRL E(tokens)/T(cycle) refinement`.

WHIRL is therefore not the origin of BigCherry adaptive MTP. Its relevant contribution is a measured cost model: conditional per-position acceptance, measured cycle time by draft count, 3% hysteresis and bounded neighbour probes.

Single-owner rules:
- 1255 owns the pure front-draft-depth controller.
- 1268 owns runtime wiring/configuration for that controller.
- FMTP05 may consume effective front depth but owns only future/replay work scheduled under an already-submitted target verification window.
- Do not add a second front-depth controller, scheduler, n-gram implementation, CLI surface or dispatch table.

## Current blocking facts

1. **1268 is not runnable at the current active source pin.** Its disposition is `known_broken / FAILED_NEEDS_RECONCILIATION`: upstream probabilistic-MTP changes broke the begin-reset, draft-reset and depth-limit anchors and 1268 is not in the build recipe. Reconcile 1268 before any WHIRL policy experiment.
2. 1210 remains a mandatory correctness prerequisite. Historical 1268 performance evidence is motivation only: its validation contract failed greedy correctness and must not be reused as promotion evidence.
3. 1317/1318 are useful timing instrumentation but do **not** contain counterfactual cycle time or acceptance for depths that were not executed. Existing traces cannot faithfully replay arbitrary fixed/current/WHIRL policies.
4. 1317 timing is file-static and documented for single-slot interpretation. Initial calibration must use `parallel=1`; multi-user/concurrency conclusions require a separate scheduling experiment.
5. FMTP03 hardware work already showed why acceptance alone is insufficient: ahead execution reduced step time about 5-8% in representative lanes while effective throughput stayed approximately neutral because promoted-front acceptance fell. This is internal evidence for optimizing accepted target tokens per wall-clock cycle.
6. FMTP02-FMTP07 are paused by owner decision. This item does not authorize or implicitly restart them.

## Required execution order

### Gate 0 - reconcile the existing owner

1. Rebase/reconcile `1268_prbe52_adaptive_mtp_wiring` against the current probabilistic-MTP source.
2. Restore apply/idempotence/composition tests with 1255 + 1210.
3. Prove adaptive-off is behaviorally equivalent to fixed-depth control.
4. Pass the 1210 greedy-token identity gate and repeated same-process correctness before collecting policy performance evidence.
5. Resolve upstream #29924/equivalent before interpreting temperature>0 mixed MTP+n-gram lanes.

If Gate 0 fails, stop BCOP38. Do not work around it with a new controller.

### Gate 1 - collect an identifiable depth calibration set

Use `parallel=1`. Deliberately exercise every candidate front depth used by the policy, initially `k in {1,2,3,4}` or the current supported range. For each round persist at minimum:

`{session, workload, context_bucket, k, n_drafted, n_accepted, cycle_us, draft_us, target_submit_us, target_sync_us, process_us, sample_us}`.

Requirements:
- cycle boundaries must refer to the same speculative round;
- retain raw 1317/1318 logs and parser output;
- enough observations per `k` and workload/context segment to estimate `T(k)` and conditional acceptance;
- do not synthesize an unobserved depth's timing from another depth;
- do not call this trace replay unless the dataset actually contains observations for every compared action.

### Gate 2 - offline discriminator

Fit/evaluate:
1. best fixed depth per held-out workload/context segment;
2. current 1255 climb/drop policy;
3. WHIRL-style `argmax_k E(tokens|k)/T(cycle|k)` with 3% hysteresis and bounded neighbour probing.

Use train/calibration rounds for policy estimates and held-out rounds for scoring. Score accepted target tokens per measured wall-clock cycle and policy regret versus the best fixed action. Report uncertainty/sample counts, not only point estimates.

This is a policy-model discriminator, not proof of counterfactual hardware performance. If predicted held-out advantage over current 1255 is <5%, close BCOP38 without changing runtime code.

### Gate 3 - modify only 1255

Only after Gate 2 >=5%, replace the internals of the existing 1255 controller with the E/T policy. Keep 1268's existing opt-in/configuration surface. No parallel mode or new controller is permitted.

### Gate 4 - hardware qualification

gfx1201 first, then gfx1100. Compare fixed-best, current adaptive and E/T adaptive using ABBA >=5 repetitions across coding, prose, repetitive/tool-call and representative 8K/64K/128K contexts.

Require:
- exact greedy IDs;
- multi-request same-process correctness;
- >=5% median effective-TG improvement over current adaptive;
- <=2% regression in every held-out representative lane;
- no swap/OOM/semantic regression.

If hardware fails these gates, retain current 1255.

## Mixed MTP+n-gram boundary

WHIRL's n-gram co-drafting is a separate proposer-selection mechanism. llama.cpp already supports mixed proposers. Use upstream behavior as the control after #29924/equivalent. Do not implement local proposer arbitration unless a later measured residual >=5% remains after the front-depth experiment.

## Terminal disposition

- **Promote:** Gates 0-4 pass.
- **Reject/close:** Gate 2 predicts <5%, Gate 4 fails, or correctness fails. Preserve WHIRL as evaluated external evidence.
- **Blocked:** 1268 cannot be reconciled at the current pin. Fix/rebase the existing owner; do not create a replacement controller.

## Provenance

- https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt commit 10579a7365a3bc86c4f8e41aaab20e73e1571e5e
- https://github.com/tsaipifong/whirl-llm
- https://github.com/tsaipifong/whirl-llm/blob/main/src/model/spec.cpp
- https://github.com/tsaipifong/whirl-llm/blob/main/docs/guide/en/speculative-decoding.md
- https://github.com/ggml-org/llama.cpp/issues/24507
- https://github.com/ggml-org/llama.cpp/issues/23184
