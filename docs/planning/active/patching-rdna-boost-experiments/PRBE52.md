---
id: PRBE52
order: 0
plan: patching-rdna-boost-experiments
state: done
created-at: '2026-09-09T10:57:05.575180+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# PRBE52: adaptive MTP depth — terminal b11474 disposition

## Decision (2026-10-13)

**Done, no active implementation or hardware queue.** BigCherry PR #51, merged 2026-10-09, already rejected both 1255 and 1268 after hardware qualification on b11474. Their patch.toml files both say state=rejected and the adaptive-mtp experiment definitions and spent queues were removed. This supersedes this item's old pending wiring, WHIRL calibration and hardware queue instructions. Do not re-enable either patch based on historical optimistic b11402 screens. Preserve the historical investigation in Git history and releases/evidence/prbe52-adaptive-no1210-source-audit.md.

**Measured BigCherry evidence:** One-binary, activation-attested Brutus ABBA runs: Flash-Next fixed depth 3 was 86.0–87.1 t/s (8K) and 70.0–71.4 t/s (98K). Adaptive cap 3/floor 2 was 76.3–80.9 and 62.3–67.5 t/s; cap 5/floor 3 was 76.8–78.0 and 63.5–64.4 t/s. The 24K cap 3/floor 2 results overlapped control. Dual-XTX Qwen3.8-27B Q8_0 adaptive cap 4/floor 2 was 64.8–65.4 vs fixed 71.4–74.6 t/s (10K) and 66.4–66.8 vs 72.9–73.3 (32K). These are historical measurements from 1268/SUMMARY.md, not new benchmarks. No tested adaptive configuration beat the best fixed depth.

**Correctness limits:** Depth-changing greedy outputs differed, sometimes between adaptive repeats. Fixed depth 3 versus 4 also changed text without the adaptive patch; differing kernel arithmetic by verify-batch shape is a hypothesis, not a proven sole cause. One earlier hipGraphExecDestroy illegal access was not root-caused and did not recur in later screens. Do not interpret altered tokens, dropped verification work or an unexplained crash as an optimisation.

## Implementation and ownership

- 1255_nro06_adaptive_mtp_depth/patch.py injects bigcherry_nro06_adaptive_mtp in common/speculative.cpp. It has per-sequence depth and a 32-drafted-token window: <=60% acceptance drops depth, >=72% climbs, reset near depth 3. This is acceptance-only hysteresis, not a measured E(tokens)/T(cycle) controller. It is inert without 1268.
- 1268_prbe52_adaptive_mtp_wiring/patch.py owns the disabled-by-default n_min_adaptive CLI/env field in common/common.h and common/arg.cpp, per-sequence draft cap/state in common/speculative.cpp, effective server budget, verify_h_rows-1 feedback, pending_h checkpoint get/set and forced-front/look-ahead interaction. Its manifest requires 1255 and 1321, not 1210. The implementation remains rejected, not production.
- 1321/1322 own look-ahead/promotion; PRBE20/1210 owns verify-batch numerical route qualification; FMTP owns its separate pipeline. No additional dispatcher, scheduler, cache, graph manager or telemetry framework is warranted.
- Production Flash-Next uses gfx1100 dual XTX plus gfx1201 R9700 target and gfx1030 6900 XT draft, with no ordinary GPU P2P. The 27B control uses dual gfx1100. Gains on external 5090/dual R9700 cannot be transferred without hardware qualification.

## Upstream and fork reconciliation

- llama.cpp PR #29924 merged 2026-10-04; pinned b11552 (23b0202a189c44a54625aadcb37a946dd1d6278d) already guards candidate-vector truncation with dp.result_q && !dp.result_q->empty() in common/speculative.cpp. Temperature>0 n-gram truncation is **not** a missing prerequisite or adaptive-depth benefit.
- llama.cpp PR #27210 is still open (updated 2026-09-11). Its actual diff adds draft-mtp-adaptive and accept_partial feedback for checkpoint-restored rounds. Use as a correctness reference, not a reason to port another controller; it recommends max depth >=7 whereas local experiments used caps 3–5.
- WHIRL src/model/spec.cpp has DraftAccept, DraftTiming, pickDrafts, CycleCost and allocDrafts; its E/T mechanism requires measured per-depth cycle cost and acceptance. BigCherry traces cannot supply unobserved counterfactual outcomes. No WHIRL runtime integration is authorised.
References: https://github.com/ggml-org/llama.cpp/pull/29924 ; https://github.com/ggml-org/llama.cpp/pull/27210 ; https://github.com/tsaipifong/whirl-llm/blob/main/src/model/spec.cpp .

## Reopen-only bounded gate

1. Require a genuinely new workload/architecture measurement and 12h independent inactivity; no competing queued/running speculative lane. Reuse 1317/1318 timing evidence, not a new profiling system.
2. Measure each proposed depth k in the same model, quant, context, request mix and physical GPU topology. Persist k, drafted, accepted, cycle/draft/verify microseconds, graph UID, verify rows and source/build. Split calibration and held-out rounds; never synthesize acceptance or cost for unobserved k. Close if predicted gain over **best fixed k** is <5% or measured E2E ceiling <3%.
3. Require multi-request same-process, np=1/2, warm checkpoint restore, 8K/24K/98K, fixed-shape numerical baseline, full logits/KLD and MTP acceptance, graph on/off, memory safety and no missing verification. Resolve a reproducible graph-destruction fault first.
4. Prefer upstream speculative ownership if reimplementation is justified; keep default off. Hardware promotion: >=4 independent sessions per eligible architecture, >=10 balanced paired rounds/session, CI95-low >=3% effective decode improvement over best fixed depth, <=1% prefill/control regression and full correctness. Otherwise remain done.

## Audit validation

2026-10-13: 15/15 actual-source assertions passed (patch status/dependencies, controller/reset, per-sequence wiring, pending_h checkpoint, verify-row feedback, pinned upstream guard). No repository pytest, patch application, HIP build, GPU run or new performance result. BCOP186 records the thin disposition.
