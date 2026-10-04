---
id: QFP15
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T04:00:27.523971+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Run-to-run nondeterminism in Flash-Next decode (draft acceptance varies with identical greedy output)

## Description

Same build, same prompt, temperature 0: greedy target output is identical, but MTP drafted/accepted counts differ run to run (2026-10-04 v3-1313b 80K: base arms 171/252 vs 170/255; 1313 arm 169/258). The target and the MTP draft sampler are both greedy (draft = top-1 after top-k 10 at common/speculative.cpp, no RNG; seed irrelevant), so some GPU result differs bitwise between runs and flips a draft near-tie or the p_min cutoff. Consequences: single-request quick screens carry ~+-2% t/s noise from acceptance alone (masked 1313's ~2-3% ms/step gain at 80K); speculative identity checks cannot be exact.

## Steps

1. Patch 1315_mtp_draft_trace (BIGCHERRY_DRAFT_TRACE=1): per draft step top id, p as %a, FNV-1a of the draft hidden row; per accept the FNV-1a of the target hidden row (pending_h). Run the same build three times on the same ~24K prompt (queue-determinism.sh) and diff the trace lines.
2. If the first divergence is an accept target_h hash: the target is nondeterministic. Bisect with a per-node output hash (ggml eval callback) at that step across two runs; first differing node = culprit. If only draft step lines diverge with identical target_h: the 6900 draft path is nondeterministic; same bisection on ctx_dft.
3. Fix by suspect class: (a) prefill GEMM library nondeterminism (hipBLASLt/rocBLAS split-k atomics / algorithm selection) -> deterministic env/flags; (b) RCCL large-message AllReduce order during prefill -> fixed-order path or deterministic RCCL algorithm; (c) a kernel combining partials with float atomics (attention split/parallel blocks, GDN, MoE reductions) -> fixed-order reduction. Already deterministic: 1291 small cpu-root AllReduce (fixed rank order), 1294 TOP_K ties.
4. Re-run step 1: all three runs must match bit-exactly (drafted and accepted counts identical).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Three runs of one build: identical BIGCHERRY_DRAFT_TRACE streams, identical drafted/accepted counts and greedy text at ~24K and ~80K; any fix shows no t/s regression (ms/step ABBA).

## Effort & Risk

Diagnosis ~0.5-1 day (instrumentation + node-hash bisection). Fix S (env flag) to M (kernel reduction order). Risk: a deterministic reduction can cost a little speed; measure.

## Standards



## Acceptance Criteria



## Notes

Payoff beyond clean benchmarks: exact spec-vs-nospec and spec-vs-spec identity checks; 1-2% wins provable from short runs instead of long ABBAs. Until fixed, judge launch-reduction screens by ms/step and confirm with multi-request ABBA. Related: QFP13 (screens affected), FMTP01 Gate 0 (acceptance statistics need deterministic runs).

## Change Log

- 2026-10-04T04:00:27.523971+00:00 (created-by): Created by agent
