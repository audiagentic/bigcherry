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

2026-10-04 run 1 (flashnext-det-d24k, b-det = v3 + 1312 + 1313 + 1315/1316/1317, three runs of one binary): base-a and base-b traces are bit-identical through the first (warm-up 'Hello') request, including the target hidden hash (target_h=511799406ebb49c1); they diverge at the FIRST draft step of the ~24K request (id 1144 vs 1044) and the next target_h differs. So the nondeterminism arises in long-prompt processing (target and/or draft prefill), not in decode. The middle arm differs from line 1 (even the warm-up) - unexplained, check separately. Leading hypothesis: prefill-size AllReduces (> 64 KiB) go to RCCL, whose reduction order is not fixed; decode-size ones use 1291's fixed-order CPU sum. Test queued: queue-determinism2.sh with BIGCHERRY_AR_CPU_ROOT_LARGE_MAX_BYTES=256 MiB (all AllReduces through the fixed-order CPU sum). If still divergent: 1316 node hashes over the prefill graphs.

2026-10-04 run 2 (flashnext-det2-d24k, all AllReduce sizes through the fixed-order cpu-root CPU sum via BIGCHERRY_AR_CPU_ROOT_LARGE_MAX_BYTES=256 MiB): still nondeterministic, and now even the warm-up 'Hello' request differs: base-a starts with draft id 1942 (as in run 1), new and base-b with id 353; new vs base-b share the first draft round but the target accepted 1 vs 0 tokens with different target_h. RCCL is NOT the cause; the target computation itself differs on a tiny prompt with only small fixed-order AllReduces. Two discrete modes (1942 vs 353 starts) suggest a per-process choice made at startup (kernel/algorithm selection, tuning, library heuristics, device/stream assignment) rather than continuous float noise. Next: run 3 (queue-determinism3.sh) with 1316 BIGCHERRY_NODE_HASH=0:2 hashes every node of the first two graphs per context across three runs; the first differing node names the op.

## Change Log

- 2026-10-04T04:00:27.523971+00:00 (created-by): Created by agent
- 2026-10-04T05:06:40.485818+00:00 (updated-by): Updated: section:notes
- 2026-10-04T05:47:30.490172+00:00 (updated-by): Updated: section:notes
