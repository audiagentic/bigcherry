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

Diagnosis must now separate target and draft before adding more node-level instrumentation. Run the same target binary/model twice while changing only the draft device: current gfx1030 6900 XT versus one gfx1100 XTX. Pin all draft-side launch/config knobs and record the first draft top-id/probability plus target hidden hash. If the mode follows the draft device while target hashes remain stable, investigate gfx1030 QSA/indexer/TOP_K/reduction order. If it persists identically on XTX, move upstream to shared draft graph/state logic rather than adding architecture-specific fixes.

Use a true upstream control, not `[source.bigcherry]`: build an unpatched checkout at the current llama.cpp release pin (b11396 as of 2026-10-05 scan) and run the same no-overlay trace. This closes the run-5 attribution error and establishes whether the residual exists upstream before BigCherry spends code on it.

Do not revive 1316's per-node eval callback under `-sm tensor`; it changes graph execution boundaries and trips meta-backend invariants. For any later bisection, hash only selected materialized outputs at existing graph boundaries after compute, or add an observation hook that does not split/rebuild the graph.

## Code Samples & Guidance

Keep the diagnostic observational. Reuse `BIGCHERRY_DRAFT_TRACE` and existing device selection. Do not add another sampler or TOP_K implementation. For placement A/B, the only intended variable is the draft backend/device; target placement, prompt, context, MTP depth, p_min and target binary must remain identical.

## Files

- `docs/planning/active/patching-qwen-flash-next/QFP15.md`
- existing 1315 draft-trace patch/tooling and queue scripts; extend rather than create a second trace path
- `common/speculative.cpp` only if later evidence proves shared speculative state logic is the first divergent owner
- QSA/TOP_K backend code only if the divergence follows gfx1030

## Validation

Primary diagnostic matrix: 3 runs each for draft=gfx1030 and draft=gfx1100 at warm-up, ~24K and ~80K; compare target hidden hashes, draft top-id/probability, drafted/accepted counts and greedy text. Pure-upstream current-release control: at least 3 identical runs with no BigCherry overlay patch-set. Any eventual fix: identical trace streams across 3 runs at both contexts and no >1% median ms/step regression.

## Effort & Risk

Diagnosis ~0.5 day because it reuses 1315 and changes placement rather than adding graph instrumentation. Fix scope remains evidence-dependent. Avoid deterministic-kernel rewrites until the placement matrix identifies the owner; forcing deterministic reductions globally can cost throughput for a benchmark-quality issue that currently does not change production greedy text.

## Standards

QFP15 owns determinism diagnosis only. QFP13 owns performance acceptance/noise handling; FMTP01 owns speculative-policy tuning; QFP07 owns target attention placement; 1294 remains the single TOP_K tie-breaking implementation. Do not duplicate those mechanisms here.

## Acceptance Criteria

Diagnosis is complete when the residual is classified as target-side, draft-side gfx1030-specific, shared/upstream, or BigCherry-overlay-specific with a reproducing A/B. A code fix is promoted only if it removes the observed divergence without >1% median ms/step regression. If pure upstream b11396 reproduces only acceptance wobble while BigCherry remains greedy-stable, downgrade this item to benchmark hygiene and use multi-request ABBA/ms-step for <=3% optimisation claims instead of carrying a throughput-costly deterministic path.

## Notes

Payoff beyond clean benchmarks: exact spec-vs-nospec and spec-vs-spec identity checks; 1-2% wins provable from short runs instead of long ABBAs. Until fixed, judge launch-reduction screens by ms/step and confirm with multi-request ABBA. Related: QFP13 (screens affected), FMTP01 Gate 0 (acceptance statistics need deterministic runs).

2026-10-04 run 1 (flashnext-det-d24k, b-det = v3 + 1312 + 1313 + 1315/1316/1317, three runs of one binary): base-a and base-b traces are bit-identical through the first (warm-up 'Hello') request, including the target hidden hash (target_h=511799406ebb49c1); they diverge at the FIRST draft step of the ~24K request (id 1144 vs 1044) and the next target_h differs. So the nondeterminism arises in long-prompt processing (target and/or draft prefill), not in decode. The middle arm differs from line 1 (even the warm-up) - unexplained, check separately. Leading hypothesis: prefill-size AllReduces (> 64 KiB) go to RCCL, whose reduction order is not fixed; decode-size ones use 1291's fixed-order CPU sum. Test queued: queue-determinism2.sh with BIGCHERRY_AR_CPU_ROOT_LARGE_MAX_BYTES=256 MiB (all AllReduces through the fixed-order CPU sum). If still divergent: 1316 node hashes over the prefill graphs.

2026-10-04 run 2 (flashnext-det2-d24k, all AllReduce sizes through the fixed-order cpu-root CPU sum via BIGCHERRY_AR_CPU_ROOT_LARGE_MAX_BYTES=256 MiB): still nondeterministic, and now even the warm-up 'Hello' request differs: base-a starts with draft id 1942 (as in run 1), new and base-b with id 353; new vs base-b share the first draft round but the target accepted 1 vs 0 tokens with different target_h. RCCL is NOT the cause; the target computation itself differs on a tiny prompt with only small fixed-order AllReduces. Two discrete modes (1942 vs 353 starts) suggest a per-process choice made at startup (kernel/algorithm selection, tuning, library heuristics, device/stream assignment) rather than continuous float noise. Next: run 3 (queue-determinism3.sh) with 1316 BIGCHERRY_NODE_HASH=0:2 hashes every node of the first two graphs per context across three runs; the first differing node names the op.

2026-10-04 run 3 (1316 node hashes): incompatible with the tensor-split meta backend - the eval callback splits the graph per node and ggml-backend-meta.cpp:2229 asserts (i_start == cgraph->n_nodes); only the first 18 nodes (embedding .. hc_pre .. cache_r view) were hashed and they were identical across runs. 1316 needs a non-callback design for -sm tensor (e.g. hash selected graph outputs after compute). Startup logs of a run in each mode (det2 base-a vs base-b) are identical apart from port/timing, so the mode is not a startup choice. New leading hypothesis: stale/uninitialized device memory read (depends on VRAM left by earlier processes and on per-run allocation addresses) - prime suspect the 1235/1307 Q8_1 activation cache and its producers (address+generation keyed reuse). Note GGML_HIP_Q8_1_CACHE_MODE=verify is parsed but not implemented in the MMVQ path. Run 4 queued (queue-determinism4.sh): all v3/v4 runtime flags off; deterministic => bisect the flag-gated patches.

2026-10-04 run 4 (flashnext-det4-d24k, all v3/v4 runtime flags OFF: Q8_1 cache, RMS/ACT/HC producers, rollback no-CONT, scale-act fusion): still nondeterministic from the very first draft step of the warm-up (base-a id 1942; new id 353 p=0x1.03571ep-1; base-b id 353 p=0x1.172e36p-1). The flag-gated patches (1307-1313) are cleared. Remaining suspects: always-on patches in the build (1291 cpu-root AllReduce, 1292, 1294, 1297, 1302, 1303 attention split) or upstream itself. Run 5 queued (queue-determinism5.sh): stock build with no patches, RCCL, no attention split, CTX 65536.

2026-10-04 run 5 (flashnext-det5-d24k, stock build with NO BigCherry patches, RCCL AllReduce, no attention split, CTX 65536): upstream is MORE nondeterministic than our builds - the greedy TEXT differs across all three runs (md5 8fe14447 / 184aaffd / 8b3d504c) and warm-up acceptance differs (3/10, 3/9, 4/8). Our always-on patches reduce nondeterminism (1291 fixed-order CPU-root AllReduce and 1294 deterministic TOP_K ties make the greedy text identical); the residual draft-acceptance wobble in our builds is upstream behaviour not yet removed by them. Remaining suspects for the residual: the draft context on the 6900 (QSA/indexer top-k in the MTP layer - check whether 1294's deterministic ties cover the draft path and gfx1030), RCCL-dependent prefill ARs for long prompts, and any atomics-based reduction on the target's decode path. Priority lowered: our production build is already greedy-stable; the residual only adds +-2% single-request t/s noise (use ms/step or multi-request ABBA). Next if resumed: run 2-run traces with the 6900 draft swapped to a deterministic configuration (e.g. draft on an XTX) to split target vs draft.

CORRECTION 2026-10-04: the run-5 'stock-none' build was not pure upstream - [source.bigcherry] always includes the serving-core, upstream-fixes and validated-enhancements patch-sets (0840 hybrid AllReduce dispatch, 0860, 1225, 1237, 1241, 1253, 1274, 1265, ...). Run 5 therefore shows: base patch-sets WITHOUT 1291/1294 -> greedy text differs across runs; WITH 1291/1294 -> text identical, residual draft wobble. Pure-upstream behaviour is untested (would need a source without overlay patch-sets).

2026-10-05 optimisation review: stop broad nondeterminism bisection until two cheap ownership controls are complete: (1) draft placement gfx1030 -> gfx1100 with target held fixed, and (2) a genuinely unpatched upstream b11396 control. This consolidates the next work around existing 1315 tracing and avoids another meta-backend-unsafe node-hash mechanism. If the residual follows gfx1030, inspect draft QSA/indexer/TOP_K/reduction ordering; if it reproduces unchanged on XTX and pure upstream, treat it as upstream benchmark noise unless it changes greedy output. Current production greedy stability means deterministic rewrites must clear a stricter no-regression gate than ordinary performance patches.

## Change Log

- 2026-10-04T04:00:27.523971+00:00 (created-by): Created by agent
- 2026-10-04T05:06:40.485818+00:00 (updated-by): Updated: section:notes
- 2026-10-04T05:47:30.490172+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:14:06.301778+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:33:33.339610+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:54:25.842364+00:00 (updated-by): Updated: section:notes
- 2026-10-04T07:55:38.201733+00:00 (updated-by): Updated: section:notes
- 2026-10-05T07:04:00+11:00 (updated-by): Deep review: consolidate diagnosis on draft-placement and pure-upstream controls; forbid meta-backend graph-splitting instrumentation.
