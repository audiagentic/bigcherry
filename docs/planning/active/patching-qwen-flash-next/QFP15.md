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

## 2026-10-14 audit: seed provenance required before attributing MTP nondeterminism

**Authoritative disposition:** pending/diagnostic-only. The first differing `BIGCHERRY_DRAFT_TRACE step` line **cannot identify target versus draft causality**: `1315_mtp_draft_trace/patch.py` hashes the draft row after `common_sampler_sample` and `llama_get_embeddings_nextn_ith`, but hashes the target `pending_h` only *after* `accept()`. The first draft input seed from `common_speculative_impl_draft_mtp::process` is not logged. Either a different pre-draft target seed or a different draft computation can produce the same first differing step. `accept()` selects `verify_h[min(n_accepted,n_rows-1)]`, so a changed acceptance count can alter the next seed without changing individual target rows. This is a trace-identifiability defect, not a reproduced GPU race.

**Hash contract:** `bc_fnv1a` uses offset 1469598103934665603, not standard FNV-1a 14695981039346656037. Legacy same-helper hashes remain internally comparable, but are not canonical FNV-1a and do not prove byte identity. The archived 'stock-none' run retained BigCherry base overlays; pure upstream remains untested. Fixed-order CPU-root routing failing to eliminate the observed variation rejects RCCL as the sole cause of that *run*, not every RCCL effect. 1316 per-node callbacks violate tensor-split Meta graph boundaries and must not be reused.

**Source ownership:** pinned b11552 `common/speculative.cpp::{process,draft,accept}`; existing 1315 owns optional trace only. 1291/1294/1303/1321/1359, QFP07/QFP13 and FMTP retain provider, sampler, attention placement, frontier, handoff and benchmark ownership. No new sampler, graph callback, allocator, telemetry registry or hardware queue.

**Bounded gates:** (0) inspect completed receipts for fixed source/patch/model, prompt, request order, physical target/draft UUID, context, `n_ubatch`, MTP depth, `p_min`, graph and KV settings; missing identity => `UNATTRIBUTABLE`. (1) Only if necessary extend default-off 1315 at the existing process/draft boundary to correlate `{request,seq,round,pos,token,pre_draft_seed_hash,seed_source_row,accept_n_rows,accept_i_h,draft_device}` with existing step and accept records. No new synchronization, context-owned pointer retention or graph changes; version any corrected hash. (2) Once seed identity is proven, compare three repeated same-process requests and three fresh starts each for gfx1030 versus gfx1100 draft at 8K/24K/80K, fixed target placement and draft/target ubatch, model, sampler and MTP settings. Use existing ABBA/trace harness only after Radiance hardware lanes are idle. Hash alone is insufficient: use exact rows/full logits where causality requires it.

**Terminal:** `UNATTRIBUTABLE`, `TARGET_SEED_DIVERGES`, `STABLE_SEED_DRAFT_DIVERGES`, `ACCEPT_ROW_SELECTION`, `UPSTREAM_REPRO` (only genuinely unpatched same-pin control), or `BENCHMARK_NOISE_ONLY`. No kernel fix unless first divergent producer and stable A/A control are proven. Require multi-request/ubatch, long-context, greedy/logit/MTP parity, graph lifetime, complete work accounting and <=1% ms/step regression.

**Upstream/forks:** llama.cpp [#30257](https://github.com/ggml-org/llama.cpp/pull/30257) already supplies MTP state propagation in b11552. [#30459](https://github.com/ggml-org/llama.cpp/pull/30459), merged 2026-10-14, adds a draft-only `n_ubatch` control *after* b11552; wait for a qualified pin and treat it as a workload-shape confounder, not a determinism fix. [vLLM speculative metrics](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/metrics.py) separates drafted and accepted counts by position; reuse the measurement concept in existing receipts, not its kernels. [SGLang EAGLE worker](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_worker_common.py) documents hidden-state cross-stream lifetime dependencies, not an AMD performance result.

**Validation:** 4/4 host-only trace-identifiability assertions plus Python compilation passed; source/PR diffs inspected. No repository pytest, HIP build, GPU execution, benchmark or new measured improvement. Historical 2026-10-12 eight-start rescreen remains the measured benchmark-noise reference. Older steps/notes below are historical hypotheses superseded where inconsistent with this disposition.

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



### Re-screen of verdicts taken from single quick-ab readings (2026-10-12, pin b11552, base build b-m1340)

Why: quick-ab's 256-token single readings swing between ~38.6 and ~42.4 ms/step from one server start to the next for the same binary, so a one-ABA 'neutral' or 'few percent off' was never a measurement. Method (tools/lab/flash-next/queue-rescreen.sh): Flash-Next 24K, 512 decode tokens, eight server starts per arm (build candidates in alternating blocks of four; flags A B B A A B B A on one build).

| candidate | decode ms/step, mean (min-max) | acceptance | prefill t/s | greedy text | verdict |
|---|---|---|---|---|---|
| 1205 paired K/V projection (build) | 39.10 (38.0-41.0) vs base 38.71 (38.0-39.5) | 56.1% vs 55.7% | 1455 vs 1453 | identical | neutral, as screened |
| 1206 mul_mat + add view fusion (build) | 39.14 (38.0-40.5) vs base 38.78 (38.0-40.3) | 56.1% vs 55.8% | 1449 vs 1456 | identical | neutral, as screened |
| BIGCHERRY_DRAFT_VOCAB_N=32768 (flag, 1297) | 38.47 (37.8-39.1) vs 38.67 (38.0-39.5) | 52.5% vs 56.2% | 1456 vs 1452 | identical | the screen's '-10% ms/step' does not reproduce; acceptance is lower, so 65536 stays |
| 1306 shared-expert split (flag) | 37.32 (36.6-38.2) vs 39.00 (38.0-40.3) | 63.9% vs 55.7% | 1457 vs 1456 | differs | looked like +4% on this one prompt - see below |

1306 over four prompts (queue-decode-acceptance.sh on build b-rs-1306, ABBA per prompt): decode +13.7%, -8.7%, -6.0%, -1.7%; pooled 69.1 vs 69.8 t/s (-0.9%), acceptance 54.6% vs 55.5%, prefill -0.1%, faster in 1 of 4. The one-prompt gain was the continuation that prompt happened to take once the text changed. 1306 is speed-neutral; its only measured effect is the VRAM it frees (~185 MiB per XTX, ~135 MiB on the R9700 in the 2026-10-04 notes). It stays `evaluated`.

So all four earlier verdicts stand, now with measurements behind them. Two rules that follow: a change that alters the output text is judged on several prompts, never one; and a change that keeps the text is judged on several starts per arm. Not re-screened: 1298, 1301, 1304, 1245, 1262, 1215, 1255 (rejected, no experiment recipe left to build them).

## Change Log

- 2026-10-04T04:00:27.523971+00:00 (created-by): Created by agent
- 2026-10-04T05:06:40.485818+00:00 (updated-by): Updated: section:notes
- 2026-10-04T05:47:30.490172+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:14:06.301778+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:33:33.339610+00:00 (updated-by): Updated: section:notes
- 2026-10-04T06:54:25.842364+00:00 (updated-by): Updated: section:notes
- 2026-10-04T07:55:38.201733+00:00 (updated-by): Updated: section:notes
- 2026-10-05T07:04:00+11:00 (updated-by): Deep review: consolidate diagnosis on draft-placement and pure-upstream controls; forbid meta-backend graph-splitting instrumentation.
- 2026-10-12T03:41:17.231762+00:00 (updated-by): Updated: section:notes
