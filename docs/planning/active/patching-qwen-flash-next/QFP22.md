---
id: QFP22
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-04T21:08:00.115435+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Cleanup + promotion validation: 1321/1322 MTP-ahead and 1332 QSA token chunks

## Description

Owner 2026-10-05: 1321, 1322 and 1332 look helpful; put them on the cleanup and promotion-validation list. Fixes go inside each patch (owner rule). Promotion uses the lightweight tier: adoption ABBA vs the promoted base (v6), activation evidence, offline tests, greedy check (vs f32 reference where a near-tie is involved, see QFP17).

## Steps

1321 + 1322 (MTP-ahead, promote together; 1322 requires 1321):
1. Cleanup: drop the tail_p_min / BIGCHERRY_MTP_AHEAD_PMIN knob (only made things worse) unless a variant needs it; keep default off.
2. Rework candidate before validation: promote only full-length fronts (tail shorter than a fresh draft -> fresh draft), per FMTP03 notes.
3. Validation: multi-request ABBA (8 arms/depth) at ~8K and ~64K on v6, BIGCHERRY_MTP_AHEAD=0/1; promotion stats line as activation; greedy identity. Current single-ABA evidence: per-step -7..9%, t/s +1-2% (within noise) - need the ABBA to show a real win.

1332 (QSA token chunks):
4. Fix inside 1332: no-MTP single-token decode after a chunked prefill segfaults in ggml_backend_meta_graph_compute (stale meta bookkeeping for views of the kq_mask input). Replace input views with ggml_get_rows of the chunk's kq_mask rows from a small per-chunk I32 row-index graph input (set_input hook), or fix the meta view bookkeeping.
5. Before choosing either representation, port/qualify upstream llama.cpp #29958's graph-shape invariant: reserve and runtime graphs must not branch on mutable cache/runtime state. Run with GGML_SCHED_DEBUG_REALLOC=1 and require one node/leaf count per context across prefill, first decode, cache sharing/reuse and QSA chunk tails.
6. Validation: no-MTP decode + MTP serving at 24K/80K; ub1024+chunk256 vs v6 ub512 prefill/decode ABBA at 80K fill and a short-prompt prefill (where larger ub should help most); greedy checked against the f32 reference at near-ties (QFP17).
7. Decide the production setting (ub1024 + BIGCHERRY_QSA_CHUNK=256 in a feature set) only if the ABBA shows a win.

8. Add winners to the flashnext feature set (new row in 0910, e.g. flashnext-v7) and promote via validated-enhancements.

## Detailed Solution & Technical Design

### 1332: make chunking a graph-shape boundary, not a view-lifetime trick

The crash signature says the optimisation is sound (ub1024 fits) but the representation is unsafe across meta-graph rebuild/reuse. Do not add another QSA chunk scheduler. Keep QFP17/1332 as the sole owner and change only how each chunk is represented in the graph.

Preferred experiment: materialize chunk row selection with `GGML_OP_GET_ROWS` from the canonical mask tensor. Add a tiny I32 row-index input whose contents are refreshed by the existing input callback for each chunk. The graph then owns an ordinary tensor result rather than a view carrying offsets/strides into a previous meta allocation. This deliberately trades a small gather for stable graph ownership.

Before accepting that cost, benchmark two implementations behind the same 1332 knob: (A) GET_ROWS materialization and (B) corrected meta-view bookkeeping if the stale-view root cause is proven. Do not maintain both after qualification. Promote the lower-latency implementation only if it survives graph reuse, no-MTP decode and MTP serving.

Memory/performance invariant: chunking must bound temporary mask storage by `n_kv * qsa_chunk` rather than outer ubatch. Keep outer ub1024/2048 intact for MoE/GDN; only QSA selection/masked attention is chunked. Measure peak allocator bytes and QSA kernel+copy time separately so a gather cost cannot be hidden by the VRAM win.

### New upstream graph-shape invariant: #29958

Upstream llama.cpp PR #29958 (opened 2026-10-04) fixes unexpected scheduler reallocations in qwen4exp and glm5-next k-pool models. The important mechanism is directly relevant to 1332: graph topology was branching on mutable state (`inp->cache_safe`, `n_tokens`, `n_kv`), so the full-context reserve could not predict later decode shapes. A decode re-reserve then discarded worst-case sizing and later state growth aborted under `GGML_SCHED_DEBUG_REALLOC=1`.

For qwen4exp, upstream removes the cache-safe topology branch: `new_pool_rep` is always present and fresh pooled keys are always scattered before gathering the full pool. For glm5-next, gather/dense choice is derived from context constants (`cparams.n_ubatch` and a bound on selection size) rather than per-call token/KV state. Upstream reports fixed node counts across reserve/decode and no reallocation in the reproduction matrix; it also reports the constant dense path faster than gather on M2 Ultra (16K: 259/23.9 t/s dense vs 110/21.6 gather; 2.5K: 284/24.4 vs 273/21.5).

Apply the invariant, not necessarily the exact k-pool code: **QSA chunking must not change graph node/leaf topology based on chunk offset, tail length, cache sharing, MTP state, or first-decode state after reserve.** If GET_ROWS uses a variable-length row-index tensor for tail chunks and therefore changes graph topology, do not promote it in that form. Prefer a fixed `qsa_chunk` graph shape with an explicit valid-row mask/length, or derive a small finite set of shapes entirely from context constants and reserve all of them before execution. Corrected meta views are preferable if they preserve a single constant topology without materialization cost.

This also gives a cheaper root-cause discriminator for the current post-prefill crash: enable `GGML_SCHED_DEBUG_REALLOC=1`, log graph node/leaf counts and reserve/realloc events, then compare unchunked, chunk256 full chunks, tail chunks, and first TG decode. If the crash is preceded by a shape change/re-reserve, fix topology before investigating meta-view lifetime. If topology is constant and no realloc occurs, continue the view-bookkeeping/GET_ROWS A/B.

Cross-engine mechanism check (2026-10-05): vLLM's ROCm attention backend explicitly routes unsupported/non-standard cache geometry to a Triton path rather than forcing the native kernel; its native ROCm paged-attention path is constrained by LDS. The transferable principle is narrow capability dispatch: keep the fast representation for supported graph shapes and use a safe materialized path only where chunk/view lifetime requires it. Do not import vLLM's cache format or add a second attention backend.

### Consolidation boundaries

- QFP22 owns cleanup/promotion decisions only; implementation remains in 1321/1322/1332.
- QFP17 owns QSA-prefill memory/performance evidence; QFP22 references it rather than duplicating allocator telemetry.
- FMTP03 owns MTP-ahead policy semantics; 1321/1322 implement the promoted primitive.
- QFP07 owns decode attention placement; QFP13 owns launch-gap/fusion work. QFP22 must not grow either scope.
- Upstream #29958 is a correctness/design prerequisite for 1332, not a new BigCherry patch unless current pinned upstream lacks the fix when 1332 is promoted.
- Upstream #29901 tiled lightning-indexer work is orthogonal: qualify it under QFP17 after 1332 correctness, not inside the chunk fix.

## Code Samples & Guidance

Pseudo-shape for the 1332 safe path (adapt names to current llama.cpp graph builder):

```cpp
// Prefer a context-constant shape. If tails vary, keep qsa_chunk fixed and
// pass valid_rows separately rather than rebuilding a different graph.
struct ggml_tensor * qsa_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, qsa_chunk);
ggml_set_input(qsa_rows);

// canonical mask remains the source of truth; avoid view offsets into stale meta storage
struct ggml_tensor * chunk_mask = ggml_get_rows(ctx, kq_mask, qsa_rows);
// consume chunk_mask only inside this QSA chunk; mask rows >= valid_rows
```

Input callback fills valid indices from `chunk_begin`; tail slots must be made semantically inert without changing graph topology. If GET_ROWS layout requires the opposite axis, transpose/reshape once at the canonical mask boundary rather than introducing per-layer copies.

Instrumentation required for the A/B: graph-build/reuse count; graph node/leaf count; scheduler reserve/realloc count; chunk count; bytes materialized by GET_ROWS; QSA gather time; QSA attention time; peak compute buffer; first decode graph after prefill. A candidate that merely moves the crash, unexpectedly reallocates after reserve, or adds >3% prefill wall time versus the current chunked fast path is rejected.

## Files

- `docs/planning/active/patching-qwen-flash-next/QFP22.md` - promotion/consolidation contract.
- Patch 1332 sources touching QSA mask construction/meta graph inputs - implement the selected representation there, not in a new patch.
- QFP17 - retain allocator/performance evidence and ABBA results; link results back here.

## Validation

Per patch: patch-lint + offline tests; adoption ABBA vs v6 with complete separation or a stated enabler rationale; activation evidence; greedy identity (or f32-reference agreement at near-ties); 1332 additionally no-MTP decode and MTP serving without crashes.

1332 correctness matrix before performance claims:
- no-MTP: prefill 1K/24K/80K -> at least 128 decode tokens, chunk 128/256/512;
- MTP: 24K/80K -> at least 128 accepted/verified target steps;
- outer ubatch 512/1024 and a tail chunk (`n_tokens % chunk != 0`);
- graph reuse: two sequential requests in one server process plus prompt-cache/reuse lane if supported;
- cache-sharing/batched lane where available, because #29958 shows cache sharing can flip qwen4exp runtime state;
- `GGML_SCHED_DEBUG_REALLOC=1`: zero unexpected reallocations after reserve and stable node/leaf topology for each context-constant shape;
- gfx1100 and gfx1201 target placements.

Performance gate: ub1024+chunk256 must retain the demonstrated fit advantage, save >=1 GiB peak compute memory versus unchunked ub1024 at the long-context lane, and improve prefill >=5% versus v6 ub512 on at least one representative lane without >2% decode regression. GET_ROWS/materialization overhead must be <=3% of prefill wall versus the current crash-prone chunked fast path; otherwise fix meta-view bookkeeping instead.

## Effort & Risk

Medium. The optimisation already demonstrates the desired memory effect; risk is graph ownership/lifetime and now graph-shape stability, not QSA mathematics. Highest-risk failures are silently selecting wrong mask rows on tail chunks or invalidating scheduler reserve assumptions, so greedy/f32-reference and debug-realloc checks are mandatory.

## Standards

One owner per mechanism; no parallel chunk scheduler or cache format. Correctness, graph-shape stability and graph-lifetime proof precede promotion. Record negative A/B results so a slower materialization path is not rediscovered later.

## Acceptance Criteria

- 1321/1322 promoted or explicitly parked with ABBA evidence.
- 1332 crash fixed inside 1332 and promoted or parked with ABBA evidence.
- 1332 uses one qualified chunk representation; temporary QSA mask storage scales with chunk size, not outer ubatch.
- 1332 graph topology is context-constant or fully pre-reserved: no mutable cache/token/chunk-tail branch can trigger unexpected scheduler reallocation.
- No-MTP and MTP graph-reuse matrix passes on gfx1100/gfx1201 with greedy/f32-reference agreement.
- Winners enabled through a new BIGCHERRY_FEATURES set.

## Notes

2026-10-05 optimisation scan: upstream llama.cpp #29958 is now the highest-priority 1332 mechanism to qualify because it fixes qwen4exp graph reallocation by eliminating mutable-state topology branches. Test its invariant before spending more time on GET_ROWS/view lifetime. Other open work (#29953 MMQ allocation-width consistency, #29948 MMQ+GLU fusion, #29927 AMD `amdgcn_perm`, #29901 tiled lightning indexer) remains orthogonal and should not be folded into 1332.

2026-10-05 1332 rework results (qfp17-chunk-prof, 31.8K-token prefill under rocprof, 240K f16, flashnext): ub512 c0 936 t/s; ub512 c256 923 (-1.4%, was -7% before the compact causal filter); ub1024 c256 1052 (+12%); ub1024 c512 1097 (+17%, now fits). Kernel time summed over GPUs: matmul 27.7 s (ub512) -> 19.4 s (ub1024, -30%: the actual gain); flash attention 5.5 s unchunked, 7.3 s at c256, 5.5 s at c512 (chunk cost = FA tile/launch efficiency, recovered at c512); mask build ~0.9 s, concat ~0.25 s (negligible); RCCL AllReduce 22-23 s (~30% of all kernel time - prefill AllReduce work deferred per owner); k_get_rows_float 1.0 -> 4.9 s from the compact causal gather (single-element rows) - follow-up: cheaper gather (e.g. only tail cells, pools already carry visibility via -inf scores; verify multi-sequence) or a fused gather. MTP 24K: speed unchanged (75.1 vs 75.0/76.0), text = f32-side near-tie. No-MTP decode still crashed on that build: GGML_SCHED_DEBUG_REALLOC=1 showed 'unexpected graph reallocation' (last 49-token ubatch took the dense path -> topology change after reserve, #29958 class) -> fixed in 9de9f46f (K = ceil(n_ubatch/chunk) fixed chunks for every batch >= K tokens). Confirmation queued (queue-chunk-confirm: 80K fill unprofiled 512:0/1024:256/1024:512 + no-MTP decode + MTP identity).

chunk8 (2026-10-05, build b-chunk8 = deploy-v6-plus-chunk with 1007 + fixed 1332): no-MTP crash RESOLVED by patch 1007 - chunk 256 completes (39.1 t/s decode vs 38.4 chunk 0, rc=0). Text identity still OPEN: no-MTP greedy md5 differs (c0 eda3ae34 vs c256 0db7bf7a) and 24K MTP text differs (acceptance 169/258 vs 173/242; 41.6 vs 41.5/41.3 ms/step; 70.8 vs 74.3/74.7 t/s). 80K sweep: ub1024 c512 947.4/945.0 prefill, 66.2/67.4 decode vs ub512 c0 868.6/891.3, 60.5/61.2. Next: move 1007 to patch-set.upstream-fixes; rework 1332 to a materialised contiguous per-chunk mask and drop the ggml.c FA assert relaxation (RV4220 top suspect); 1332 promotion blocked until text identity vs dense holds. External-notes follow-ups tracked in QFP28.

chunk9 (2026-10-05, b-chunk9 = 1332 with a ggml_cont contiguous per-chunk mask, no ggml.c assert relaxation; commit abd5010b): REJECTED and reverted. (a) ub1024 chunk 512 no longer loads at 240K f16 - ROCm out of memory on device 2 (R9700) during reserve, because the contiguous copy sits next to the scattered mask and removes the saving the patch exists for. (b) Text still differs from dense with the contiguous mask: no-MTP chunk 256 md5 dcf5a52d vs dense eda3ae34 (strided build gave 0db7bf7a), so the strided view / relaxed assert is NOT the sole cause of the dense-vs-chunk difference (RV4220 top suspect not confirmed). Baselines unchanged: 24K MTP 41.5/41.1 ms/step, 80K ub512 870.9 t/s prefill, 60.5 decode; no-MTP chunk 256 39.2 t/s vs 37.0. Next: md5 identity cannot separate last-bit kernel-tiling differences from corruption - compare per-token logits/top-k of chunked vs dense (and vs chunk sizes) and run the QFP28 multi-request gate before any promotion decision.

Pin bump b11402 regression resolved (2026-10-05). Bisect on Brutus (builds at b11401, 0eb6d9a81 #29940, 2ca15f540 #29612, 0bb496dbd #29622, each vs b11402, same recipe; old-pin and new-pin patched source trees differ only in the 107 upstream-changed files): #29435, the MMQ commits and #29612 cleared; #29622 (mixed token/embd batches) is the cause on both models - this contradicts the GPT ranking recorded above (#29612 primary). Fix: patch 1333_mixed_batch_on_demand (mixed select branch built only for a mixed ubatch, no flag), validated and in validated-enhancements: Flash-Next 24K MTP 41.2 ms/step (old pin 41.8, unpatched b11402 43.4/44.4); 27B prefill 10K 1290 (old 1291), 32K 1245 (old 1250, unpatched 1235); mixed-batch checks pass on all devices. Open: native-llama.cpp comparison (queue-native-1333.sh, running), 0.4% 27B 32K prefill residual with another cause, upstream report, mixed batches at real model size, two stray samples on Brutus late in the session (1145.7 t/s prefill; 44.4 ms/step on the old-pin build). Outside the build, surfaced by the improved pin-bump: 1268 fails over the build selection (common/speculative.cpp), 1250 blocked by dependency, test_1314 fails on a pristine tree - none caused by the bump.

b11402 residual placed (2026-10-05, queue-27b-pairs.sh, existing bisect binaries, 4 samples a side, 27B dual-XTX prefill at 32K, greedy text and acceptance identical): old pin vs build at 0eb6d9a81 (#29940) 1250.9 / 1250.9 / 1251.9 / 1249.6 vs 1249.3 / 1250.3 / 1249.6 / 1251.3 t/s - no gap, so #29941/#29939/#29940 are cleared; old pin vs build at 2ca15f540 (#29612) 1249.9 / 1249.6 / 1249.8 / 1247.8 vs 1246.6 / 1245.6 / 1246.6 / 1245.7 t/s - complete separation, about 0.3%. Between the two points are only #29806 (CPU tinyBLAS) and #29612 (CUDA swizzling refactor: fattn-mma-f16.cuh, mma.cuh), so #29612 is the residual's cause by elimination (not isolated by its own build). No gap at 10K. Native b11402 + 1333 also reaches 1247 t/s at 32K, consistent with the residual being upstream's. Not measured: Flash-Next long-context prefill, where an FA load-path cost would be largest. Candidate mitigation (from the GPT review above): restore the pre-#29612 AMD MMA load/address path under the AMD WMMA guard. Native comparison for 1333 done: 27B prefill +1.3% at 10K, +0.8% at 32K on stock b11402 (patch README).

## Change Log

- 2026-10-04T21:08:00.115435+00:00 (created-by): Created by agent
- 2026-10-05: Deepened 1332 graph-lifetime fix, A/B qualification, consolidation boundaries and promotion gates.
- 2026-10-05: Added upstream #29958 constant-graph-shape prerequisite, debug-realloc discriminator, fixed-tail-shape guidance and cache-sharing validation lane.

## Reviews

- RV4217
- 2026-10-04T23:43:03.592762+00:00 (updated-by): Updated: section:notes
- RV4219
- RV4220

## Ledger-events

- chg_20261004_235349_long-context-flash-next-candid_9064
- 2026-10-04T23:53:59.321183+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-05T04:10:51.942507+00:00 (updated-by): Updated: section:notes
- chg_20261005_041827_fixed-a-crash-in-multi-gpu-ten_1501
- 2026-10-05T04:18:31.126073+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-05T04:38:52.451161+00:00 (updated-by): Updated: section:notes
- chg_20261005_060430_moved-the-llamacpp-pin-to-b11_6653
- 2026-10-05T06:04:34.204505+00:00 (updated-by): Updated: section:ledger-events

- chg_20261005_082539_recovered-a-slowdown-introduce_6606

## Plan Review — 2026-10-05: b11402 HIP regression triage

Scope: pin `0504396140d1c882f5f6ee34466a42db7ae90114` -> b11402 `d89651a7b205c03c4a0b13cd0646d400dc929f79`, same `bigcherry` + `deploy-v6-plus-chunk` 34/34 patch composition. Measured outputs/draft acceptance are unchanged; this review is performance-only. The leading common suspect is upstream #29612 (`2ca15f5404760548c39e7b92bd43116a09414a1a`): it changes the AMD WMMA FlashAttention MMA load/address code without changing the selector, while single-token vec/tile paths are untouched. That shape can explain both the MTP-only Flash-Next loss and the long-prefill dense loss.

### Upstream commit reachability on Brutus HIP

| Commit | HIP reachability / path | AMD behavior change | Disposition |
|---|---|---|---|
| #29941 `dd266785` MMQ expert allocation | **Yes**, quantized `MUL_MAT_ID` / MoE MMQ in `mmq.cu` on gfx1100/gfx1201/gfx1030. | Changes only `src1_q8_1` pool allocation tail from `J_max(..., ne11)` to `J_max(..., ne12)`. No memset, kernel selector, grid, launch count or kernel code change. `J_max` rounds widths <8 to 0, so for 2-5-token verify both old and new terms are 0. At widths >=8 it can add at most the compiled J tail and can shift pool pressure/addressing. Dense non-MoE is unaffected. | Low for observed Flash MTP regression unless actual target batch reaches >=8; not a dense culprit. Correctness fix must not be reverted as mitigation. |
| #29939 `2bc56357` `blocks_per_col` | **No** on HIP AMD: changed code is inside `BLACKWELL_MMA_AVAILABLE` NVFP4 quantization. | None on gfx1100/1201/1030. | Reject. |
| #29940 `0eb6d9a8` `neu_padded` scope | **Yes**, `mm_ids_helper` used by MoE `MUL_MAT_ID`. | Moves one `constexpr` into the already-selected specialized branch. Same launch, shared memory, routing and arithmetic for instantiated specializations; only compile/template scoping can differ. | Very low. |
| #29612 `2ca15f54` FA swizzle refactor | **Yes on gfx1100/gfx1201** when `BEST_FATTN_KERNEL_MMA_F16` is selected. `AMD_WMMA_AVAILABLE` is defined for RDNA3/RDNA4. **Not the gfx1030 RDNA2 drafter path**: RDNA2 does not get that macro and falls back to vec/tile here. | AMD still has `swz=false` (swizzling is Turing-only), but the commit replaces the old `load_ldmatrix<swz>` / trans helpers and row/column address calculation with compile-time-stride `load_ldmatrix_swizzled<stride>` helpers and linear offsets in `fattn-mma-f16.cuh`/`mma.cuh`. No intended selector/grid/launch-count change, but AMD template instantiation, inlining, address arithmetic and therefore ISA/VGPR/codegen can change. Vec/tile FA is untouched. No compile flags changed. | **Primary suspect for both regressions.** |
| #29435 `d89651a7` (b11402) whole-tile FA scheduling | Host launcher is compiled/reached, but the new policy is gated by `GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK`. | On AMD `prefer_whole_tiles=false`; mask scan and prior stream-K decision are semantically unchanged. Extra `async_kv_preload` argument is host/template plumbing only. | Reject unless b11401->b11402 A/B unexpectedly proves otherwise. |
| #29622 `0bb496db` mixed token+embedding batches | **Yes, host-side on every request**, including MTP hook batches. Qwen4Exp PLE input handling is also changed. | Batch allocator now scans all entries, supports token/embedding mixes, builds section-major positions and emits a `type` vector only for genuinely mixed ubatches. Homogeneous raw-token batches get extra O(n) bookkeeping but no new GPU kernel. MTP batches carrying both token+embedding can take new host/input paths. | Secondary Flash-MTP suspect; weak dense-prefill fit. |
| #29806 `a7b94df2` CPU tinyBLAS tails | Only `ggml-cpu` x86 tinyBLAS. | No HIP device behavior; only matters if a measured matmul is intentionally/fallback-routed to CPU. Current all-GPU target/expert lanes do not. | Reject. |
| #29895 `a7fb71fa` (b11401) logging/router | Server/log host code is reachable, but router subprocess changes are irrelevant to the normal single server and log formatting runs only when a log entry is emitted. | No GPU work. No new per-token logging was added. | Reject unless verbose/high-frequency logging is enabled in the benchmark. |

#29934 is Vulkan-only and unreachable in `GGML_HIP`.

### Why the regression shape points at FA MMA, not the drafter

- Flash-Next no-MTP uses a one-token target batch; AMD FA selection commonly stays vec/tile at that shape. MTP target verification uses 2-5 query tokens and can cross the AMD WMMA MMA threshold (`Q->ne[1] * gqa_ratio_eff`), while #29612 changes only the MMA implementation. This exactly permits “MTP -3%, no-MTP flat” without changing acceptance or text.
- gfx1030 is not on the #29612 AMD-WMMA path. Therefore a #29612 regression should appear on target gfx1100/gfx1201 verify kernels, not primarily on the 6900 XT drafter. If rocprof instead shows the extra ~1.1 ms on gfx1030, demote #29612 immediately.
- Dense 27B prefill has large query batches and therefore exercises MMA FA; the loss increasing from ~0.8% at 10K to ~1.4% at 32K is consistent with attention taking a larger wall-time share with context. Dense decode can remain flat if its small verify shape selects vec/tile or if MMA FA is a much smaller fraction of the step.
- #29622 gives the other plausible MTP-only mechanism, but it does not naturally explain the separated 27B prefill regression, and its homogeneous-token overhead is small. Treat it as the next boundary if FA kernel time does not move.
- #29941 looks attractive because Flash-Next is 512-expert MoE, but for the stated 2-5-token verify widths the changed `J_max` term is exactly zero on both pins. It becomes relevant only if profiling shows a real >=8-token `MUL_MAT_ID` batch or an unexpected allocator effect elsewhere.

### BigCherry interaction audit for the measured 34-patch composition

`source.bigcherry` contributes serving-core + upstream-fixes + validated-enhancements; `deploy-v6-plus-chunk` adds only 1331 + 1332. Important consequences:

- **0200/0300 are selected.** 0200 can route quantized matmuls through replay/forced candidates; 0300 uses `ggml_cuda_mmq_get_J_max` for forced-J safety. #29941 changes the upstream MMQ allocation to use the same real MoE token-width dimension (`ne12`) that 0300 already reasons about. This is a safety-alignment change for widths >=8, not a J selector/grid change; at widths 2-5 it is inert.
- **1237/1265 are selected.** They compact MoE MMQ launch geometry on gfx1100/gfx1201/gfx1030. #29941 executes earlier in the same `ggml_cuda_mul_mat_q` path and can enlarge one pool allocation before 1237's compact-grid workspace, but it does not alter 1237's block map/grid. Only investigate this interaction if MMQ/pool profiling moves and the real width is >=8.
- **1303 is selected.** It changes attention/KV placement across the meta split, so it determines which target GPU pays an FA regression; it does not touch #29612 code or change between pins.
- **1307-1313 are selected.** These optimize Q8_1/MMVQ/fused decode producers. They do not touch the #29612 FA implementation. A pin-specific interaction is unlikely unless profiling shows the regression outside FA.
- **1332 is selected but explicitly disables chunking for `n_tokens <= 8`.** It therefore cannot create the 2-5-token MTP decode regression. It can change QSA prefill FA shapes for larger batches and may amplify a #29612 MMA change there, but no Flash-Next prefill pin A/B is currently in the evidence above.
- **1295 is not in `deploy-v6-plus-chunk`**, and its default gather threshold would not fire at a ~24K cache anyway. It cannot explain this measured regression.
- **1202 and 1266 are experiment-only, not in the measured 34 patches.** Thus 1266's direct co-tenancy with `fattn-mma-f16.cuh` is not part of this regression. If 1266 is retested on b11402 it must be requalified because #29612 rewrote the surrounding load/address helpers despite clean anchors. 1202 is the separate BF16 tile path and is not relevant to these f16-KV measurements.
- **No cleanly-applied anchor in the measured selection was found to reinterpret an upstream-changed selector.** The important semantic change is upstream #29612 itself; BigCherry mainly changes placement/workload shapes that decide how often that kernel runs.

### Ranked suspects

Flash-Next 24K MTP decode:
1. **#29612 FA swizzle/load refactor — ~75% confidence.** Best fit to MTP-vs-no-MTP and target-vs-drafter architecture split.
2. **#29622 mixed-batch host refactor — ~15%.** MTP-specific host path is reachable; confirm only if GPU kernel census does not account for the +1.1 ms.
3. **#29941 MMQ allocation — <=5%.** Structurally MoE-specific but inert for widths 2-5 because `J_max=0`; raise only if actual width >=8 or pool telemetry changes.
4. **#29940 — <=2%.** Same MoE helper arithmetic/launch.
5. **#29435/#29939/#29806/#29895 — ~0-1% each** under the stated run.

27B dense prefill:
1. **#29612 — ~90% confidence.** Only material AMD GPU-kernel change in the range that directly matches large-batch FA; increasing loss with context is consistent.
2. **#29622 — ~5%.** Extra host batch preparation is reachable but unlikely to cost 0.8-1.4% of GPU-bound prefill.
3. **#29435 — ~1%.** Explicit NVIDIA-only behavioral gate; b11401/b11402 is the cheap proof.
4. Others: effectively excluded by backend/path (MoE-only, Blackwell-only, CPU-only or logging-only).

### Cheapest discriminating experiments, in order

1. **Profile before rebuilding.** rocprofv3 kernel census old pin vs b11402 on (a) ~100 Flash MTP target steps at 24K and (b) 32K dense prefill. Aggregate count/total/mean by GPU and kernel family: `flash_attn_ext_f16*` MMA vs tile/vec, `mul_mat_q*`, `quantize*_mmq_q8_1*`, `mm_ids_helper`, and launch gaps. Confirmation for #29612: same FA-MMA launch counts/grid family and unchanged vec/tile/MMQ times, but target gfx1100/gfx1201 FA-MMA total/mean grows by approximately the missing wall time; gfx1030 remains flat. If kernel totals are flat but host gaps grow, move to #29622 and existing 1319/1325 host-submit diagnostics.
2. **b11401 `a7fb71fab83b474a0892b9a05aaa3a8ddca2729b` vs b11402 `d89651a7...`.** This isolates #29435. Expected: both are equally slow on AMD. If b11401 recovers, #29435 has an unexpected AMD/codegen effect despite the NVIDIA guard and should be reduced immediately.
3. **`a7b94df2c616bc1f62a73b964b4a71cb0dcc488e` vs `2ca15f5404760548c39e7b92bd43116a09414a1a`.** Exact before/after pair for #29612. Run one dense 32K prefill pair plus Flash MTP/no-MTP. Confirmation: both regressions first appear at `2ca15f54`; no-MTP stays flat.
4. **`2ca15f54` vs `0bb496dbd3af0add77ff82c406a915b41e839d56`.** Exact #29622 boundary. Only needed if #29612 boundary is clean or host-gap profiling moves. Confirmation: MTP wall/host submit changes at `0bb496db` while per-kernel GPU time is stable.
5. **`16c163d561b976d8375a17db5660105abe47cda1` vs `dd266785c2595775001c1c714bd9d92b3ef34cde`.** Exact #29941 boundary; run only if a real >=8-token MoE MMQ shape is observed or pool/MMQ timing moves. For the stated 2-5 width it should be identical.
6. **`2e7c58c5477478c8cf6e199cfaa5dcd5a4319c81` vs `0eb6d9a8137ff13deb1b3a755b01b1e5b7f89229`.** Exact #29940 boundary; lowest-priority MoE check.

A single revert of #29435 on b11402 is cheap. Prefer the exact `a7b94df2`/`2ca15f54` boundary over reverting #29612 on top of b11402 because #29435 later edits the same FA file and makes the revert conflict-prone/less clean. Likewise, do not use a #29941 revert as a production mitigation: it restores a known OOB fault.

### Mitigation / upstream disposition

If #29612 is confirmed, add a narrow BigCherry upstream-fix patch that restores the pre-#29612 **unswizzled AMD** MMA load/address implementation under `AMD_WMMA_AVAILABLE` / `AMD_MFMA_AVAILABLE`, while leaving the new NVIDIA/Turing swizzle refactor intact. Do not change FA selection policy first: the cleanest fix is to recover the old AMD machine-code path for the same selected kernel and prove identical output plus recovered kernel time. Re-run on both gfx1100 and gfx1201; gfx1030 is the negative control. This should be reported upstream with the exact `a7b94df2` vs `2ca15f54` A/B, rocprof per-kernel delta, kernel launch counts and unchanged output.

If #29622 is confirmed instead, optimize the homogeneous raw-token fast path in `llama_batch_allocr`/MTP hook handling rather than reverting mixed-batch support; report upstream if stock llama.cpp reproduces the host-submit regression. If #29941 is unexpectedly implicated, retain its corrected allocation and optimize/reuse the scratch/pool behavior or reconcile 0300/1237 workspace sizing; never reintroduce the under-allocation.

Promotion/pin decision: keep b11402 correctness only if required by an upstream fix; otherwise the performance gate remains open until the #29612 boundary/profile test is resolved. No BigCherry patch/code change is authorized by this review alone.
- 2026-10-05T08:25:42.858557+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-05T08:25:48.546505+00:00 (updated-by): Updated: section:notes
- 2026-10-05T08:53:19.695307+00:00 (updated-by): Updated: section:notes
