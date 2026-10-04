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

## Change Log

- 2026-10-04T21:08:00.115435+00:00 (created-by): Created by agent
- 2026-10-05: Deepened 1332 graph-lifetime fix, A/B qualification, consolidation boundaries and promotion gates.
- 2026-10-05: Added upstream #29958 constant-graph-shape prerequisite, debug-realloc discriminator, fixed-tail-shape guidance and cache-sharing validation lane.

## Reviews

- RV4217
