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
5. Validation: no-MTP decode + MTP serving at 24K/80K; ub1024+chunk256 vs v6 ub512 prefill/decode ABBA at 80K fill and a short-prompt prefill (where larger ub should help most); greedy checked against the f32 reference at near-ties (QFP17).
6. Decide the production setting (ub1024 + BIGCHERRY_QSA_CHUNK=256 in a feature set) only if the ABBA shows a win.

7. Add winners to the flashnext feature set (new row in 0910, e.g. flashnext-v7) and promote via validated-enhancements.

## Detailed Solution & Technical Design

### 1332: make chunking a graph-shape boundary, not a view-lifetime trick

The crash signature says the optimisation is sound (ub1024 fits) but the representation is unsafe across meta-graph rebuild/reuse. Do not add another QSA chunk scheduler. Keep QFP17/1332 as the sole owner and change only how each chunk is represented in the graph.

Preferred experiment: materialize chunk row selection with `GGML_OP_GET_ROWS` from the canonical mask tensor. Add a tiny I32 row-index input whose contents are refreshed by the existing input callback for each chunk. The graph then owns an ordinary tensor result rather than a view carrying offsets/strides into a previous meta allocation. This deliberately trades a small gather for stable graph ownership.

Before accepting that cost, benchmark two implementations behind the same 1332 knob: (A) GET_ROWS materialization and (B) corrected meta-view bookkeeping if the stale-view root cause is proven. Do not maintain both after qualification. Promote the lower-latency implementation only if it survives graph reuse, no-MTP decode and MTP serving.

Memory/performance invariant: chunking must bound temporary mask storage by `n_kv * qsa_chunk` rather than outer ubatch. Keep outer ub1024/2048 intact for MoE/GDN; only QSA selection/masked attention is chunked. Measure peak allocator bytes and QSA kernel+copy time separately so a gather cost cannot be hidden by the VRAM win.

Cross-engine mechanism check (2026-10-05): vLLM's ROCm attention backend explicitly routes unsupported/non-standard cache geometry to a Triton path rather than forcing the native kernel; its native ROCm paged-attention path is constrained by LDS. The transferable principle is narrow capability dispatch: keep the fast representation for supported graph shapes and use a safe materialized path only where chunk/view lifetime requires it. Do not import vLLM's cache format or add a second attention backend.

### Consolidation boundaries

- QFP22 owns cleanup/promotion decisions only; implementation remains in 1321/1322/1332.
- QFP17 owns QSA-prefill memory/performance evidence; QFP22 references it rather than duplicating allocator telemetry.
- FMTP03 owns MTP-ahead policy semantics; 1321/1322 implement the promoted primitive.
- QFP07 owns decode attention placement; QFP13 owns launch-gap/fusion work. QFP22 must not grow either scope.
- Upstream #29901 tiled lightning-indexer work is orthogonal: qualify it under QFP17 after 1332 correctness, not inside the chunk fix.

## Code Samples & Guidance

Pseudo-shape for the 1332 safe path (adapt names to current llama.cpp graph builder):

```cpp
// graph input, length = qsa_chunk
struct ggml_tensor * qsa_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, chunk_n);
ggml_set_input(qsa_rows);

// canonical mask remains the source of truth; avoid view offsets into stale meta storage
struct ggml_tensor * chunk_mask = ggml_get_rows(ctx, kq_mask, qsa_rows);
// consume chunk_mask only inside this QSA chunk
```

Input callback fills `qsa_rows[i] = chunk_begin + i`. Tail chunks use their actual length; do not pad indices beyond the valid query range. If GET_ROWS layout requires the opposite axis, transpose/reshape once at the canonical mask boundary rather than introducing per-layer copies.

Instrumentation required for the A/B: graph-build/reuse count; chunk count; bytes materialized by GET_ROWS; QSA gather time; QSA attention time; peak compute buffer; first decode graph after prefill. A candidate that merely moves the crash or adds >3% prefill wall time versus the current chunked fast path is rejected.

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
- gfx1100 and gfx1201 target placements.

Performance gate: ub1024+chunk256 must retain the demonstrated fit advantage, save >=1 GiB peak compute memory versus unchunked ub1024 at the long-context lane, and improve prefill >=5% versus v6 ub512 on at least one representative lane without >2% decode regression. GET_ROWS/materialization overhead must be <=3% of prefill wall versus the current crash-prone chunked fast path; otherwise fix meta-view bookkeeping instead.

## Effort & Risk

Medium. The optimisation already demonstrates the desired memory effect; risk is graph ownership/lifetime, not QSA mathematics. Highest-risk failure is silently selecting wrong mask rows on tail chunks, so greedy/f32-reference checks are mandatory.

## Standards

One owner per mechanism; no parallel chunk scheduler or cache format. Correctness and graph-lifetime proof precede promotion. Record negative A/B results so a slower materialization path is not rediscovered later.

## Acceptance Criteria

- 1321/1322 promoted or explicitly parked with ABBA evidence.
- 1332 crash fixed inside 1332 and promoted or parked with ABBA evidence.
- 1332 uses one qualified chunk representation; temporary QSA mask storage scales with chunk size, not outer ubatch.
- No-MTP and MTP graph-reuse matrix passes on gfx1100/gfx1201 with greedy/f32-reference agreement.
- Winners enabled through a new BIGCHERRY_FEATURES set.

## Notes

2026-10-05 optimisation scan: upstream llama.cpp open work remains concentrated around #29953 MMQ allocation-width consistency, #29948 MMQ+GLU fusion, #29927 AMD `amdgcn_perm`, and #29901 tiled lightning indexer. None should be folded into 1332. vLLM ROCm's capability-dispatch approach is useful design evidence for keeping the safe fallback narrow.

## Change Log

- 2026-10-04T21:08:00.115435+00:00 (created-by): Created by agent
- 2026-10-05: Deepened 1332 graph-lifetime fix, A/B qualification, consolidation boundaries and promotion gates.
