---
id: QFP25
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-05T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Native HIP selected-index QSA prefill attention (eliminate dense mask and full-KV scan)

## Description

QFP17 already identifies `QSA-SPARSE-PP` as the long-term prefill path, and QFP04 notes the same interface seam from the selected-cell decode work: pass selected QSA cells into Flash Attention instead of materializing a dense `[n_kv,n_tokens]` mask. QFP25 is the concrete implementation owner for that kernel/interface. QFP17 remains the prefill attribution/acceptance umbrella; QFP04 remains owner of gather-based small-token/decode QSA and cache-type/dequant details.

The current HIP path pays two avoidable long-context costs: dense QSA mask construction/retention and attention traversal over the full KV range while a mask suppresses almost all positions. Production QSA selects roughly a small fixed set of cells plus the causal tail, so the algorithm should consume compact selected ranges directly and perform online-softmax attention only over those positions.

This item supersedes neither 1332 nor its promotion work. 1332 is a bounded-memory bridge that chunks the dense path; QFP25 removes the dense representation entirely. If QFP25 succeeds, 1332 should remain a fallback/compatibility path rather than a prerequisite.

## Steps

1. Freeze the semantic reference. For representative 24K/80K/200K states, dump per-query QSA selected cell IDs after dead-slot remap plus the exact dense mask rows consumed by current HIP FA. Prove the compact representation expands to the same allowed KV positions, including causal tail, dump/dead slots, padded selection entries and duplicates.
2. Define one compact input contract reused by decode/prefill where possible. Preferred first form: I32 selected cell/token indices plus metadata `{n_selected, cell_size/range, tail_begin, n_kv}` per query. Do not add a second top-k/indexer output alongside `bc_qsa_idx` unless layout conversion is measurably cheaper than teaching the kernel that layout.
3. Add an opt-in `flash_attn_ext`/HIP backend capability for selected-index sparse attention, or a narrowly named QSA op if the existing FA API cannot express indirect KV reads without destabilizing other backends. Unsupported backends must reject/fall back to dense QSA; never reinterpret a dense-mask flag as sparse indices.
4. Implement f16-KV correctness kernel first for Qwen4Exp head dimensions/shapes at the production pin. One workgroup handles a query/head tile, iterates selected KV tiles, and maintains online softmax `{m,l,acc}`. No `[n_kv,T]` mask allocation and no full-context K/V traversal are permitted on the sparse path.
5. Tile selected positions for coalescing. Start with selected KV blocks of 16 or 32 tokens and query tile 1/2/4/8 depending on register pressure. Stage compact indices/range descriptors in LDS; load K/V directly from cache. Use wave reductions already used by the HIP FA implementation rather than inventing a separate reduction primitive.
6. Preserve QSA cell semantics. If an index denotes a cell/range, expand that cell in-kernel in the same token order as the dense mask. Apply causal clipping for the current query position and include the mandatory tail. Deduplicate only if reference expansion proves the current dense path treats duplicate selections idempotently; otherwise preserve exact multiplicity/order semantics.
7. Add a density/shape fallback. Sparse indirect access loses when selection density approaches full context or for very short contexts. Dispatch only when `selected_tokens / n_kv` and `n_tokens` meet a measured threshold, configured through runtime profile rather than model-specific magic constants in patch code (QFP23).
8. After f16 qualification, add q8_0 KV dequant-on-load only if production requires quantized KV. Reuse QFP04/1296 cache-type work; do not materialize a gathered f16 cache. Quant dequant must happen per selected tile in registers/LDS.
9. Fuse/remap preparation only after the attention kernel dominates. 1327/QFP13 leaves several device remap operations; if their cost is material in prefill, add a single compact-index preparation kernel that emits exactly QFP25's input. Do not optimize launch count before removing the dense scan.
10. Benchmark standalone sparse attention and end-to-end QFP17 lanes; then decide whether dense chunking 1332 remains useful as a fallback or can be parked for production.

## Detailed Solution & Technical Design

### Compact contract

Keep selection semantics explicit. A practical first contract is:

```cpp
struct bc_qsa_sparse_desc {
    const int32_t * selected;   // [n_query, max_selected] cell or token ids
    const int32_t * counts;     // [n_query] if variable
    int32_t cell_tokens;        // 1 when selected already contains token ids
    int32_t tail_begin;         // base tail start, clipped per query
    int32_t n_kv;
    int32_t max_selected;
};
```

Do not literally pass a host struct to the kernel if GGML op params/tensors are a better fit. The invariant is that the kernel can enumerate the same allowed KV tokens without inspecting an `n_kv`-wide mask.

If QSA uses selected cells rather than individual tokens, keeping cell IDs is preferable: it reduces index traffic and lets one workgroup read contiguous K/V ranges. Expand `cell_id * cell_tokens + lane` in-kernel and clip at `n_kv`/causal position.

### Online softmax

For each query/head, process sparse K/V tiles in deterministic enumeration order:

```cpp
float m = -INFINITY;
float l = 0.0f;
vec<float,D> acc = 0;

for (selected tile s) {
    scores = dot(q, K[s]) * scale + any_non_qsa_bias;
    float m2 = max(m, max(scores));
    float alpha = exp(m - m2);
    float p = exp(scores - m2);
    acc = acc * alpha + sum_j(p[j] * V[s+j]);
    l   = l   * alpha + sum_j(p[j]);
    m = m2;
}
out = acc / l;
```

Use the existing FA math helpers/exp behavior so the sparse path is numerically comparable to the dense reference. If the current kernel uses base-2 exponent scaling or packed vector reductions, preserve those details rather than translating this pseudocode literally.

### RDNA workgroup shape

Start from the existing HIP/CUDA FA implementation's head-dimension templates and wave primitives. Candidate mapping for f16, D=128/256:

- one block per `(query_tile, head)`;
- 4-8 waves/block depending on D and query tile;
- KV tile 16/32 tokens;
- indices/range base in LDS, K/V streamed with vector loads;
- Q resident in registers/LDS across all selected tiles;
- partial score/max/sum reduced within wave or across waves using existing shared-memory reduction pattern.

Do not force WMMA/MFMA on gfx1100/gfx1201 unless the dot shape and resource profile prove a win. Indirect sparse loads can make occupancy/memory latency more important than theoretical matrix throughput.

### Ordering and duplicates

The dense mask defines a set, while a naive selected-index loop can accidentally treat duplicates as repeated probability mass. Before kernel work, normalize the compact source to a set representation or prove upstream TOP_K/remap cannot produce duplicate live token ranges. If cell ranges overlap the causal tail, avoid double-counting: either mark tail cells out of the selected list or perform a deterministic duplicate filter on compact indices. The reference expansion test is mandatory.

### Memory invariant

Sparse-path graph allocation must not contain any f16/f32 tensor whose size is `O(n_kv * n_query)` solely for QSA masking. Small causal metadata or selected-index tensors are `O(k * n_query)`. Use 1329/1331 allocation tracing to enforce this as a testable invariant.

### Backend/API boundary

Prefer extending the existing FA extension with an explicit sparse selection input/capability enum if its backend dispatcher already owns K/V cache types and head-dim specialization. If that makes non-HIP backends ambiguous, introduce a QSA-specific op with `supports_op` checks. CPU reference may expand compact selection and compute a straightforward f32 attention for correctness; it need not be fast.

## Code Samples & Guidance

Expected source ownership at the current pin:

- `src/models/qwen4exp.cpp`: QSA graph construction; emit compact selection to sparse attention and stop constructing dense mask on the qualified path.
- `ggml/src/ggml-cuda/fattn*.cu/.cuh` or current Flash Attention extension implementation: sparse HIP kernel/dispatch; inspect exact pin before patching.
- QSA/indexer/remap backend code and `patches/1295_qsa_gather_decode` / `1327_qsa_host_remap`: reuse compact selection semantics; no duplicate selection pipeline.
- `tests/test-backend-ops.cpp` or dedicated QSA reference fixture: direct sparse-vs-dense/CPU correctness.
- 1329/1331 allocation traces and QFP17 lab scripts for memory/E2E evidence.

Suggested dispatch pseudocode:

```cpp
const bool sparse_ok = qsa_sparse_enabled &&
                       kv_type_supported &&
                       head_dim_supported &&
                       n_query >= sparse_min_q &&
                       selected_token_count * sparse_density_den < n_kv * sparse_density_num;
if (sparse_ok) {
    ggml_flash_attn_qsa_sparse(... selected ...);
} else {
    build_dense_qsa_mask_and_fattn(...);
}
```

Threshold values belong in QFP23 runtime profiles after tuning; intrinsic capability checks remain code.

## Files

Qwen4Exp QSA graph source; CUDA/HIP Flash Attention implementation and dispatcher; QFP04/1295/1296 selection/cache-type code where reusable; 1327 compact remap preparation if needed; direct backend tests; QFP17 profiling/acceptance scripts.

## Validation

Correctness matrix: CPU f32 compact expansion vs existing dense mask for context 1K/24K/80K/200K; query batches 1/4/8/64/256/512/1024; tail boundaries; top-k tie/dead-slot cases; duplicate/overlap cases; non-multiple selection/tail sizes. Compare selected token set exactly before comparing floating outputs.

Direct kernel: sparse vs dense FA max/mean error and top logits through one QSA layer; gfx1100 and gfx1201; f16 first, q8_0 only when implemented. Record VGPR/SGPR/LDS/spills, occupancy, selected-token bandwidth and kernel wall.

End-to-end: QFP17 ABBA at 10K/80K/200K, ub512 and larger ubatch if memory permits. Record prefill t/s, TTFT, peak compute buffer, QSA/indexer/FA wall and decode control. Greedy/reference should use the f32-side near-tie policy already established in QFP17 rather than demanding byte identity to an MTP batch-numerics path.

## Effort & Risk

L/high. This changes attention representation and numerical traversal order. Main risks are duplicate/overlap semantics, causal-tail indexing, poor indirect-load coalescing, register pressure at D=256, and an API extension that leaks Qwen-specific semantics into generic FA. Keep the first kernel narrow and fall back aggressively.

## Standards

QFP17 is acceptance owner; QFP25 is implementation owner. One compact selection representation; CPU/reference oracle first; no dense QSA mask on the sparse path; explicit backend capability/fallback; profile-derived dispatch thresholds in QFP23; no correctness tolerance widening to hide selection bugs.

## Acceptance Criteria

- Compact selection expands to exactly the same allowed KV token set as the reference dense mask across the correctness matrix.
- Qualified sparse graph allocates no QSA-specific `O(n_kv*n_query)` dense mask.
- Direct sparse attention is >=2x faster than dense HIP QSA attention at an 80K+ representative shape or reduces QSA attention+mask critical wall >=50%.
- End-to-end long-context prefill improves >=8% on at least one production-representative 80K+ lane, with no decode regression >1% and accepted f32-reference/logit contract.
- Unsupported shapes/types/backends fall back cleanly to dense QSA/1332 without graph-lifetime failures.

## Notes

This mechanism was already named in QFP17 and foreshadowed in QFP04, so QFP25 does not create a new optimization direction; it supplies the missing implementation contract. 1332 remains useful evidence that dense QSA memory is the ubatch limiter, but its modest E2E gain after making ub1024 fit strengthens the case for removing the dense scan rather than only chunking it.

## Change Log

- 2026-10-05T00:00:00+00:00 (created-by): Created by agent as implementation owner for QFP17 QSA-SPARSE-PP / QFP04 selected-index FA seam.
