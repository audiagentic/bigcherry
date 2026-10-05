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

QFP17 already names `QSA-SPARSE-PP`; QFP04 already identifies selected-index attention as the correct long-term seam. QFP25 is the concrete implementation owner.

Pinned `0504396` source confirms the current Qwen4Exp path constructs a dense per-query mask: `mask_all = [-inf]`, remaps dead selected slots into dump rows, `ggml_set_rows(... sel_idx ...)` writes zeros for selected cells, then takes an `n_kv` view and adds the ordinary `kq_mask` before dense attention. The sparse path must consume the **same remapped compact selection** before `mask_all` is materialized and reproduce the final allowed KV set exactly.

Current dense graph fragment at this pin:

```cpp
const int64_t n_kv = inp_kpool->n_kv;

ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, kq_mask->type, n_kv + n_sel, 1, 1, 1);
mask_all = ggml_fill(ctx0, mask_all, -INFINITY);
mask_all = ggml_repeat_4d(ctx0, mask_all, n_kv + n_sel, n_tokens, 1, 1);
mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + n_sel, n_tokens);
...
sel_idx = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);

ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros,
        ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));
...
sel = ggml_add(ctx0, sel, kq_mask);
cb(sel, "indexer_sel", il);
```

QFP25 bypasses `mask_all/zeros/set_rows/add` on the qualified path.

## Steps

1. Refactor QSA selection so the remapped I32 `sel_idx` is available as a first-class graph value before dense mask creation.
2. Add a reference expander that converts compact indices + causal/tail metadata into the exact current dense mask; use it in CPU/offline tests.
3. Add a distinct sparse-QSA attention op/capability rather than overloading dense-mask semantics in `GGML_OP_FLASH_ATTN_EXT` sources.
4. Implement f16 K/V HIP correctness kernel first for production D/GQA shapes.
5. Enumerate selected cells + mandatory causal tail exactly once; prevent duplicate probability mass where selection overlaps tail or another cell.
6. Dispatch sparse only for measured prefill density/shape thresholds; fallback to the current dense graph for everything else.
7. Add q8_0 dequant-on-load only after f16 is correct/profitable, reusing QFP04/1296 traits.
8. Only then fuse index remap/preparation if it remains material.
9. Enforce memory invariant: no QSA-only `O(n_kv*n_query)` tensor on sparse path.
10. Validate standalone and QFP17 end-to-end lanes.

## Detailed Solution & Technical Design

### 1. Refactor Qwen4Exp selection result, do not recompute TOP_K/remap

Introduce a small graph-local result object in `src/models/qwen4exp.cpp`:

```cpp
struct qwen4exp_qsa_selection {
    ggml_tensor * idx = nullptr;       // I32 [n_sel, n_tokens], after dead-slot remap
    ggml_tensor * dense = nullptr;     // optional fallback mask [n_kv, ...]
    int64_t n_sel = 0;
    int64_t n_kv  = 0;
};
```

Split the current mask builder at the existing `sel_idx = ggml_cast(... GGML_TYPE_I32)` point:

```cpp
qwen4exp_qsa_selection llama_model_qwen4exp::graph::build_qsa_selection(
        llm_graph_input_attn_kv * inp,
        llm_graph_input_kpool * inp_kpool,
        ggml_tensor * kq_mask,
        /* existing score/top-k args */,
        int il,
        bool need_dense) {
    qwen4exp_qsa_selection out;

    // Existing score -> TOP_K -> live/dead-slot remap code unchanged.
    ggml_tensor * sel_idx = /* existing remapped I32 tensor */;
    cb(sel_idx, "indexer_sel_idx", il);

    out.idx   = sel_idx;
    out.n_sel = inp_kpool->n_sel;
    out.n_kv  = inp_kpool->n_kv;

    if (!need_dense) {
        return out;
    }

    // Move the existing mask_all/zeros/set_rows/view/add code into this block unchanged.
    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx0, kq_mask->type,
                                                out.n_kv + out.n_sel, 1, 1, 1);
    mask_all = ggml_fill(ctx0, mask_all, -INFINITY);
    mask_all = ggml_repeat_4d(ctx0, mask_all, out.n_kv + out.n_sel, n_tokens, 1, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, out.n_kv + out.n_sel, n_tokens);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, kq_mask->type, out.n_sel, 1, 1, 1);
    zeros = ggml_fill(ctx0, zeros, 0.0f);
    zeros = ggml_repeat_4d(ctx0, zeros, out.n_sel, n_tokens, 1, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, out.n_sel, n_tokens);

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros,
        ggml_reshape_3d(ctx0, sel_idx, out.n_sel, n_tokens, 1));

    const size_t row = sel->nb[2];
    sel = ggml_view_4d(ctx0, sel, out.n_kv,
        kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3],
        row, row*kq_mask->ne[1], row*kq_mask->ne[1]*kq_mask->ne[2], 0);
    out.dense = ggml_add(ctx0, sel, kq_mask);
    cb(out.dense, "indexer_sel", il);
    return out;
}
```

Do not literally duplicate the current remap code. The implementation patch should mechanically move existing lines into the helper so sparse/dense share one `sel_idx` producer.

### 2. New op instead of abusing dense `FLASH_ATTN_EXT`

First implementation should use a new explicit op, e.g. `GGML_OP_FLASH_ATTN_QSA`, because existing FA source slots/op_params already carry generic mask/sinks/precision semantics and silently interpreting an I32 tensor as a dense mask is unsafe.

Header constructor:

```cpp
GGML_API struct ggml_tensor * ggml_flash_attn_qsa(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * sel_idx,   // I32 [n_sel, n_query]
        struct ggml_tensor  * q_pos,     // I32 [n_query], absolute cache/query positions
        float                 scale,
        int32_t               n_kv,
        int32_t               n_sel,
        int32_t               cell_tokens,
        int32_t               tail_tokens);
```

Constructor in `ggml/src/ggml.c`:

```cpp
struct ggml_tensor * ggml_flash_attn_qsa(
        struct ggml_context * ctx,
        struct ggml_tensor * q,
        struct ggml_tensor * k,
        struct ggml_tensor * v,
        struct ggml_tensor * sel_idx,
        struct ggml_tensor * q_pos,
        float scale,
        int32_t n_kv,
        int32_t n_sel,
        int32_t cell_tokens,
        int32_t tail_tokens) {
    GGML_ASSERT(sel_idx->type == GGML_TYPE_I32);
    GGML_ASSERT(q_pos->type   == GGML_TYPE_I32);

    ggml_tensor * dst = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F32, v->ne[0], q->ne[2], q->ne[1], q->ne[3]);

    dst->op = GGML_OP_FLASH_ATTN_QSA;
    dst->src[0] = q;
    dst->src[1] = k;
    dst->src[2] = v;
    dst->src[3] = sel_idx;
    dst->src[4] = q_pos;

    ggml_set_op_params_f32(dst, 0, scale);
    ggml_set_op_params_i32(dst, 1, n_kv);
    ggml_set_op_params_i32(dst, 2, n_sel);
    ggml_set_op_params_i32(dst, 3, cell_tokens);
    ggml_set_op_params_i32(dst, 4, tail_tokens);
    return dst;
}
```

Verify the exact op-param helper indexing convention at implementation time; if mixed f32/i32 helpers index byte offsets differently at this pin, use a packed POD copied with `memcpy` into `op_params`.

### 3. Graph dispatch in Qwen4Exp

At the current `if (sel) build_attn_qsa(...)` branch, sparse selection should be explicit:

```cpp
const bool sparse_qsa = params.cparams.bigcherry_qsa_sparse &&
                        n_tokens >= params.cparams.bigcherry_qsa_sparse_min_q &&
                        qsa_sel.n_sel > 0;

if (sparse_qsa) {
    ggml_tensor * q_pos = inp->get_pos(); // use the real existing position input/helper at pin
    cur = ggml_flash_attn_qsa(
        ctx0, Qcur, Kcur, Vcur,
        qsa_sel.idx, q_pos,
        kq_scale,
        (int32_t) qsa_sel.n_kv,
        (int32_t) qsa_sel.n_sel,
        (int32_t) hparams.indexer_kpool,
        (int32_t) /* current mandatory QSA tail length */);
    cb(cur, "attn_qsa_sparse", il);
} else {
    GGML_ASSERT(qsa_sel.dense != nullptr);
    cur = build_attn_qsa(inp, Qcur, Kcur, Vcur,
                         qsa_sel.dense, qsa_sel.n_sel, kq_scale, il);
}
```

`inp->get_pos()` is illustrative: wire the actual graph position tensor already used for RoPE/KQ masking. Do not add a host position array if a graph input already exists.

### 4. Backend supports/dispatch

CUDA/HIP backend should reject unsupported shapes/types explicitly:

```cpp
case GGML_OP_FLASH_ATTN_QSA: {
    const ggml_tensor * q   = op->src[0];
    const ggml_tensor * k   = op->src[1];
    const ggml_tensor * v   = op->src[2];
    const ggml_tensor * idx = op->src[3];

    if (idx == nullptr || idx->type != GGML_TYPE_I32) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) {
        return false;
    }
    if (k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        return false; // first implementation only
    }
    const int64_t D = q->ne[0];
    return D == 128 || D == 256;
}
```

No CPU silent fallback through the same op unless a real CPU reference implementation is added. Model graph should choose dense path when target backend lacks support; test `supports_op` during qualification.

### 5. Compact enumeration semantics

The current dense mask is set-like: selected live rows become zero; unselected rows stay `-inf`; then ordinary `kq_mask` is added. A sparse loop must not count duplicates twice.

Normalize one query's selected cells to a deterministic set before attention. First correctness implementation may use a small local sorted list because `n_sel` is small compared with `n_kv`:

```cpp
__device__ int bc_unique_sorted_cells(
        const int32_t * src, int n,
        int32_t * dst, int32_t dump_begin) {
    int m = 0;
    for (int i = 0; i < n; ++i) {
        const int32_t x = src[i];
        if (x < 0 || x >= dump_begin) {
            continue; // remapped dead/dump slots are not real KV
        }
        int p = m;
        while (p > 0 && dst[p - 1] > x) {
            dst[p] = dst[p - 1];
            --p;
        }
        if (p > 0 && dst[p - 1] == x) {
            continue;
        }
        dst[p] = x;
        ++m;
    }
    return m;
}
```

This O(k²) code is for the CPU/reference or tiny-k correctness kernel only; production should use the deterministic order/top-k guarantees or a warp/LDS dedupe once semantics are proven. Do not ship quadratic per-query sorting blindly.

### 6. HIP kernel skeleton: f16 KV first

Add a dedicated file or sibling in current FA sources, e.g. `fattn-qsa.cu/.cuh`. Use existing CUDA portability macros and wave reductions. Skeleton:

```cpp
template <int D, int KV_TILE>
static __global__ void flash_attn_qsa_f16(
        const float * __restrict__ q,      // adapt type/layout to actual FA input
        const half  * __restrict__ k,
        const half  * __restrict__ v,
        const int32_t * __restrict__ sel,
        const int32_t * __restrict__ q_pos,
        float * __restrict__ dst,
        int n_kv,
        int n_sel,
        int cell_tokens,
        int tail_tokens,
        int n_head_q,
        int n_head_kv,
        int n_query,
        float scale) {
    const int iq = blockIdx.x;
    const int hq = blockIdx.y;
    if (iq >= n_query || hq >= n_head_q) {
        return;
    }

    const int hk = hq % n_head_kv; // verify GQA head mapping against current FA code
    const int pos_q = q_pos[iq];

    float m = -INFINITY;
    float l = 0.0f;
    float acc[D / WARP_SIZE + 1]; // illustrative lane-striped accumulator
    #pragma unroll
    for (int i = 0; i < (D + WARP_SIZE - 1) / WARP_SIZE; ++i) acc[i] = 0.0f;

    auto visit_token = [&](int kv) {
        if (kv < 0 || kv >= n_kv || kv > pos_q) return;

        float dot = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            dot += q[bc_q_index(iq, hq, d)] * __half2float(k[bc_k_index(kv, hk, d)]);
        }
        dot = warp_reduce_sum(dot); // use exact existing helper / block reduction as needed
        dot *= scale;

        const float m2 = fmaxf(m, dot);
        const float alpha = expf(m - m2);
        const float p = expf(dot - m2);

        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            const int a = d / blockDim.x;
            acc[a] = acc[a] * alpha + p * __half2float(v[bc_v_index(kv, hk, d)]);
        }
        l = l * alpha + p;
        m = m2;
    };

    // Selected cells.
    for (int s = 0; s < n_sel; ++s) {
        const int cell = sel[iq * n_sel + s];
        const int base = cell * cell_tokens;
        #pragma unroll
        for (int j = 0; j < cell_tokens; ++j) {
            visit_token(base + j);
        }
    }

    // Mandatory tail; skip tokens already covered by selected cells in final implementation.
    const int tail0 = max(0, pos_q + 1 - tail_tokens);
    for (int kv = tail0; kv <= pos_q && kv < n_kv; ++kv) {
        if (!bc_token_already_selected(kv, sel + iq*n_sel, n_sel, cell_tokens)) {
            visit_token(kv);
        }
    }

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const int a = d / blockDim.x;
        dst[bc_o_index(iq, hq, d)] = acc[a] / l;
    }
}
```

The lambda/index helpers/accumulator above are intentionally source-shaped, not drop-in final code: CUDA device lambdas and lane-striped accumulation may be unsuitable under the project's compile flags. Implement the same state machine using the existing FA kernel's vector/register layout and `warp_reduce_*` helpers. The plan requirement is: one online-softmax state across **unique** selected+tail tokens, direct K/V reads, no dense scan.

### 7. Production workgroup shape

After correctness, replace one-token-at-a-time enumeration with selected KV tiles:

```cpp
for (int s0 = 0; s0 < n_unique_tokens; s0 += KV_TILE) {
    // LDS: int32 kv_idx[KV_TILE]
    // coalesced K tile loads
    // compute scores for KV_TILE
    // tile max -> online rescale -> tile weighted-V accumulation
}
```

Start KV_TILE 16/32; query tile 1 first at D=256, then test 2/4/8. Do not force MFMA/WMMA until resource counters show it wins on gfx1100 and gfx1201.

### 8. q8_0 follow-up

Do not gather/dequantize a full f16 cache. Extend `visit_token/tile` with QFP04/1296 dequant traits:

```cpp
if constexpr (KV_TYPE == GGML_TYPE_Q8_0) {
    // load only blocks covering selected kv token + D values
    // dequantize block into registers/LDS
} else {
    // f16 direct
}
```

Keep f16 and q8 dispatch separate until both pass direct tests.

## Code Samples & Guidance

Implementation order:

```text
1. refactor sel_idx out of current dense-mask builder; dense output byte-identical
2. CPU/reference compact-expansion test against current dense mask
3. add explicit GGML_OP_FLASH_ATTN_QSA constructor + supports_op
4. f16 one-query correctness kernel
5. unique selected+tail semantics
6. tile/coalesce and tune RDNA geometry
7. switch Qwen4Exp prefill dispatch under env/profile gate
8. allocation trace proves mask_all/indexer_sel dense tensors disappear
9. q8_0 only if required
```

Do not start by modifying the production dense FA kernel.

## Files

- `src/models/qwen4exp.cpp`: split compact `sel_idx` from dense materialization; sparse dispatch.
- `ggml/include/ggml.h`, `ggml/src/ggml.c`: explicit sparse-QSA op.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: supports/compute dispatch.
- `ggml/src/ggml-cuda/fattn-qsa.{cu,cuh}` or current FA sibling file.
- CPU/reference test helper + `tests/test-backend-ops.cpp`.
- QFP04/1296 quant traits for optional q8 K/V.

## Validation

Selection semantics first: for 1K/24K/80K/200K and query batches 1/4/8/64/256/512/1024, expand compact indices+tail and compare the exact allowed KV boolean set to the existing final `sel + kq_mask` result. Include dead/dump slots, ties, duplicates, tail overlap and partial final cell.

Kernel: sparse vs dense/CPU f32 on gfx1100/gfx1201; D=128/256; GQA mappings; tail boundaries. Record max/mean error, VGPR/SGPR/LDS/spills, occupancy and wall.

E2E: QFP17 ABBA 10K/80K/200K, ub512 and larger ubatch if memory permits. Greedy/logit check uses existing f32-reference near-tie policy.

## Effort & Risk

L/high. Main risks are selection-set semantics, causal position mapping, GQA layout, duplicate/tail double-counting, register pressure and over-broad generic API changes.

## Standards

One compact selection producer shared with dense fallback; explicit op/capability; CPU/reference oracle first; no dense QSA mask on sparse path; aggressive fallback; thresholds in QFP23 profiles.

## Acceptance Criteria

- Compact expansion exactly matches reference allowed KV set.
- Sparse graph has no QSA-specific `O(n_kv*n_query)` dense allocation.
- Sparse attention+mask wall falls >=50% or direct sparse kernel >=2x at representative 80K+ shape.
- E2E long-context prefill >=8% improvement on one production lane, decode <=1% regression.
- Unsupported backend/type/shape falls back to dense QSA/1332 safely.

## Change Log

- 2026-10-05T00:00:00+00:00 (created-by): Created as QFP17 QSA-SPARSE-PP implementation owner.
- 2026-10-05: Added concrete qwen4exp graph refactor, explicit GGML op, backend dispatch and HIP kernel skeleton grounded in the 0504396 dense-mask code.

## b11402 implementation review - 2026-10-05 prefill round

This section supersedes the **implementation order**, not the long-term direct-selected-index design above. Current pin is b11402 `d89651a7b205`. While this review was in progress, `patches/1334_hip_sparse_flash_attn` landed on `patch-refactor`; use it as the smallest hardware proof before adding a Qwen-specific op.

### What QSA computes at this pin

`src/models/qwen4exp.cpp::build_inp_kpool()` fixes the compact-list bound as:

```cpp
inp->n_sel = kpool*min(n_pool, indexer_top_k/kpool) + kpool - 1;
```

Production `kpool=4`, `indexer_top_k=2048` therefore gives **2051 individual KV-cell indices/query** once enough pools exist: 512 selected pools are expanded by `ggml_get_rows(pool_idxs, top_k)` to 2048 member-cell indices and concatenated with the 3-cell incomplete tail.

`build_qsa_sel()` then converts that compact set into dense semantics:

1. lightning indexer scores pools; `TOP_K` selects pools;
2. `pool_idxs` expands selected pools to cells and `tail_idxs` is appended;
3. dead/invisible slots are remapped to private dump rows `n_kv + slot` so duplicate dead writes are harmless;
4. `mask_all [n_kv+n_sel,n_tokens]` is filled `-inf`, selected rows are scattered to zero, then a view drops dump rows;
5. ordinary causal `kq_mask` is added.

`build_attn_qsa()` stores new K/V, reshapes that dense mask, points K/V at the **full cache**, and calls `build_attn_mha(..., mask, ..., n_sel, ...)`. At b11402 the CUDA FA implementation contains sparse MMA variants but mask compaction/selection is compiled out for HIP, so the production RDNA path executes dense FA over `n_kv`; the `-inf` mask suppresses unselected probabilities but does not avoid their K/V scan.

### 1295 decode gather: why it does not scale to ub512

`1295_qsa_gather_decode` retains the pre-remap compact `sel_idx` and, only for `n_tokens <= 8`, contiguous single-stream K/V and a large enough cache, gathers each query's selected K/V rows before FA. It pads the 2051 selections to 2304 and materializes tensors equivalent to `[D=256, Hkv=4, 2304, n_tokens]` in f16.

At 512 queries this is **2.25 GiB for K + 2.25 GiB for V = 4.50 GiB/ubatch**. Even without its 256-cell padding, 2051 selections are ~2.003 GiB each / ~4.006 GiB K+V; the I32 index list itself is only ~4.0 MiB. Therefore per-query gather + ordinary FA is a useful reference/microbenchmark, not the production prefill implementation.

### Prefill design options

1. **Materialized per-query gather + compact ordinary FA.** Simplest semantics and reuses 1295. Reject for production ub512 because of the 4.0-4.5 GiB K/V scratch above. Keep only as a small-query correctness oracle or very small query tile.
2. **Selected-index FA, direct K/V indirection. Preferred.** Keep K/V in the cache; for each query/query tile load only K/V rows named by the compact selection. b11402 already has the essential sparse MMA loader (`flash_attn_ext_f16<..., use_sparse_kernel=true>`); `1334_hip_sparse_flash_attn` is the lowest-risk first experiment because it enables the upstream mask->indices + indirect-K/V path on RDNA instead of inventing new attention math. If it wins, phase B should bypass dense-mask compaction and feed Qwen's compact cell list directly to the same loader, eliminating the remaining `O(n_kv*n_query)` QSA mask.
3. **Block-sparse over k-pool structure.** Preserve selected pool blocks and reuse K/V tiles across queries that chose the same pool. Potentially best locality, but 512 queries have different selections; the union can become dense and grouping/sort/dedup becomes a second scheduler. Defer until direct-index FA is measured.

At the measured first-100K point, dense FA is ~97 ms/ubatch on each XTX. Selection density is ~2051/100000 = 2.05%, so the arithmetic/data-read ideal is only a few ms; random-indirect loads and online-softmax overhead dominate before that floor. First practical target: **8-20 ms/ubatch**. Saving 77-89 ms versus 97 ms is ~12-14% of observed 100K wall; hard FA Amdahl is 18.9/120 = 15.8% wall (maximum throughput uplift +18.7%). At 200K, linear-context extrapolation makes dense FA ~194 ms/ubatch and ~25.1% of the ~304.7 s wall; a roughly fixed selected-set kernel in the 8-20 ms range would save ~23-24% E2E, below the +33.5% hard Amdahl speedup ceiling.

### 1334 correctness condition before model acceptance

1334 enables upstream's **query-group union** compaction. Its RDNA path selects the `(D=256,DV=256,ncols1=8,ncols2=8)` sparse specialization. `flash_attn_mask_to_sparse_indices<8>` ORs visibility across eight queries but allocates/caps each group's list at `n_kv_max`; Qwen passes the per-query bound (`n_sel`, ~2051), not a proven eight-query union bound. If eight queries select different pools, the union can exceed 2051 and truncation would silently omit valid KV rows.

Therefore the first backend test must include both:

- 512 queries sharing the same 2051-cell mask (performance/mechanism case);
- rotating/disjoint 2051-cell masks across each 8-query group (correctness/adversarial union case).

Compare against dense FA/CPU f32 and inspect compacted `counts`. **Do not run 1334 as a production model arm unless every group is untruncated.** If wide union is unsafe, force/instantiate `ncols1=1` as the next experiment; that uses the same indirect sparse loader with one query/list and avoids union semantics. Only after that proof should a direct-Qwen compact-index op be implemented.

### Exact b11402 seams

Qwen graph:

- `src/models/qwen4exp.cpp::build_inp_kpool()`: `inp->n_sel = ...` is the authoritative compact-list bound.
- `build_qsa_sel()`: anchors are `sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);`, the following `ggml_build_forward_expand(gf, sel_idx);`, and later `sel_idx = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);`. Phase B should return/retain the compact live-cell representation here and materialize `mask_all` only for fallback.
- `build_attn_qsa()`: seam is after K/V cache writes and before `// the selection mask already carries the causal mask`; phase B dispatches direct selected-index FA here.
- `build_layer_attn()`: existing `if (sel) { cur = build_attn_qsa(...) }` remains the model-level fallback switch.

FA/backend:

- `ggml/src/ggml-cuda/fattn.cu`: `flash_attn_mask_to_sparse_indices`, `ggml_cuda_flash_attn_ext_compact_mask`, `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse`, `...switch_ncols1`, `...switch_ncols2`.
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`: `flash_attn_ext_f16<..., use_sparse_kernel>` and `ggml_cuda_flash_attn_ext_mma_f16_case`; reuse its sparse index K/V load rather than a scalar attention kernel if the RDNA specialization proves correct/performs.
- Only if phase B requires an explicit op: `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cuda/ggml-cuda.cu` supports/compute dispatch, and Meta split handling for the new op. Do not add these API seams merely to qualify 1334.

### BigCherry package outline after 1334 proof

Do not create a competing patch now. `1334_hip_sparse_flash_attn` is already the opt-in phase-A package (`BIGCHERRY_FA_SPARSE=1`, default off). If phase A proves the RDNA loader but dense compaction remains material, create a **new next-free-ID** package (recheck the shared branch; do not assume 1335) for direct Qwen indices, e.g. `qsa_selected_index_fattn`.

Edits for that phase-B package:

- `qwen4exp.cpp`: expose compact selected cells from `build_qsa_sel`; direct sparse dispatch before dense mask materialization; fallback constructs the existing mask unchanged.
- FA source: entry that accepts `I32 [n_sel,n_query]` directly; reuse sparse MMA K/V indirection and online-softmax implementation.
- GGML/backend/Meta only if a distinct op is required after trying a private/fused-node seam.
- env docs: `BIGCHERRY_QSA_SEL_FA=0|1` default `0`; optional `BIGCHERRY_QSA_SEL_FA_MIN_KV=<cells>` default `32768`. Keep `BIGCHERRY_QSA_GATHER` for <=8-token 1295; do not overload it.
- patch tests: exact anchors/apply/idempotence, default-off source identity, unsupported shape/backend fallback, no dense-QSA allocation on selected path.

Fail closed unless production prerequisites match: HIP RDNA3/RDNA4, f16 K/V first, D=256 production head layout, valid I32 selection bounds, supported contiguous cache layout, no multi-stream/transposed-V special case, and context above threshold. Unsupported cases execute current dense QSA; <=8-token decode may continue using 1295 independently.

### Smallest proof / hardware gate

Before phase-B code, extend `test-backend-ops` or add a minimal FA microbenchmark for **KV=100000, Q=512, D=256, production GQA layout, n_sel=2051**, f16 K/V on gfx1100 and gfx1201:

1. dense masked FA baseline;
2. 1334 sparse, shared selected set;
3. 1334 sparse, rotating/disjoint selected sets per 8-query group;
4. if arm 3 truncates/fails, sparse `ncols1=1` prototype with the same masks.

Record sparse-list `count/max_count`, FA kernel wall, output max/mean error, VGPR/SGPR/LDS/spills and occupancy. Continue only if correctness is exact to the dense allowed-set semantics and direct sparse FA is >=2x (prefer <20 ms versus the ~97 ms model bucket). Then run QFP17 ABBA at ~100K and ~200K, flag 0/1, followed by composition with 1332 ub1024 only after each is independently proven.

Correctness gate: backend output vs dense + CPU-f32 reference; model logits/top-k at the known near-tie rather than byte identity against MTP; multi-request/cache-reuse stress; decode control <=1%.

### Correction - production KV heads / gather scratch

The memory arithmetic in `1295 decode gather: why it does not scale to ub512` above used `Hkv=4`. Production b11402 Qwen4Exp is **24 query heads / 2 KV heads** (GQA ratio 12). Correct f16 materialized-gather scratch at `Q=512`, `D=256`, `Hkv=2` is:

- `n_sel=2051`: per K or V = `512*2*256*2051*2` bytes = **1.0015 GiB**; K+V = **2.0029 GiB**.
- padded `n_sel=2304`: per K or V = **1.125 GiB**; K+V = **2.25 GiB**.
- I32 selection list: **4.006 MiB** unpadded / **4.5 MiB** padded.
- padded query tiles `Q={8,16,32}`: K+V scratch = **{36,72,144} MiB**.

The earlier `Hkv=4` / `4.50 GiB` total figures are superseded. The design conclusion is unchanged: full-ub512 materialized gather is still too expensive versus direct selected-index K/V loads; small query-tiled gather remains useful only as a correctness oracle or first prototype.

### 1334 RDNA dispatch correction after concurrent `bbe4bb96`

The shared branch changed 1334 after the review above: RDNA sparse WMMA now requires `ncols1*ncols2 >= 16`, and the current sparse specialization exists at `ncols2=8`; production prefill takes the 8x8 grouped path. Therefore the earlier suggestion to force `ncols1=1,ncols2=8` is **not executable on the current RDNA WMMA path**.

Revised failure branch for the smallest proof:

1. instrument or otherwise expose the **unclamped** `row_count`/group-union size before `min(row_count, n_kv_max)` for real Q=512 masks and the adversarial disjoint case;
2. if every 8-query union is `<=2051`, 1334 phase A remains semantically safe and can proceed to timing;
3. if any union exceeds 2051, reject 1334-as-is for model use: the compaction capacity must bound the group union, not the per-query `n_sel`; simply using the worst-case `8*n_sel` also expands sparse work/heuristic thresholds enough to erase much of the intended proof;
4. in that case, the next correctness implementation is the phase-B **per-query direct Qwen selected-index path** (or a new one-query-capable HIP FA specialization), not a forced 1x8 instantiation of the current WMMA kernel.

This supersedes only the earlier `ncols1=1` fallback sentence/arm; the shared-mask/adversarial-union tests, direct-index design, seams and acceptance gates remain unchanged.
