---
id: QFP26
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-05T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Persistent grouped MoE prefill MMQ and routed gate/up fusion

## Description

QFP17 already owns `MOE-MMQ-PP2`; QFP26 is the implementation owner after 1237/1265 compact-grid work. The important starting point is concrete: patch 1237 already builds `block_start[n_experts+1]` and `block_expert[max_m_blocks]` from `expert_bounds[]` with `mmq_build_moe_block_map()`, then launches `mul_mat_q<...>` over only real `(expert, expert-local-J-tile)` work. QFP26 must extend/reuse this map, not rebuild a second histogram/scan/task pipeline.

The two staged experiments are:
1. **scheduler experiment:** keep exactly the current MMQ math but feed compact work to a bounded persistent/static-strided worker grid to reduce long-tail workgroup imbalance;
2. **fusion experiment:** for the existing production topology `MUL_MAT_ID(gate) + MUL_MAT_ID(up) + GLU`, load one routed activation/task once, accumulate gate and up, and emit only the GLU result.

Patch 1239 already proves the exact graph topology and fusion guards used by production: gate/up must share the **same `cur` and `ids` tensor objects**, and each MM result must have exactly one GLU consumer. Reuse those guards/test semantics.

## Steps

1. Profile routed expert prefill after 1237/1265 at 10K/80K/200K, ub512/1024 where possible. Continue only if routed MoE gives >=3% E2E Amdahl bound.
2. Instrument 1237 compact map: real `total_blocks=block_start[n_experts]`, tasks/expert, max/p50/p90 task duration and end-of-kernel CU tail.
3. First scheduler arm: no new task descriptor. Reuse `block_expert[]` + `block_start[]`; persistent workers atomically claim compact block index `m` and derive `zt/jt` exactly as 1237 does today.
4. Compare persistent atomic claim, chunked atomic claim `{4,8}`, and deterministic static-strided assignment. Keep the simplest winner; persistence is not a goal itself.
5. Only then test larger expert-local row/token tiles. Allocation and launch must share the same selected J/tile width (upstream #29941/#29953 invariant).
6. Add gate/up fused routed MMQ only for a graph matching the existing 1239 fusion predicate: same `cur`, same `ids`, single consumer, compatible types/shape/precision.
7. Keep down projection separate. Do not fuse route-slot reduction/scatter; each output route slot remains single-writer/deterministic.
8. Preserve Meta/tensor-split semantics. QFP26 changes local expert compute only, not expert placement or collective semantics.
9. Gate by confirmed production quant types and architecture; unmatched type/shape/arch uses current 1237/1265 path.
10. Consolidate map prep: the promoted path may execute `mmq_build_moe_block_map()` once, never both 1237 prep plus a QFP26 duplicate prep.

## Detailed Solution & Technical Design

### 1. Reuse the exact 1237 compact representation

Current 1237 prep kernel (source-shaped excerpt, keep semantics):

```cpp
static __global__ void mmq_build_moe_block_map(
        const int32_t * __restrict__ expert_bounds,
        const int n_experts,
        const int J,
        int32_t * __restrict__ block_start,
        int32_t * __restrict__ block_expert) {
    extern __shared__ int32_t s_start[];
    const int tid = threadIdx.x;

    for (int e = tid; e < n_experts; e += blockDim.x) {
        const int count = expert_bounds[e + 1] - expert_bounds[e];
        s_start[e] = (count + J - 1) / J;
    }
    __syncthreads();

    if (tid == 0) {
        int total = 0;
        for (int e = 0; e < n_experts; ++e) {
            const int n = s_start[e];
            s_start[e] = total;
            total += n;
        }
        s_start[n_experts] = total;
    }
    __syncthreads();

    for (int e = tid; e <= n_experts; e += blockDim.x) {
        block_start[e] = s_start[e];
    }
    const int total = s_start[n_experts];
    for (int m = tid; m < total; m += blockDim.x) {
        int lo = 0, hi = n_experts;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (s_start[mid] <= m) lo = mid + 1;
            else                   hi = mid;
        }
        block_expert[m] = lo - 1;
    }
}
```

Current compact kernel mapping is:

```cpp
const int m_block = blockIdx.y;
if (m_block >= rd30_block_start[rd30_n_experts]) return;
zt = rd30_block_expert[m_block];
jt = m_block - rd30_block_start[zt];
wt = 0;
```

QFP26 scheduler must derive **the same** `(zt,jt)` from claimed `m`; this keeps correctness independent of scheduling order.

### 2. First scheduler experiment: static-strided compact workers

Before atomics, test a bounded worker grid with deterministic static striding. Add a scheduler mode to `mul_mat_q` for non-stream-K compact MoE:

```cpp
enum bc_moe_sched : int {
    BC_MOE_SCHED_DIRECT = 0,   // current 1237: blockIdx.y == m
    BC_MOE_SCHED_STRIDED,
    BC_MOE_SCHED_ATOMIC,
};
```

Kernel-side compact index loop should factor the existing tile body rather than recurse into the kernel:

```cpp
template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static __device__ __forceinline__ void bc_run_moe_tile(
        /* existing mul_mat_q args */,
        const int zt,
        const int jt,
        const int it) {
    // Move/factor only the body after 1237 derives wt/zt/jt/it.
    // No math/index changes in scheduler-only experiment.
}
```

Then:

```cpp
const int total = rd30_block_start[rd30_n_experts];
for (int m = blockIdx.y; m < total; m += gridDim.y) {
    const int zt = rd30_block_expert[m];
    const int jt = m - rd30_block_start[zt];
    bc_run_moe_tile<type,J,fallback,prec_src1>(/*...*/, zt, jt, blockIdx.x);
}
```

Host launch:

```cpp
const int total_upper = (int) rd30_max_m_blocks;
const int worker_blocks = std::min(total_upper,
    bc_moe_workers_per_cu(cc) * ggml_cuda_info().devices[id].nsm);

if (bc_sched == BC_MOE_SCHED_STRIDED && worker_blocks > 0) {
    block_nums = dim3(nty, worker_blocks, 1);
}
```

Use the real device CU/SM field name at the current pin (`nsm` here is illustrative if the struct differs). Do not hard-code 96/64 CUs.

Why static-strided first: it removes the launch grid's dependence on `total` without a global atomic and still lets long blocks be spread across a bounded resident grid. If task durations are similar, it should beat or equal an atomic queue.

### 3. Persistent/atomic arm reusing the same map

Allocate one counter from the existing CUDA pool only when atomic mode is selected:

```cpp
ggml_cuda_pool_alloc<uint32_t> bc_next(ctx.pool(id));
bc_next.alloc(1);
GGML_CUDA_CHECK(cudaMemsetAsync(bc_next.ptr, 0, sizeof(uint32_t), stream));
```

Kernel claim with optional chunking:

```cpp
template <int CLAIM>
__device__ __forceinline__ uint32_t bc_claim(uint32_t * next) {
    __shared__ uint32_t base;
    if (threadIdx.x == 0) {
        base = atomicAdd(next, CLAIM);
    }
    __syncthreads();
    return base;
}

const int total = rd30_block_start[rd30_n_experts];
for (;;) {
    const uint32_t base = bc_claim<CLAIM>(bc_next);
    if (base >= (uint32_t) total) break;

    #pragma unroll
    for (int k = 0; k < CLAIM; ++k) {
        const uint32_t m = base + k;
        if (m >= (uint32_t) total) break;
        const int zt = rd30_block_expert[m];
        const int jt = (int) m - rd30_block_start[zt];
        bc_run_moe_tile<type,J,fallback,prec_src1>(/*...*/, zt, jt, blockIdx.x);
    }
    __syncthreads();
}
```

Important: `CLAIM > 1` means one workgroup serially executes several compact tasks. Only test this if each task body has no persistent shared state across calls; reset/reinitialize all shared scratch inside `bc_run_moe_tile` or keep CLAIM=1.

### 4. Do not read `block_start[n_experts]` back to host

The current host only knows `rd30_max_m_blocks`, while the real total is device-computed. Do not add a D2H sync just to size a persistent grid. Launch a bounded worker count based on the upper bound/CU count; workers read `block_start[n_experts]` on device and exit/loop accordingly.

### 5. Gate/up fusion must reuse the existing production matcher contract

Patch 1239 builds the exact test graph:

```cpp
ggml_tensor * gate_mm = ggml_mul_mat_id(ctx, gate_w, cur, ids);
ggml_tensor * up_mm   = ggml_mul_mat_id(ctx, up_w,   cur, ids);

ggml_tensor * out = (glu_op == GGML_GLU_OP_SWIGLU_OAI)
    ? ggml_swiglu_oai(ctx, gate_mm, up_mm, 1.702f, 7.0f)
    : ggml_glu_split(ctx, gate_mm, up_mm, glu_op);
```

The existing CUDA fusion scanner requires pointer identity for `cur`/`ids` and one consumer for each projection. Extend **that** fusion path for prefill routed MMQ; do not add a Qwen4Exp-only graph matcher.

Dispatcher guard shape:

```cpp
static bool bc_can_fuse_moe_gate_up_pp(
        const ggml_tensor * gate,
        const ggml_tensor * up,
        const ggml_tensor * glu,
        int cc) {
    if (gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID ||
        glu->op != GGML_OP_GLU) {
        return false;
    }
    if (gate->src[1] != up->src[1] || gate->src[2] != up->src[2]) {
        return false; // same activation + same ids objects
    }
    if (!ggml_are_same_shape(gate, up)) {
        return false;
    }
    if (gate->src[0]->type != up->src[0]->type) {
        return false; // mixed-type specialization is a separate experiment
    }
    const int64_t n_tokens = gate->ne[2] * gate->ne[3]; // verify real routed layout
    return n_tokens >= bc_moe_pp_fuse_min_tokens(cc, gate->src[0]->type);
}
```

Use the existing graph consumer-count/elidability helper rather than manually recounting consumers if available in `ggml_cuda_can_fuse()`.

### 6. Fused MMQ kernel interface

Do not create two independent compact maps. Pass the same `expert_bounds`, `block_start`, `block_expert` into one dual-weight launch:

```cpp
template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>
static __global__ void mul_mat_q_moe_gate_up(
        const void * __restrict__ w_gate,
        const void * __restrict__ w_up,
        const void * __restrict__ x_q81,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        void * __restrict__ dst_glu,
        const int32_t * __restrict__ block_expert,
        const int32_t * __restrict__ block_start,
        int n_experts,
        /* existing MMQ strides/config */,
        ggml_glu_op glu_op,
        float alpha,
        float limit) {

    const int m = bc_next_m(/* direct/strided/atomic mode */);
    if (m < 0 || m >= block_start[n_experts]) return;

    const int zt = block_expert[m];
    const int jt = m - block_start[zt];

    // Reuse existing MMQ tile loaders/dequant traits. Load activation tile once.
    bc_mmq_x_tile x = bc_load_q81_tile(/* x_q81, expert_bounds, zt, jt ... */);

    bc_mmq_acc gate_acc = bc_dot_weight_tile<type,J,fallback,prec_src1>(w_gate, x, zt, jt, /*...*/);
    bc_mmq_acc up_acc   = bc_dot_weight_tile<type,J,fallback,prec_src1>(w_up,   x, zt, jt, /*...*/);

    bc_store_glu(dst_glu, gate_acc, up_acc, glu_op, alpha, limit, /* route-slot indices */);
}
```

`bc_mmq_x_tile`, `bc_mmq_acc`, `bc_dot_weight_tile`, `bc_store_glu` are **refactoring targets**, not claimed current symbols. Extract them from the current `mul_mat_q` body only after the scheduler-only stage proves the factorization does not regress the ordinary kernel. If factoring the entire body causes codegen loss, implement a sibling fused template sharing lower-level quant/dequant helpers instead.

### 7. Exact GLU semantics

Do not assume `silu(gate)*up` universally. Dispatch by the actual `ggml_glu_op`. For the two existing 1239 test variants:

```cpp
__device__ inline float bc_apply_glu(float gate, float up,
                                     ggml_glu_op op, float alpha, float limit) {
    switch (op) {
        case GGML_GLU_OP_SWIGLU:
            return (gate / (1.0f + expf(-gate))) * up;
        case GGML_GLU_OP_SWIGLU_OAI: {
            const float g = fminf(gate, limit);
            const float u = fminf(fmaxf(up, -limit), limit);
            return (g / (1.0f + expf(-alpha * g))) * (u + 1.0f);
        }
        default:
            // Do not silently approximate GEGLU/etc. Add exact existing helper or fall back.
            return NAN;
    }
}
```

The literal formula must be replaced with/reuse the exact existing GGML CUDA GLU helper so fast-math/order matches current fused semantics; this sample documents required dispatch/fallback, not permission to create a numerically different activation.

### 8. Output ownership/determinism

Each compact `(expert,jt,it)` task must write the same route-slot region as current MMQ. Never atomically accumulate multiple experts directly into the final token output inside the persistent kernel. Keep current route-slot result and downstream reduction/scatter order. Scheduling may reorder **independent writers**, not floating-point reductions.

## Code Samples & Guidance

Implementation sequence for an agent:

```text
A. instrument current 1237 map: total compact tasks + task-duration/tail census
B. factor (zt,jt,it) tile body with DIRECT mode; prove identical performance/correctness
C. add STRIDED mode; AB
D. add ATOMIC mode only if tail evidence justifies it
E. tune row/J geometry with one shared selector/allocation width
F. reuse existing MUL_MAT_ID+GLU fusion detector from 1239 topology
G. add one same-type gate/up fused MMQ specialization
H. extend quant/arch coverage only from measured production types
```

Required diagnostics:

```text
BIGCHERRY_MOE_PP layer=%d tokens=%lld total=%d workers=%d sched=direct|strided|atomic claim=%d max_tasks_expert=%d wall_us=...
BIGCHERRY_MOE_GATEUP layer=%d type=%s J=%d tasks=%d fused=1 glu=%d
```

## Files

- `patches/1237_rd30_moe_mmq_compact_grid/patch.py`: reuse/factor compact map and launch state.
- `patches/1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2/`: architecture extension; no duplicate scheduler.
- `ggml/src/ggml-cuda/mmq.cuh`: `mmq_build_moe_block_map`, compact `mul_mat_q`, scheduler/fused template.
- Current CUDA fusion scanner/dispatcher in `ggml/src/ggml-cuda/ggml-cuda.cu`.
- `patches/1239_hi119_fused_moe_glu_test_case/patch.py`: direct correctness fixture/topology.
- QFP17 profiling scripts + runtime thresholds under QFP23 after tuning.

## Validation

Direct scheduler tests: same compact map/output under DIRECT/STRIDED/ATOMIC across routing patterns `{uniform, one-hot, Zipf, many-empty, all-zero}`, token counts `{64,256,512,1024,2048}`, tail J tiles and production expert count/top-k. Verify route-slot output byte-for-byte for scheduler-only change.

Fused tests: extend 1239 with production routed quant type/shape. Native unfused graph vs fused graph through ordinary backend compute; shared `cur`/`ids` object identity; negative tests for extra consumer, differing ids, differing activation, mixed type and small-N fallback.

Hardware gfx1100/gfx1201: task p50/p90/max, kernel tail, CU occupancy, VGPR/LDS/spills, launch count, MoE critical-rank wall and E2E prefill. Decode/MTP must remain current path.

## Effort & Risk

L/high. Scheduler stage is moderate; factorization/codegen and dual-weight fused MMQ are high risk. Main risks: scheduler overhead exceeding tail savings, shared-memory state across claimed tasks, VGPR spills from dual accumulators, mismatched J allocation/launch width, and numerical drift in GLU helper choice.

## Standards

QFP17 acceptance umbrella; reuse 1237's map and 1239's production fusion contract; no second routing/task pipeline; single-writer route-slot outputs; one J selection for allocation+launch; confirmed quant types only; fallback-first.

## Acceptance Criteria

- Measured routed-MoE Amdahl bound >=3% before fused work.
- Scheduler winner reduces MoE kernel-tail p90 >=20% or critical MoE wall >=10% without >1% regression elsewhere; if DIRECT wins, record persistent scheduling rejected.
- Gate/up fusion removes one activation reload/materialization path and separate projection/GLU launch boundaries, with >=10% routed-MoE wall improvement and no damaging spills.
- Combined E2E prefill >=3% on representative lane; decode/MTP <=1% regression; accepted numerical contract.

## Notes

1237's real benefit is dispatch/block compaction, not removing full wasted compute; QFP26 therefore starts from its compact map and specifically targets residual imbalance/repeated activation traffic. Upstream dense #29948 is mechanism evidence only.

## Change Log

- 2026-10-05T00:00:00+00:00 (created-by): Created as QFP17 MOE-MMQ-PP2 implementation owner.
- 2026-10-05: Added source-shaped scheduler/fusion code grounded in 1237 compact-map implementation and 1239 production MUL_MAT_ID+GLU test topology.
