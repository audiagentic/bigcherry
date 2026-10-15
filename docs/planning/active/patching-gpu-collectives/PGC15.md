---
id: PGC15
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-10-05T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Prefill token-tiled compute/AllReduce overlap for tensor-split Meta execution

## Description

PGC12 measured the current prefill limit directly: on the profiled pp2048 ubatch, per-device compute is ~0.93 s while AllReduce service is ~2.4 s. Provider selection alone cannot remove that serialized service. This item owns the generic implementation for producing and reducing tensor-parallel outputs in token tiles so communication for tile `i` can overlap compute for tile `i+1`.

QFP17 remains the Flash-Next prefill measurement/acceptance umbrella. PGC15 owns the backend/scheduler primitive, stream/event lifetime, eligibility proof and generic tensor-split execution. PGC14 owns RCCL transport throughput; PGC12 owns phase-aware provider routing.

Pinned source contract at llama.cpp `0504396140d1`: Meta currently obtains one provider callback by proc address:

```cpp
typedef bool (*ggml_backend_comm_allreduce_tensor_t)(void * comm_ctx, struct ggml_tensor ** tensors);
```

and `ggml_backend_meta_graph_compute()` calls it once per reduction boundary after every rank's subgraph has completed. PGC15 extends that exact seam; it does not add a second collective registry.

## Steps

1. Add a prefill AllReduce trace recording graph uid, subgraph, tensor name/op, shape/type, split state, bytes, token stride, producer, immediate consumers, provider and per-rank producer/AR timestamps.
2. Mark a boundary tile-safe only when every rank contributes the same logical token range, the range is byte-contiguous, the reduction is SUM, the producer supports token-range execution and no consumer can observe an unreduced tile.
3. Extend the existing backend proc-address contract with an optional range callback. Whole-tensor callback remains the fallback and ABI behavior for providers without range support.
4. Add a provider-owned communication stream/event ring. Do not require Meta itself to expose events: at this pin Meta reports `.events = false`; the range provider already owns the simple GPU backends and can use their CUDA/HIP streams/events internally.
5. Implement one producer family first. The producer must expose a real token-range launch; do not shorten tensor metadata and hope an existing kernel only writes the requested rows.
6. First stage may wait for all reduced tiles before the existing whole-tensor consumer. This still overlaps `producer(tile i+1)` with `AR(tile i)`. Tile consumers are phase 2 only.
7. Sweep tile `{64,128,256,512}` and depth `{2,3,4}`. Keep whole-message fallback below a measured byte threshold.
8. Fail closed for unsupported provider, non-contiguous token axis, graph capture incompatibility, recurrent producer, tail-layout mismatch or rank disagreement.
9. Compose independently with PGC14 using four arms: stock/patched RCCL × untiled/tiled.
10. Extend to additional AR families only by measured Amdahl ranking.

## Detailed Solution & Technical Design

### 1. Extend the existing communication ABI narrowly

At `ggml/include/ggml-backend.h`, next to `ggml_backend_comm_allreduce_tensor_t`, add an optional proc-address-only typedef. Do **not** replace the existing callback:

```cpp
// Existing ABI stays unchanged.
typedef bool (*ggml_backend_comm_allreduce_tensor_t)(
        void * comm_ctx, struct ggml_tensor ** tensors);

// BigCherry experiment: all ranks reduce the same byte range of their tensor.
// offset/size refer to tensor->data and must be identical on all ranks.
typedef bool (*ggml_backend_comm_allreduce_tensor_range_t)(
        void * comm_ctx,
        struct ggml_tensor ** tensors,
        size_t offset,
        size_t size,
        uint32_t slot,
        bool wait_for_completion);
```

Provider registration remains via `ggml_backend_reg_get_proc_address()`. In `ggml_backend_meta_context` add:

```cpp
ggml_backend_comm_allreduce_tensor_t       comm_allreduce       = nullptr;
ggml_backend_comm_allreduce_tensor_range_t comm_allreduce_range = nullptr;
```

and constructor lookup:

```cpp
if (comm_ctx != nullptr) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(
        ggml_backend_get_device(simple_backends[0]));

    comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");

    comm_allreduce_range = (ggml_backend_comm_allreduce_tensor_range_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor_range");

    GGML_ASSERT(comm_allreduce != nullptr);
}
```

No range proc -> current whole-tensor path exactly.

### 2. Prove byte-range mapping; never infer it from `ne[]` alone

For the first implementation require tokens to occupy one contiguous outer dimension:

```cpp
static bool bc_token_range(
        const ggml_tensor * t,
        int64_t token_axis,
        int64_t t0,
        int64_t nt,
        size_t * offset,
        size_t * bytes) {
    if (!ggml_is_contiguous(t) || token_axis < 0 || token_axis >= GGML_MAX_DIMS) {
        return false;
    }

    size_t inner = ggml_type_size(t->type);
    for (int d = 0; d < token_axis; ++d) {
        inner *= (size_t) t->ne[d];
    }

    // Exact first-version invariant: one token slice is tightly packed.
    if (t->nb[token_axis] != inner || t0 < 0 || nt <= 0 || t0 + nt > t->ne[token_axis]) {
        return false;
    }

    *offset = (size_t) t0 * t->nb[token_axis];
    *bytes  = (size_t) nt * t->nb[token_axis];
    return true;
}
```

If the real target tensor places token on another logical axis or has padding, add an explicit shape-specific mapper for that producer; do not weaken this check globally.

### 3. Provider-side ring: the provider owns streams/events

The CUDA/HIP collective implementation should allocate persistent state once in `comm_init`, not once per graph:

```cpp
struct bc_ar_slot {
    cudaEvent_t ready [GGML_BACKEND_META_MAX_DEVICES] = {};
    cudaEvent_t done  [GGML_BACKEND_META_MAX_DEVICES] = {};
    uint64_t generation = 0;
};

struct bc_ar_range_state {
    cudaStream_t comm[GGML_BACKEND_META_MAX_DEVICES] = {};
    bc_ar_slot slots[4];
    uint32_t depth = 3;
};
```

Use the existing `cuda*` portability wrappers used by ggml-cuda so HIP builds map these to HIP APIs. Creation belongs beside the current AllReduce provider context.

For each range call, the provider must wait on the producer stream without host synchronization. The exact stream accessor is backend-private; implement it beside the existing provider rather than exporting a generic stream API.

```cpp
static bool bc_allreduce_range(
        bc_comm_ctx * ctx,
        ggml_tensor ** tensors,
        size_t offset,
        size_t size,
        uint32_t slot_id,
        bool wait_for_completion) {
    bc_ar_slot & s = ctx->range.slots[slot_id % ctx->range.depth];

    for (size_t r = 0; r < ctx->n_backends; ++r) {
        cudaStream_t compute = ggml_cuda_backend_stream(ctx->backends[r]); // private helper to add/reuse
        GGML_CUDA_CHECK(cudaEventRecord(s.ready[r], compute));
        GGML_CUDA_CHECK(cudaStreamWaitEvent(ctx->range.comm[r], s.ready[r], 0));
    }

    // Reuse the existing provider implementation, but pass data+offset / count=size.
    // The RCCL implementation calls ncclAllReduce/rcclAllReduce on ctx->range.comm[r].
    if (!bc_provider_allreduce_bytes(ctx, tensors, offset, size, ctx->range.comm)) {
        return false;
    }

    for (size_t r = 0; r < ctx->n_backends; ++r) {
        GGML_CUDA_CHECK(cudaEventRecord(s.done[r], ctx->range.comm[r]));
        if (wait_for_completion) {
            cudaStream_t compute = ggml_cuda_backend_stream(ctx->backends[r]);
            GGML_CUDA_CHECK(cudaStreamWaitEvent(compute, s.done[r], 0));
        }
    }
    ++s.generation;
    return true;
}
```

`ggml_cuda_backend_stream`/`bc_provider_allreduce_bytes` are proposed private helpers, not claimed upstream symbols. Their implementation must be a refactor of the existing provider's stream and collective call, not duplicated transport code.

### 4. Meta call-site: preserve the current nodes vector

The current Meta path already collects the last node from each rank subgraph before calling `comm_allreduce`. Keep that ownership model:

```cpp
std::vector<ggml_tensor *> nodes;
nodes.reserve(n_backends);
for (size_t j = 0; j < n_backends; ++j) {
    auto & bcj = backend_ctx->backend_configs[j];
    ggml_cgraph * cg = bcj.cgraphs[i].cgraph_main;
    nodes.push_back(cg->nodes[cg->n_nodes - 1]);
}
```

Tiled path becomes:

```cpp
static bool bc_meta_reduce_tiles(
        ggml_backend_meta_context * ctx,
        ggml_tensor ** nodes,
        size_t n_backends,
        int64_t token_axis,
        int64_t n_tokens,
        int64_t tile_tokens) {
    if (ctx->comm_allreduce_range == nullptr) {
        return false;
    }

    for (int64_t t0 = 0, it = 0; t0 < n_tokens; t0 += tile_tokens, ++it) {
        const int64_t nt = std::min<int64_t>(tile_tokens, n_tokens - t0);
        size_t off0 = 0, bytes0 = 0;
        if (!bc_token_range(nodes[0], token_axis, t0, nt, &off0, &bytes0)) {
            return false;
        }
        for (size_t r = 1; r < n_backends; ++r) {
            size_t off = 0, bytes = 0;
            if (!bc_token_range(nodes[r], token_axis, t0, nt, &off, &bytes) || off != off0 || bytes != bytes0) {
                return false;
            }
        }

        const bool last = t0 + nt == n_tokens;
        if (!ctx->comm_allreduce_range(ctx->comm_ctx, nodes, off0, bytes0, (uint32_t) it, last)) {
            return false;
        }
    }
    return true;
}
```

This code is only useful **after** producer tiles become ready independently. Calling it after a whole producer graph completes changes message size but gives no compute/communication overlap; instrumentation must reject that false success.

### 5. Producer seam: real token-range launch required

Implement the first producer as a dedicated range-capable launch wrapper. Example shape for a row-parallel projection backend:

```cpp
struct bc_token_tile {
    int64_t t0;
    int64_t nt;
};

template <typename launch_full_fn>
static void bc_launch_projection_tile(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        bc_token_tile tile,
        cudaStream_t stream,
        launch_full_fn launch) {
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const size_t x_off = (size_t) tile.t0 * src1->nb[1];
    const size_t y_off = (size_t) tile.t0 * dst ->nb[1];

    launch(
        (const char *) src1->data + x_off,
        (char *) dst->data + y_off,
        tile.nt,
        stream);
}
```

Adapt the token axis/strides to the selected real kernel. The key requirement is that `tile.nt` reaches the kernel launch and grid; a view alone is insufficient if the backend cached the original dimensions.

## Code Samples & Guidance

Implementation order for an agent:

1. Add range typedef + optional proc lookup; build with no provider implementation and prove fallback is byte-identical.
2. Refactor current RCCL provider's whole-tensor call into a `(..., offset, bytes, stream[])` internal helper; whole-tensor passes `offset=0, bytes=ggml_nbytes(tensor)`.
3. Add provider comm streams/events and direct synthetic range tests.
4. Add `bc_token_range()` assertions/trace in Meta, still after whole producer; prove ranges/counts.
5. Add one real token-range producer and enqueue range AR immediately after each tile.
6. Only then claim overlap; rocprof must show producer tile `i+1` concurrent with collective tile `i`.

Required trace:

```text
BIGCHERRY_AR_TILE uid=%llu sg=%zu name=%s t0=%lld nt=%lld off=%zu bytes=%zu slot=%u provider=%s
```

## Files

- `ggml/include/ggml-backend.h`: optional range callback typedef.
- `ggml/src/ggml-backend-meta.cpp`: lookup, eligibility, range validation, call sequencing.
- `ggml/src/ggml-cuda/allreduce.{cu,cuh}` or current provider files: range helper + private comm stream/event ring.
- First measured producer backend file: actual token-range launch.
- Existing AR telemetry patches/tools; QFP17 records E2E evidence.

## Validation

Synthetic direct tests: 2/3 ranks, f32 tensors, token counts `{1,63,64,65,255,256,257,1024}`, all tail cases, compare range-tiled result byte-for-byte to whole exact AllReduce. Negative tests: mismatched rank range, non-contiguous tensor, missing range proc, unsupported provider.

Hardware: Brutus 2x XTX and production 2x XTX+R9700. ABBA pp1024/2048/4096 and Flash-Next 10K/80K/200K fills. Record prefill t/s, TTFT, exposed collective wall, producer wall, overlap %, PCIe throughput, rank skew and decode control.

Graph reuse/capture on/off, repeated requests and prompt-cache reuse must not stale-reuse events/slots.

## Effort & Risk

L. Primary risk is producer tiling, not the range collective. Other risks: RCCL per-tile latency, event/stream ownership, graph capture, rank ordering and padding/stride mistakes.

## Standards

One provider registry; no host sync in tiled hot path; exact byte-range proof; private provider stream helpers rather than generic HIP leakage; fail closed; source-shaped code must be reconciled to the exact provider implementation before patch generation.

## Acceptance Criteria

- One high-volume prefill family shows timeline-proven `compute(tile i+1)` overlapping `AR(tile i)`.
- Exact f32 direct tests match whole reduction byte-for-byte.
- Recover >=30% of that family's serialized AR wall or improve representative Flash-Next prefill >=10%; otherwise do not generalize.
- Decode <=1% change.
- No global/device synchronization or unbounded stream/event allocation.

## Notes

PGC12 identified overlap as the main prefill collective lever. PGC15 is the concrete implementation owner. PGC14 reduces remaining service time and is independently selectable.

## Change Log

- 2026-10-05T00:00:00+00:00 (created-by): Created by agent from prefill plan scan.
- 2026-10-05: Added source-shaped implementation skeleton grounded in llama.cpp 0504396 Meta/provider ABI and backend event model.

## Phase-1 patch outline - b11402 / Brutus prefill round

This supersedes only the old-pin implementation details above; the ownership and fail-closed design remain. Current pin is llama.cpp b11402 `d89651a7b205`. Active 3-rank provider is RCCL/NCCL, and its implementation is in `ggml/src/ggml-cuda/ggml-cuda.cu`. **Do not edit `allreduce.cu` in phase 1**: that file implements the upstream internal provider, whose b11402 entry point asserts `n_backends == 2`; Brutus prefill has 3 ranks. Add internal/root3 range support only after RCCL overlap is proven.

### Exact b11402 seams / anchors

1. `ggml/include/ggml-backend.h`: beside `ggml_backend_comm_allreduce_tensor_t`. Add optional range capability typedefs; leave the existing callback untouched.
2. `ggml/src/ggml-backend-meta.cpp` around b11402 lines ~1800-1845, `struct ggml_backend_meta_context`: existing anchors are `void * comm_ctx = nullptr;` and `ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;`. Add range callback + preflight callback and proc-address lookup in the constructor.
3. `ggml/src/ggml-backend-meta.cpp` around ~2430-2470, `ggml_backend_meta_graph_compute()`: current anchor is the loop which runs every rank's `bcj.cgraphs[i].cgraph_main`, then builds `nodes` from the last node and calls `backend_ctx->comm_allreduce(...)`. This is the serialization point to replace only for a preflight-qualified tiled boundary.
4. `src/models/qwen4exp.cpp`, `llama_model_qwen4exp::graph::build_layer_attn()` (b11402 ~930-1050): the first producer is already named by the anchor `cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s); cb(cur, "attn_output", il);`. **No model edit is required in phase 1.** Meta qualifies the terminal simple-backend node corresponding to this `GGML_OP_MUL_MAT` boundary.
5. `ggml/src/ggml-cuda/ggml-cuda.cu` around ~1000-1200, `ggml_backend_cuda_comm_allreduce_nccl()`: factor the large/small collective body into a private range helper accepting element offset/count, an explicit stream array, and the reduction precision selected for the **whole original reduction**. Do not recalculate the F32/BF16 threshold per tile: a whole reduction can be above the threshold while an individual tile is below it. Preserve the current F32->BF16 / RCCL BF16 / BF16->F32 behavior selected before partitioning.
6. `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml_backend_cuda_reg_get_proc_address()`: export the two optional range symbols. Existing whole-tensor provider remains the fallback.

### Patch package

Next free ID at review time: `patches/1335_prefill_ar_tile_overlap/` (recheck before creation). Proposed flag defaults OFF:

```python
GROUP = "core"
STATE = "untested"

ENV_DOCS = (
    EnvDoc("BIGCHERRY_AR_TILE_TOKENS", "<tokens>", "0",
           "prefill attention-output token tile size for compute/AllReduce overlap; 0 disables"),
    EnvDoc("BIGCHERRY_AR_TILE_TRACE", "0|1", "0",
           "log tiled AllReduce eligibility, ranges, slots and fallback reasons"),
)
```

Initial accepted values: 128/256; 64/512 may be diagnostic only. Use ordinary `Edit` anchors against the exact strings above; no broad regex replacement of provider code. Unit tests must apply/idempotence-check every edit and assert default-off preserves the original whole-provider call.

### Communication ABI: add preflight, not partial fallback

A callback failure after tile 0 has executed cannot safely fall back to the whole-tensor path. Add a provider preflight so all failure happens before any tiled producer work:

```cpp
typedef bool (*ggml_backend_comm_allreduce_tensor_range_supported_t)(
        void * comm_ctx, struct ggml_tensor ** tensors,
        size_t max_offset, size_t max_size);

typedef bool (*ggml_backend_comm_allreduce_tensor_range_t)(
        void * comm_ctx, struct ggml_tensor ** tensors,
        size_t offset, size_t size, uint32_t slot,
        bool wait_for_completion);
```

Meta takes the tiled path only if `range_supported()` succeeds for every planned range. After that, a runtime range failure is an execution error, not a silent whole-tensor retry.

The provider's precision decision is also part of preflight. At the b11552 RCCL pin, the F32/BF16 choice is made from the whole reduction's element count. For example, a 768 x 512 reduction is above the 131072-element boundary while a 768 x 128 tile is below it; choosing independently per tile would silently change the arithmetic. Snapshot the whole-reduction policy before launching tile zero and carry it through every range. If the range provider cannot honor that explicit policy, reject the tiled path before any producer work is submitted.

### First producer: attention `wo`, not FFN/MoE

Qwen4Exp has two reductions/layer, but start with the attention output projection. It is a plain terminal `GGML_OP_MUL_MAT` (`attn_output`) whose logical activation/output token axis is contiguous. FFN/MoE is deliberately phase 2: its terminal partial result depends on routed `MUL_MAT_ID`, expert compaction, gate/up work and weighted expert reduction; tiling it changes expert grouping and interacts directly with 1237/1265.

At Meta graph rebuild, qualify the simple-backend subgraph only when its last real node is the attention-output `MUL_MAT`. Build persistent auxiliary graphs, do not mutate the original graph at execution time:

- `prefix`: all nodes before terminal `wo`, executed once for the whole ubatch;
- `producer[tile]`: a one-node shallow clone of `wo` with an input view and output/data range for `[t0, t0+nt)`; `ne[1]=nt` reaches the backend launch/grid;
- original whole cgraph remains untouched for fallback.

For the first version require F32 contiguous activation/output with token axis 1:

```text
src1 offset = t0 * src1->nb[1]
dst  offset = t0 * dst ->nb[1]
bytes/tile  = nt * dst->nb[1]
```

Require `src1->nb[1] == src1->ne[0]*sizeof(float)` and the equivalent output invariant. A mere Meta view without changing the producer node dimensions is not sufficient.

Execution for one qualified boundary:

```text
all ranks: graph_compute_async(prefix)
for tile i:
    all ranks: graph_compute_async(producer[i]) on normal compute stream
    provider: record ready[i] on compute stream
    provider comm stream waits ready[i]
    provider: F32->BF16(range i), RCCL BF16 AllReduce(range i), BF16->F32(range i)
    provider returns without making compute wait
    -> producer[i+1] may now overlap collective[i]
last tile:
    record done on comm stream; normal compute stream waits done
continue with existing next Meta subgraph
```

One provider comm stream per device serializes tile collectives; therefore waiting on the last tile's done event also orders every earlier tile. No host/device synchronize in the hot path.

### RCCL stream/event ownership

Extend the existing CUDA communication context, not Meta, with persistent resources:

```cpp
struct bc_ar_range_slot {
    cudaEvent_t ready[GGML_CUDA_MAX_DEVICES] = {};
};
struct bc_ar_range_state {
    cudaStream_t comm[GGML_CUDA_MAX_DEVICES] = {};
    bc_ar_range_slot slot[4];
    cudaEvent_t done[GGML_CUDA_MAX_DEVICES] = {};
    // persistent BF16 scratch per rank, sized for max tile elements
};
```

Create `cudaStreamNonBlocking` streams/events at comm init and destroy them in comm free. For each rank, obtain the private `ggml_backend_cuda_context` already held in the provider and use `cuda_ctx->stream()` as the producer stream. `cudaEventRecord(ready, compute)` -> `cudaStreamWaitEvent(comm, ready)`. Run conversion and `ncclAllReduce` on `comm`. Record `done` after the BF16->F32 conversion. On final tile issue `cudaStreamWaitEvent(compute, done)`.

Do not allocate `ggml_cuda_pool_alloc` scratch independently per tile: use provider-owned persistent BF16 scratch sized during preflight/init so asynchronous tile lifetimes cannot return pool memory early. Range offset/size must be 4-byte aligned and map to integral F32 elements.

### CUDA/HIP graph capture/replay

`ggml_backend_cuda_graph_compute()` captures per-cgraph using stable graph identity. Tiled producer graphs therefore have to be created during Meta rebuild with fixed shape/offset and stable UID; do not create/resize tensors during replay. Phase 1 requires `n_tokens % tile_tokens == 0`; the final short ubatch or any variable tail falls back to the untouched whole graph. Communication remains outside each producer graph capture: `graph_compute_async(producer[i])` enqueues/captures the producer on the compute stream, then the provider records a ready event and launches RCCL on its separate stream.

Fail closed if a stable persistent tile graph cannot be constructed without changing graph shape/UID. Do not globally disable CUDA/HIP graphs to make the experiment work.

### Fail-closed matrix

Whole-tensor path is selected before execution unless all are true:

- `BIGCHERRY_AR_TILE_TOKENS` is 128 or 256;
- HIP/RCCL range callbacks and preflight are present; first version may require exactly 3 ranks;
- boundary terminal op is the Qwen4Exp attention-output plain `GGML_OP_MUL_MAT`, not `MUL_MAT_ID`/recurrent/GDN;
- split state is PARTIAL and every rank contributes (`GGML_TENSOR_FLAG_COMPUTE` set); no zero-sized rank special case;
- source/output are F32, tightly token-contiguous, aligned, same token count/stride on every rank;
- `n_tokens % tile_tokens == 0`; no tail graph;
- all range offsets/sizes agree across ranks and pass provider preflight;
- persistent tile cgraphs/resources were built for the current graph UID;
- no unsupported view/permutation/padding aliases the output range.

Trace each rejection once under `BIGCHERRY_AR_TILE_TRACE=1`; never silently run a partially tiled boundary.

### Smallest proof before the full patch

Add a standalone lab program first, e.g. `tools/lab/rccl/ar-overlap-tiled.hip`, using the **actual three Brutus ranks and two HIP streams per device**:

1. allocate F32 `[2560,512]` output per GPU plus persistent BF16 tile scratch;
2. compute stream launches an adjustable synthetic producer into tile i (or a representative HIP GEMM if convenient), records ready;
3. comm stream waits ready, converts tile to BF16, calls RCCL AllReduce on the tile, converts back, records done;
4. compute stream immediately launches producer tile i+1;
5. compare serial vs pipelined tile `{128,256}` with identical producer work and RCCL bytes.

Stop before package implementation unless rocprof shows real temporal overlap (`producer[i+1]` concurrent with `ncclDevKernel... [i]`) and pipeline wall beats serial. This also exposes CU contention: RCCL is a compute kernel on HIP, so theoretical stream concurrency is not proof of useful overlap.

### Tests / proof

Patch tests:

- apply + idempotence for header/Meta/provider edits;
- eligibility mapper: contiguous good cases; noncontiguous, tail, rank-shape mismatch, wrong op/type/provider all reject before execution;
- provider direct test over multiple ranges and repeated generations; event slots wrap without stale waits;
- graph replay: same request shape repeatedly plus cache reuse/multi-request; stable graph UIDs, no dynamic realloc;
- default-off output and provider call path unchanged.

Numerics: b11402's production 3-rank RCCL path is BF16 for these ranges, so do **not** demand generic byte identity with exact-F32 AllReduce. Compare tiled RCCL against whole RCCL using the same whole-reduction precision policy, including a boundary case where the total reduction is above the threshold but a tile is below it. Record max abs/ULP/logit deltas and gate model output against CPU-f32/top-k at near ties. For any F32-under-threshold direct test, require exact equality.

Mechanism acceptance: rocprof timeline proves overlap and `BIGCHERRY_AR_TILE_TRACE` shows only attention-output boundaries firing. Performance: same-build flag-off vs tile128/256 ABBA at ~32K/~100K/~200K, >=4 arms/config, clocks stable; first test isolated from 1334/other new prefill changes, then compose. Phase-1 promotion threshold: >=3% representative E2E prefill or >=30% of the attention-output reduction wall hidden with no offsetting regression. Full PGC15's existing >=10% target remains the bar before generalizing to multiple producer families. Decode control <=1%.

### 2026-10-15 source-grounded audit appendix

**Disposition:** PGC15 remains the sole owner of token-tiled prefill compute/AllReduce overlap. Do not implement the older range-callback sketch without freezing the original whole-boundary wire policy. This is a correctness defect in a proposed algorithm, **not** a reproduced BigCherry runtime defect. Current llama.cpp pin is b11552 (`23b0202a189c44a54625aadcb37a946dd1d6278d`).

### Source-grounded contract

- `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_graph_compute` (b11552 approximately 2463–2500) submits every rank's complete `cgraph_main`, then invokes a whole-tensor `comm_allreduce`. If that returns false, Meta executes `allreduce_fallback`. Splitting only this callback after the complete producer graph cannot overlap compute with communication.
- `ggml/src/ggml-cuda/ggml-cuda.cu::ggml_backend_cuda_comm_allreduce_nccl` (b11552 approximately 993–1075) chooses F32 when the **whole** tensor has fewer than 32768 elements for two ranks, 131072 for three ranks, or 262144 for four or more ranks. Otherwise it converts F32 to BF16, calls NCCL/RCCL, and converts back. These thresholds are **elements**, not bytes.
- A three-rank `[768,512]` output has 393216 elements (BF16 whole path); a 128-token tile has 98304 elements (naively F32). `[2560,512]` with 128-token tiles remains BF16. The range provider must freeze the **whole-boundary codec** during preflight, rather than rerun the provider heuristic for each tile. Even when the codec matches, changed collective message geometry may alter reduction order, so numerical equivalence still needs testing.
- Keep the existing `ggml_backend_comm_allreduce_tensor_t` ABI and provider registry. Do not introduce another dispatch table, allocator, scheduler, or telemetry system. PGC12 owns provider-completion evidence, PGC14 owns RCCL SHM transport, PGC16's redundant fold is retired, and QFP17 owns prefill acceptance.

### Proposed dataflow and lifetime

```text
Before any tile/producer submission:
  Identify a real, launch-tileable terminal attention-output GGML_OP_MUL_MAT.
  Read whole-boundary rank count, whole element count, F32 tensor layout,
    split-state/compute flags, device identity and graph UID.
  Freeze the whole-boundary F32/BF16 codec and zero/inactive-rank semantics.
  Validate ALL tile ranges, strides, alignment, rank agreement and graph lifetimes.
  Require the existing RCCL provider, preallocated BF16 scratch, per-rank
    compute/comm streams and persistent event generations.
  If any check fails: use untouched original whole graph, before side effects.

For each admitted tile:
  Launch real producer(tile) on each rank's compute stream.
  Record ready event on that rank's current HIP device.
  Comm stream waits for ready; run frozen-codec conversion/RCCL/conversion.
  Record done event on same comm stream and advance generation safely.
After final tile: compute stream waits for final comm completion on each rank.
After any producer/collective submission failure: terminal execution error;
  NEVER return false to Meta's whole-tensor fallback on mutated output.
```

The final completion event orders prior tiles only if every tile uses the same per-rank comm stream, is enqueued in order, and no event/scratch generation is reused while in flight. Verify cancellation, multi-request and graph replay. No new `MUL_MAT_ID`, recurrent, MTP or active Radiance implementation path is authorized.

### Cheapest gate and promotion

Host-only discriminator `pgc15_range_codec_host_gate.py`: Python compilation and **9/9 tests passed**. Covered 3-rank codec threshold, a tile-induced codec switch, BF16-vs-F32 numerical distinction, preflight before side effects, post-submit fallback prohibition, and theoretical overlap ceiling. No HIP build, GPU execution, repository pytest, or benchmark occurred.

Next gate: trace one real eligible `attn_output` Meta boundary with `{graph_uid, physical_rank_set, producer_op, whole_ne, tile_ne, codec, range, provider_completed, ready/done}`. Do not mistake 1277's preferred route for a completed provider. Require a static/host test that rejects `[768,512]/tile128` unless the range provider explicitly retains BF16. Reject unsupported layout, partial rank, missing streams or changing graph identities.

Only after that gate passes, use existing `tools/lab/rccl` controls to compare serial versus two-stream pipelined **identical work and codec** on the actual three-rank Brutus topology. Require rocprof evidence of `producer(i+1)` overlapping collective(i) and positive pipeline wall gain; otherwise terminal `NO_OVERLAP`, no patch. Then qualify default-off tile128/256 against the same-pin whole-tensor baseline with four independent order-balanced sessions, pp1024/2048/4096, long-context Flash-Next, multi-request/ubatch, graph replay, full logits/KLD/greedy/MTP acceptance, exact byte/work accounting, memory safety and injected failure after tile0. Require >=3% representative end-to-end prefill gain or >=30% of eligible attention-reduction wall hidden, <=1% decode regression and no correctness regression. Otherwise `REJECT_TILE_OVERLAP`.

Historical ~0.93 s compute and ~2.4 s collective service imply a **theoretical** ceiling of `min(0.93,2.4)/(0.93+2.4) ~= 27.9%` reduction of their summed service time if *all* work overlapped without overhead. Only an attention-output subset is eligible in phase 1, so its actual ceiling is lower and unknown; no measured speedup.

**Upstream/forks:** Current llama.cpp Meta retains whole-graph/whole-tensor boundaries. [llama.cpp #26827](https://github.com/ggml-org/llama.cpp/pull/26827) provides a negative test for MTP multi-ubatch/Meta stream ordering on no-P2P hardware. [vLLM capture ordering](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py) and [SGLang breakable graph #22218](https://github.com/sgl-project/sglang/pull/22218) suggest ordering/segmentation controls but are not transferable GPU implementations or AMD speedup evidence.
