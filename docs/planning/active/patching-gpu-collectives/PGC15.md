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
