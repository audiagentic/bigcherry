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

QFP17 remains the Flash-Next prefill measurement/acceptance umbrella. PGC15 owns the backend/scheduler primitive, stream/event lifetime, eligibility proof and generic tensor-split execution. PGC14 owns RCCL transport throughput; PGC12 owns phase-aware provider routing. Do not put provider-specific SHM internals into this item.

The first target is a row-parallel output whose reduction dimension is independent across query/prompt tokens. Current execution effectively does `producer(all T) -> AllReduce(all T) -> consumer(all T)`. The desired execution is `producer(tile0) -> AR(tile0)` while `producer(tile1)` executes, with downstream use either tile-local or fenced once all tiles are reduced.

## Steps

1. Add a prefill AllReduce trace that records `{graph uid, subgraph, node name/op, shape, dtype, split axis/state, bytes, token axis/stride, producer, immediate consumers, provider, rank compute-done, AR enter/done}`. Classify every large prefill AR as tile-safe or blocked and record the blocking dependency.
2. Define strict eligibility: tensor is contiguous or has a provably contiguous token slice; every rank contributes the same logical token range; split state is tensor-partial with identical rank membership; reduction op is SUM; producer can execute on a token slice without cross-token recurrence; and no consumer observes an unreduced tile before its completion event.
3. Add an async range primitive at the Meta collective boundary. Preferred internal shape: `comm_allreduce_range(node, byte_offset, byte_count, comm_stream, done_event)` using the existing provider; do not create a second provider registry.
4. Give each participating device a dedicated communication stream and a fixed event/slot ring (start depth 3). Production of tile `i` records `produced[i]`; comm waits that event, enqueues AllReduce for the tile, then records `reduced[i]`. Compute immediately advances to tile `i+1` when dependencies allow.
5. First implementation may fence all reduced tiles before the existing whole-tensor consumer. This overlaps producer/AR without requiring arbitrary downstream graph tiling. If that wins, add tile-local consumers only for explicitly proven tokenwise chains such as residual/add/norm where doing so extends overlap.
6. Implement one producer family first, selected from the prefill trace by largest recoverable serialized AR wall. Avoid a generic graph auto-tiler initially. Candidate families are tensor-parallel dense/attention/FFN output projections with no token recurrence; GDN recurrent kernels are not eligible across their recurrence dimension.
7. Sweep token tiles `{64,128,256,512}` and pipeline depths `{2,3,4}` at outer ubatch 512/1024/2048 where memory permits. Measure whether smaller tiles increase RCCL latency enough to erase overlap. Keep an untiled threshold for small messages/tails.
8. Add a fallback path for unsupported provider/type/layout and for the final short tile. Any event/capture incompatibility disables PGC15 for that graph and executes the current whole-tensor path.
9. Compose independently with PGC14. Run four arms where practical: stock RCCL+untiled, patched RCCL+untiled, stock RCCL+tiled, patched RCCL+tiled. This distinguishes transport service-time improvement from overlap.
10. After one producer is proven, extend by measured Amdahl ranking only. Do not tile every AR mechanically; many small ARs are latency bound and belong to decode/provider work instead.

## Detailed Solution & Technical Design

### Backend state

Use fixed-lifetime stream/event state owned by the Meta backend/collective context, not graph-local allocations:

```cpp
struct bc_ar_tile_slot {
    hipEvent_t produced[GGML_META_MAX_DEVICES];
    hipEvent_t reduced [GGML_META_MAX_DEVICES];
    uint64_t generation;
};

struct bc_ar_pipeline {
    hipStream_t comm[GGML_META_MAX_DEVICES];
    bc_ar_tile_slot slots[4];
    int depth;
    size_t tile_tokens;
};
```

Adapt types/names to the existing CUDA abstraction (`cudaStream_t` aliases under HIP) rather than adding HIP-only public interfaces. Events are recorded on each rank's compute stream after the producer tile is complete. The collective comm stream waits on the local production event. Completion is made visible to the normal compute stream with `hipStreamWaitEvent` only when that tile is consumed.

Do not use host synchronization or `hipEventSynchronize` in the hot path. The design only works if the dependency graph remains device ordered.

### Range mapping

For a logical tensor laid out `[ne0, ..., token]`, prove the byte range rather than assuming tokens are adjacent. Prefer an existing contiguous view when `token_stride == bytes_per_token`; otherwise reject the first version. The collective API should receive explicit pointer/offset/count calculated from a validated view:

```cpp
struct bc_ar_range {
    size_t offset_bytes;
    size_t length_bytes;
    int64_t token_begin;
    int64_t token_count;
};
```

No rank may calculate a different byte range from the same logical tile. Assert equality when trace/debug mode is enabled.

### Execution skeleton

Conceptually:

```cpp
for (int t0 = 0, tile = 0; t0 < n_tokens; t0 += tile_tokens, ++tile) {
    const int slot = tile % depth;
    const int nt = std::min(tile_tokens, n_tokens - t0);

    wait_slot_reuse_if_needed(slot);          // stream/event ordered, no host sync
    launch_producer_tile(t0, nt, compute);
    hipEventRecord(p.slots[slot].produced[r], compute);

    hipStreamWaitEvent(comm, p.slots[slot].produced[r], 0);
    meta_allreduce_range(dst, range(t0, nt), comm);
    hipEventRecord(p.slots[slot].reduced[r], comm);
}

for (each live tile)
    hipStreamWaitEvent(compute, reduced_event, 0);
```

The real Meta backend launches all ranks; maintain the existing collective ordering across ranks. Every rank must enqueue tiles in exactly the same sequence even if one rank has no local producer work for a split node; such a rank contributes the same zero semantics as the current collective path.

### Producer integration

Do not mutate arbitrary GGML tensor shapes in-place and hope kernels interpret a shorter token dimension. Add a narrow tile-launch seam at a producer known to support row/token ranges, or build validated tensor views at graph construction with stable allocation lifetime. The first prototype can duplicate the selected producer node per tile behind an env gate if graph construction remains tractable. A later implementation may add a reusable tiled-execution recipe under PKC02 once the dependency rules are proven.

### Numerical semantics

Within a token, reduction rank order/provider semantics remain unchanged. Splitting different tokens into separate collectives should not change arithmetic because tokens are independent reduction elements. Correctness therefore should be exact for an exact provider, subject only to an existing provider's wire precision. If outputs differ, treat that as a range/layout/order bug, not an acceptable tolerance effect.

## Code Samples & Guidance

Likely llama.cpp touch points at pin 0504396:

- `ggml/src/ggml-backend-meta.cpp`: current graph/subgraph execution and `comm_allreduce` boundary; own the pipeline streams/events and tile scheduling here.
- `ggml/src/ggml-cuda/allreduce.cu` / associated header: expose an offset/count asynchronous invocation only if the current provider interface cannot already reduce a subrange pointer.
- producer backend source selected by profiling (MMQ/GEMM/output projection); keep producer-specific tiling out of the generic collective implementation.
- `patches/0830_split_reduce_telemetry` / `1277_ar_size_trace`: extend existing telemetry rather than adding a parallel logger.

Suggested diagnostic line:

```text
BIGCHERRY_AR_TILE uid=... node=... tok=0+256 bytes=... rank=1 produce_us=... ar_us=... overlap_us=... provider=rccl
```

Measure overlap as timeline intersection, not `compute+AR-wall` arithmetic. rocprof should show copy/collective work overlapping compute kernels on the same rank where hardware/provider permits it.

## Files

`ggml/src/ggml-backend-meta.cpp`; `ggml/src/ggml-cuda/allreduce.{cu,cuh}` if range dispatch requires it; one measured producer implementation; existing AR telemetry patches/tools; new BigCherry patch package after the mechanics experiment proves the seam; QFP17 records Flash-Next end-to-end evidence.

## Validation

Direct mechanics test: synthetic 2- and 3-rank tensors with deterministic per-rank data, token counts including tails `{1,63,64,65,255,256,257,1024}`, compare tiled vs whole AllReduce byte-for-byte for f32 exact provider and within the existing provider contract for bf16/RCCL.

Hardware: Brutus 2x XTX and production 2x XTX+R9700. ABBA pp1024/2048/4096 plus Flash-Next 10K/80K/200K fills. Record prefill t/s, TTFT, critical-rank wall, AR wall, producer wall, timeline overlap %, PCIe throughput, rank arrival skew and decode control.

Test graph reuse/capture on/off. Tiled events/streams must survive repeated requests, prompt cache reuse and outer ubatch tails without stale-event reuse.

## Effort & Risk

L. The main risk is not the collective call itself; it is exposing a correct producer tile before the whole tensor exists without multiplying graph/scheduler overhead. Secondary risks: RCCL small-chunk latency, stream contention, graph-capture compatibility, zero-contribution ranks and extra temporary views.

## Standards

One collective provider registry; no host synchronization in the tiled hot path; exact token-range proof; fail closed; trace before broadening; provider accuracy contract unchanged; generic backend mechanism separated from Qwen4Exp acceptance policy.

## Acceptance Criteria

- At least one high-volume prefill AR family executes as >=2 overlapping token tiles with timeline proof of simultaneous compute and collective service.
- Tiled result matches the untiled exact path byte-for-byte for f32 direct tests and passes model greedy/KLD contract under the selected production provider.
- Recover >=30% of that family's previously serialized AR wall or improve representative Flash-Next prefill >=10%; if the first family cannot meet an Amdahl-bound 3% E2E gain, park rather than generalize.
- Decode changes <=1% because PGC15 gates on prefill-sized tensors.
- No additional global/device synchronization, no unbounded event allocation, and no graph-reuse lifetime failures.

## Notes

PGC12 already identified compute/communication overlap as the main prefill collective lever; this item is the concrete implementation owner. QFP17 step 10's conditional microbatch overlap should point here for the backend mechanism. PGC14 can reduce the remaining exposed communication service time but is not a prerequisite for proving overlap.

## Change Log

- 2026-10-05T00:00:00+00:00 (created-by): Created by agent from prefill plan scan; concrete owner for token-tiled compute/AllReduce overlap.
