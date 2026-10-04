---
id: PGC14
order: 0
plan: patching-gpu-collectives
state: pending
created-at: '2026-10-01T11:50:01.913289+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
---

# RCCL source patches: hostcall-free kernels (chipset-attached 6900 XT) and host-staged transport throughput

## Description

RCCL source is on Brutus (~/rccl-heterogeneous-src/rccl, ROCm/rccl at 57e58688; install, install-ndebug and rccl-tests builds). Two RCCL-level problems found 2026-10-01:
(1) The RX 6900 XT (gfx1030) is on an Intel chipset root port (00:1d.0, PCIe 4.0 x16) without PCIe atomics. Every RCCL collective involving it fails at the first kernel launch: HIP log 'Pcie atomics not enabled, hostcall not supported' -> 'AQL dispatch failed' -> 'the operation cannot be performed in the present state'. Reproduced with rccl-tests (ROCm 7.2.4 RCCL, the heterogeneous build, and its ndebug build), independent of NCCL_PROTO, SHM/P2P, MSCCL and HSA_FORCE_FINE_GRAIN_PCIE. gfx1030 is in RCCL's default GPU_TARGETS, so kernels exist; they request the hostcall buffer (device printf/abort/assert paths).
(2) Host-staged AllReduce is slow: rccl-tests on 2x 7900 XTX (PCIe 4.0 x8, NCCL_P2P_DISABLE=1, Simple) reaches ~1.1 GB/s at 21-42 MiB for every NCCL_BUFFSIZE 1-16 MiB, versus ~6.5 GB/s serial and ~13 GB/s pipelined host-stage ceilings; prefill AR dominates prefill time (PGC12).

The prefill transport target is not another CPU-root whole-buffer AllReduce. Large-message CPU-root has already lost in-model because it serializes the host round trip. The RCCL work here must keep the collective stream ordered while converting the SHM/host transport from a whole-buffer or shallow pipeline into a bounded multi-slot transfer pipeline that overlaps ingress, host-visible readiness, reduction/relay, and egress.

## Steps

1. Identify hostcall users in RCCL device code: dump code objects (roc-obj / clang-offload-bundler) and grep metadata for hidden_hostcall_buffer per kernel; grep device sources for printf/abort/assert/__trap.
2. Rebuild RCCL for gfx1100;gfx1201;gfx1030 with -mprintf-kind=buffered and device asserts/aborts compiled out (or replaced by trap-free error flags); verify no kernel requests hostcall.
3. rccl-tests on R9700+6900 and XTX+6900, then 4-GPU: correctness (#wrong 0) and bandwidth.
4. Transport baseline: profile SHM/host path at 1, 4, 10, 21 and 42 MiB with rocprofv3 copies + kernel timeline. Record D2H and H2D copy-engine occupancy, host wait intervals, chunk size, protocol, channel count, per-channel FIFO occupancy and whether a rank is waiting for a whole buffer when a prefix could progress.
5. Implement a fixed-depth host-stage ring in RCCL SHM transport. Start with 4 slots and 512 KiB/1 MiB/2 MiB chunks. Each slot has independent producer and consumer sequence numbers; a producer may fill slot `s+1` while slot `s` is being reduced/forwarded and slot `s-1` is returning to the GPU. Avoid a host wakeup or global barrier per slot.
6. Preserve stream ordering at collective boundaries: the user stream sees one collective completion, but internal transport workers/copy queues may pipeline chunks. No destination chunk may become reusable before every required source rank for that chunk has published the matching generation.
7. Sweep pipeline depth {2,4,8}, chunk {256 KiB,512 KiB,1 MiB,2 MiB,4 MiB}, Simple protocol/channel counts and rank ordering. Use p50/p90 bandwidth, not the single best iteration. Reject any configuration that moves the bottleneck to CPU polling or saturates the x4 R9700 link while leaving XTX copy engines idle.
8. Run the patched RCCL under BigCherry `-sm tensor` at pp1024/2048/4096 and the Flash-Next production lane. Measure critical-path collective wall and prefill t/s, then compose with PGC15 token-tiled compute/communication overlap; transport and overlap are separate levers and must be measured independently first.
9. Ship as a BigCherry-managed RCCL build (pinned commit + patch files under a new patches area or a separate repo), selected by BC_HIP_PATH/LD_LIBRARY_PATH in lab and production lanes.

## Detailed Solution & Technical Design

### Host-stage ring

Model each logical channel as a ring of `N` host-pinned slots. A slot is reusable only after its generation has completed the full ingress -> combine/relay -> egress lifecycle. Sequence numbers must be monotonic; do not reset boolean ready flags and race a late consumer from the previous generation.

Conceptual state:

```cpp
struct bc_shm_slot {
    alignas(64) std::atomic<uint64_t> src_ready[MAX_RANKS];
    alignas(64) std::atomic<uint64_t> result_ready;
    alignas(64) std::atomic<uint64_t> consumed[MAX_RANKS];
    void * host_payload[MAX_RANKS];
    void * host_result;
};

struct bc_shm_pipe {
    bc_shm_slot slot[DEPTH];
    size_t chunk_bytes;
    uint64_t generation;
};
```

This is a design sketch, not a requirement to use C++ atomics inside RCCL. Reuse RCCL's existing FIFO/step counters where possible. The invariant is the important part: `(slot, generation)` uniquely identifies ownership, and publication occurs only after the preceding DMA/write is globally visible.

For each chunk `c`:

```text
GPU D2H(c) -> publish src_ready(c)
                         | while D2H(c+1)
                         v
                reduce/relay host chunk c
                         | while reduce(c+1), D2H(c+2)
                         v
                 H2D/result(c) -> consumed(c)
```

Prefer copy-engine DMA for large payloads; compute kernels must not spin on host memory for multi-MiB prefill transfers. If the existing SHM algorithm already has chunk/FIFO steps, first deepen and decouple those steps rather than introducing a parallel transport abstraction.

### CPU work

Do not assume CPU summation is the bottleneck. Previous BigCherry large-message CPU-root experiments found that adding SIMD sum workers did not recover throughput. Instrument bytes/s and CPU cycles in the combine stage; only optimize the reduction loop if it occupies the critical slot lifetime. NUMA-pin any helper thread and pinned host allocation to the CPU root complex serving the target GPUs where topology allows.

### Interaction with PGC15

PGC14 improves the service time of a large collective. PGC15 hides collective service behind production of the next token tile. Keep the APIs independent: PGC15 should see a normal asynchronous collective on a communication stream; it must not depend on private RCCL slot state. This lets either patch be disabled and keeps provider comparisons meaningful.

## Code Samples & Guidance

Primary RCCL inspection points:

```text
src/transport/shm.cc            SHM setup/connect/proxy path
src/proxy.cc / transport proxy  progress loop and FIFO steps (exact current path: verify at 57e58688)
src/device/*                    protocol step sizes / device-side publication
```

Add lightweight counters behind a build/runtime diagnostic gate: bytes submitted/completed per copy direction, slot occupancy high-water, producer wait cycles, consumer wait cycles and per-collective pipeline depth reached. Do not printf from device kernels on gfx1030; that is part of the hostcall failure this item is fixing.

For the performance experiment, preserve the same RCCL algorithm/protocol while changing one of `{chunk_bytes, depth}` at a time before cross-product tuning. A throughput increase from more channels is not evidence that the ring itself works unless the timeline shows overlapping D2H/H2D generations.

## Files

~/rccl-heterogeneous-src/rccl (Brutus); RCCL `src/transport/shm.cc` plus the exact proxy/FIFO files reached by the profiled path; new rccl patch set in the bigcherry repo; `tools/lab/rccl/` microbench scripts/results; PGC15 for graph-level overlap composition.

## Validation

rccl-tests correctness (#wrong = 0) and busbw per device pair; llama.cpp -sm tensor 4-GPU Flash-Next load + greedy parity; prefill A/B vs stock RCCL on dual XTX and 3-rank production topology.

Transport qualification matrix: 1/4/10/21/42 MiB; 2x XTX and XTX+R9700/3-rank where supported; >=100 warmed collectives per point; p50/p90 latency and effective bus bandwidth. Timeline must show simultaneous generations in different pipeline phases rather than merely smaller serialized copies.

## Effort & Risk

L. Main risks are RCCL proxy/FIFO correctness, sequence wrap/reuse bugs, extra host polling, NUMA placement, and improving standalone bandwidth without reducing in-model wall time. Keep the stock RCCL build selectable for every experiment.

## Standards

Pinned RCCL source/patch provenance; fail closed to stock RCCL on unsupported topology; no device printf/assert dependency on the gfx1030 lane; correctness before throughput; microbenchmark and in-model evidence both required.

## Acceptance Criteria

- gfx1030-involving RCCL kernels launch without a hidden hostcall buffer and pass rccl-tests correctness.
- No regression >2% for <=1 MiB collectives relative to the stock build.
- Host-staged 21-42 MiB throughput reaches >=4 GB/s first-stage acceptance and targets >=6 GB/s on dual XTX, with timeline-proven D2H/H2D overlap.
- Flash-Next prefill improves >=5% or collective critical-path wall falls >=20% on a representative long-prefill lane without decode regression >1%.
- Patched RCCL remains independently selectable so PGC15 and provider-routing experiments can isolate transport effects.

## Notes

Owner 2026-10-01: RCCL source is available and in scope for optimisation.

2026-10-05 prefill review: PGC14 is the existing owner for PREF03/RCCL SHM pipelining. Do not create a second BigCherry host-AllReduce implementation for large messages; deepen this transport item instead.

## Change Log

- 2026-10-01T11:50:01.913289+00:00 (created-by): Created by agent
- 2026-10-05 (updated-by): Deepened host-staged prefill pipeline design, code guidance, instrumentation and acceptance gates.
