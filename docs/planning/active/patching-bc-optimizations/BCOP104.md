---
id: BCOP104
order: 104
plan: patching-bc-optimizations
state: done
created-at: '2026-10-10T01:10:30+00:00'
created-by: agent
priority: P1
work: S
---

# PGC14: fail closed on RCCL SHM CE hangs

## Discovery and disposition

The independent 2026-10-02 commit `9ded4c6e84b1` reported CE mode hanging `llama-bench`. The sweep's timeout/invalid CSV did not fail its overall status. The PGC14 illustrative coalescing sketch also allowed a short first FIFO slot followed by a full slot and copied an incorrect contiguous extent. The sweep now records and propagates per-arm failures; the unsafe sketch was withdrawn. No new hardware performance is claimed.

## Ownership and recent work

PGC14 owns SHM CE admission and transport tuning; PGC15 owns token-tiled AR, PGC12 owns provider/phase evidence, and gfx1030 hostcall is a separate lane. Preserve RCCL's existing `NCCL_STEPS` ring. Active Radiance, Flash-Next 1330/1334/1347/1357, QFP43 DFlash and QFP35/QFP41 promotion work were excluded. No new queue, transport, scheduler, cache or allocator.

## Gate / terminal outcome

With Brutus host/GPU lane idle, test direct SHM and CE modes 1/2/3 in separate processes, `NCCL_PROTO=Simple`, forced RCCL, actual SHM/provider/protocol receipts, finite timeout and exact collective correctness. Reject any hanging/incorrect mode, retain direct fallback; close CE tuning if no matched pp2048/4096 or collective-wall improvement. Coalescing requires proven safe CE plus host FIFO extent, ring and tail tests. ROCm/rccl #2187 is a gfx12 correctness consideration, not a measured gain.

## Evidence

RCCL `57e58688/src/include/param.h` and `src/transport/shm.cc`; BigCherry `tools/lab/rccl/rccl-env-sweep.sh`, commit `9ded4c6e84b1`; PGC14/PGC12/PGC15; ROCm/rccl #2187, vLLM custom AR, SGLang PCIe-IPC. Six deterministic host assertions and Bash syntax plus reduced fake-benchmark success/failure tests passed. No repository pytest, RCCL build, HIP execution or GPU benchmark.
