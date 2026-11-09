---
id: RAD50
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-08T21:44:12+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Prove CPU BAR-push MoE result transport in isolated, gated harnesses

## Description

RAD42 design B proposes CPU-writing the CPU-combined remote-MoE FP16 result
plane into engine VRAM through a CPU-accessible BAR, then letting the GPU
consume locally instead of pulling the result plane from host RAM. RAD40's
host-link measurements motivate this under XTX0 host-read contention, but
the standalone CPU BAR push rate is *lower* than an uncontended GPU pull.
The advantage is not established for inference. This item owns the
**experimental feasibility and safety gates only**, before any Radiance
or moe-remote runtime modification. RAD42 remains the owning production
design item; RAD49 is a future TP2/consumer-rank integration concern.

## Steps

1. Card-free model: generation-scoped payload ownership and distinct
   GPU-produced, CPU-write-completed, GPU-consumed states. Negative cases
   for stale epochs, early publication, early recycle and faults.
2. Hardware discovery: map PCI identity to the engine GPU and enumerate
   GPU-local allocations whose CPU access the HSA runtime permits.
   An allocatable pool is not proof of a functional BAR mapping.
3. Single-process CPU-write -> GPU-read canary on 4 KiB then 4 MiB,
   with fully checked payload, repeated generation patterns, no
   streaming and no interprocess sharing.
4. Before bandwidth tests, validate GPU-local memory sharing across
   a **separate CPU-only helper process** via HSA IPC and confirm that
   the importer can write without enabling GPU peer access. Serial
   verification is only an IPC feasibility gate, not device-live ordering.
5. Measure CPU BAR push and mapped-host GPU pull on the same destination
   and payload, across sizes, thread counts and contention arms.
   Record CPU utilisation, p50/p95 latency, time to first chunk and
   final completion, not bandwidth alone.
6. Prove per-chunk publication and visibility in a bounded, separate
   producer/consumer harness. Use distinct ready and consumed generations;
   timeout and fail closed. Run the 128/256/512/1024-row matrix.
7. Reproduce the real CPU-combine FP16 semantics (overflow sentinel
   fallback and consumer-rank/once-only ownership) on representative
   captured rows, without a serving engine.
8. Only if all gates pass, propose a new, separately reviewed
   opt-in moe-remote transport branch. No production edits,
   default change, push or PR are authorised by this item.

## Detailed Solution & Technical Design

The engine GPU may continue issuing work and request tickets. A helper's
GPU produces output and writes host-visible result planes. The existing
CPU combiner may assemble a complete chunk. The proposed *bulk* transfer
is a CPU write to GPU-local memory accessible via the BAR, followed by
a tested visibility action and a separate ready generation that the GPU
checks before consuming. Request publication is **not** evidence that
the CPU's BAR writes have completed. The consumer must acknowledge before
slot reuse. Neither a CPU fence nor an HSA pool-access flag alone proves
cross-agent visibility or ordering; the hardware canary and chunk
protocol must establish those facts.

Retain the existing mapped-host GPU-read transport as the control and
fallback. Preserve FP16 combine numerical order and overflow fallback.
Do not assume direct GPU-to-GPU peer access on this host. The test must
distinguish engine ranks from helper slots as RAD49 evolves. Never
substitute a CPU-round-tripped transfer for the GPU-driven decode path
without measuring its added wakeup and dispatch latency.

**Stop/go:** Stop on any stale-read, data mismatch, unsupported CPU mapping,
unbounded wait or GPU fault. For production consideration, first require
at least 15% improvement in contended transfer makespan or at least
0.3 ms reduction of the layer-step critical path, then at least 2%
end-to-end improvement and no p99 regression over 5% with quality parity.

## Code Samples & Guidance

The isolated harness and gate log live in
`tools/lab/rad42-bar-push/README.md`. The single-process canary may
use HIP kernels and HSA GPU-local memory allocations, but the destination
must be proven to reside in GPU memory, be CPU-addressable, and match
the selected GPU's PCI identity. Every failure should yield a nonzero
exit status with a reported reason. The actual cross-process export,
mapping, lifetime and visibility method is deliberately deferred until
the single-process gate passes.

## Files

- `tools/lab/rad42-bar-push/chunk_contract.py`
- `tools/lab/rad42-bar-push/test_chunk_contract.py`
- `tools/lab/rad42-bar-push/pool_inventory.cpp`
- `tools/lab/rad42-bar-push/bar_write_canary.hip` (Gate 2)
- `tools/lab/rad42-bar-push/critical_path_model.py` (pure scenario model)
- `tools/lab/rad42-bar-push/cpu_handoff_bench.cpp` (host-only lower bound)
- `tools/lab/rad42-bar-push/bar_ipc_canary.hip` (Gate 2b, no peer GPU)
- `tools/lab/rad42-bar-push/bar_transfer_ab.hip` (Gate 3, not run)
- `tools/lab/rad42-bar-push/bar_live_ready_canary.hip` (Gate 4 single-process canary, not run)
- `tools/lab/rad42-bar-push/combine_reference_test.cpp` (reference parity)
- `tools/lab/rad42-bar-push/PROGRESS.md`, `RUNBOOK.md` and `FINDINGS.md`
- `tools/lab/rad42-bar-push/README.md` (per-gate progress and decisions)
- Subsequent isolated lab harnesses only after the preceding gate passes.

## Validation

Card-free unit tests for lifecycle ordering and fault conditions;
`g++` and `hipcc` compile-only checks in the experimental worktree.
Hardware tests must use an exclusive GPU lease and identify the physical
GPU, ROCm release, source commit, payload size, pass/fail and repeat count.
Record both an uncontended control and the RAD40 XTX0-contention arm.
Benchmark comparisons must keep the same payload, CPU pressure and GPU
link conditions and report critical-path makespan, not just aggregate GB/s.
A wrong result, hang, fault or consumer-gate violation is a failure.

## Effort & Risk

M experimental scope, advanced synchronization/hardware semantics.
Risks: BAR visibility and coherency, CPU/GPU pointer accessibility,
cross-process GPU memory export, VRAM buffer lifetime, CPU-copy overhead,
PCIe posted-write ordering, TP2 consumer-rank identity and disabled
peer-to-peer transport. Keep serving untouched while uncertain.

## Standards

Use project-local experimental worktree, one experiment directory under
`tools/lab/`, documented run identity, and explicit retention or
retirement disposition. Respect GPU leases, agent-owned branches,
the pinned vendor tree and card-free CI. No hardware run without a
lease; do not bypass unavailable lease controls.

## Acceptance Criteria

1. Contract and GPU-accessibility gates explicitly passed or failed,
   with reproducible commands and evidence.
2. CPU-written GPU-local data verified exactly by a GPU on required
   payload sizes and repeated generations; negative failure behavior
   and device identity documented.
3. A separate push/pull and contention comparison with p50/p95 timing,
   CPU cost, chunk latency and hardware identity.
4. Per-chunk consumer safety and faults tested, no stale or duplicate
   consumption, no buffer reuse before acknowledgment.
5. MoE FP16/fallback and TP2 once-only equivalence verified against
   the existing reference on captured rows.
6. Any proposed serving integration gated on measured end-to-end
   benefit and explicit subsequent review; item remains pending until
   all acceptance criteria are met.

## Notes

Related: RAD40 (link measurements), RAD42 (production design B),
RAD51 (future production integration plan, blocked on RAD50),
RAD49 (TP2 engine and remote experts). Start from the existing
`exp/rad42-cpu-bar-push` lab worktree, not production sources.
Do not claim CPU BAR push is faster without the complete comparison.

## Change Log

- 2026-11-08T21:44:12+00:00 (created-by): Created as an isolated
  proof-first experiment item; initial Gate 0 passed 10 card-free tests.
- 2026-11-08T22:01:00+00:00 (updated-by): Gate 0 extended to 13 card-free tests;
  inventory and single-process GPU canary compiled for two GPU architectures.
  Lease command is now available, but RAD49 holds the exclusive GPU lease.
  Hardware Gates 1 and 2 remain unrun; no performance or coherence claims.
- 2026-11-08T22:27:00+00:00 (updated-by): Added model of already-overlapped CPU combine and GPU pull, 10 more card-free tests, host-only wakeup probe and GPU canary/IPC/transfer microbenchmark binaries. 23 Python tests pass; no GPU run. Actual serving integration is described in new pending RAD51, blocked by these gates.

- 2026-11-08T23:50:00+00:00 (updated-by): Checked all four GPUs idle, acquired the exclusive GPU lease and inspected ROCm devices/pools. The remote command policy rejected running the separately compiled `pool_inventory` executable. No BAR or GPU canary ran. Released the lease promptly. Gate 1 remains unverified; explicit executable allowlisting is required.

- 2026-11-08T23:55:00+00:00 (updated-by): Hardware Gate 1 API discovery PASS under exclusive RAD50 lease after fixing non-global HSA pool enumeration. XTX0/XTX1: two CPU-accessible GPU-local pools each; R9700: one; RX 6900 XT: two. Gate 2 blocked by missing `bar_write_canary-gfx1100` and `bar_write_canary-gfx1201` executable permissions; no canary ran. Lease released. CPU BAR writes, GPU correctness, IPC and latency remain unverified.

- 2026-11-09 (updated-by): Gate 2 serial CPU BAR/GPU read **passed on both RX 7900 XTX cards**: 4 KiB and 4 MiB each, 12 generations each, every word checked, exits 0. R9700 `gfx1201` canary exited 1 without output, leaving its correctness unresolved. Added lab-only early diagnostic markers and recompiled R9700 probe without retesting. Exclusive GPU lease released. Cross-process mapping, live ready signalling and speed are still unverified.

- 2026-11-09 (updated-by): R9700 gfx1201 exit-1 narrowed to the first HIP GPU-verification kernel launch; HSA VRAM CPU allocation/stores and HIP error-counter memset completed, but launch did not return. No GPU result was read and the failure mechanism is unresolved. Compiled a HIP-only smoke diagnostic without rerunning it. XTX IPC executable remains disallowed, and subsequent GPU lease was obtained by HD2701. RAD50 did not retain a lease or run further GPU tests.
