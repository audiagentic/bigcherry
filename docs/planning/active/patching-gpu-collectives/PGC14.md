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

## Steps

1. Identify hostcall users in RCCL device code: dump code objects (roc-obj / clang-offload-bundler) and grep metadata for hidden_hostcall_buffer per kernel; grep device sources for printf/abort/assert/__trap.
2. Rebuild RCCL for gfx1100;gfx1201;gfx1030 with -mprintf-kind=buffered and device asserts/aborts compiled out (or replaced by trap-free error flags); verify no kernel requests hostcall.
3. rccl-tests on R9700+6900 and XTX+6900, then 4-GPU: correctness (#wrong 0) and bandwidth.
4. Transport: profile SHM/host path at 21-42 MiB (rocprofv3 copies + kernel timeline); try chunk/pipeline depth changes in the SHM transport (src/transport/shm.cc) to overlap D2H/H2D; target >= 6 GB/s.
5. Ship as a BigCherry-managed RCCL build (pinned commit + patch files under a new patches area or a separate repo), selected by BC_HIP_PATH/LD_LIBRARY_PATH in lab and production lanes.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

~/rccl-heterogeneous-src/rccl (Brutus); new rccl patch set in the bigcherry repo

## Validation

rccl-tests correctness (#wrong = 0) and busbw per device pair; llama.cpp -sm tensor 4-GPU Flash-Next load + greedy parity; prefill A/B vs stock RCCL on dual XTX.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner 2026-10-01: RCCL source is available and in scope for optimisation.

## Change Log

- 2026-10-01T11:50:01.913289+00:00 (created-by): Created by agent
