# 1276_ar_adaptive_nway

**Status:** untested
**Plan item:** PGC10

## What it does

Composes 0840 adaptive AllReduce with 1244's N=3 root path: reductions below
`--allreduce-switch-bytes` use the internal root3 provider and reductions at
or above it use RCCL. `BIGCHERRY_AR_ROOT3_ROOT=0|1|2` selects the root rank at
pipeline initialization; the default is rank 0.

## Why

1244 already resolves pinned-host aliases for every consumer/owner pair. This
patch generalizes only the logical root/leaf indices so the validated root3
protocol can choose any of the three ranks without changing its transport or
synchronization scheme, while preserving 0840's adaptive crossover.

## Upstream

Local BigCherry composition over llama.cpp b11233. State remains untested;
this patch makes no new hardware-performance claim until N=3 full-stack
validation is recorded.

## 2026-10-09 disposition

1276 adds **root-rank selection and N=3 composition only**; it does not supply a new default provider, phase-aware routing, or a replacement for validated 1291 CPU-root. PGC10 owns the conditional comparison; QFP01 owns 1291; GP11/1244 owns root3 kernels. No current-pin, same-model RCCL/CPU-root/root3 qualification is recorded. Root3's F32-only internal size gate and 0840's 96 KiB logical-size gate are separate. Keep state `untested`, avoid production promotion and do not reuse historical RCCL-only comparisons as evidence against CPU-root. Close if no measured CPU-root bottleneck or if a bounded A/B fails PGC10's gate.
