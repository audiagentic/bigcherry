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
