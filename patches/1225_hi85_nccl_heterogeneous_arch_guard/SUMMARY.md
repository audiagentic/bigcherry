# 1225_hi85_nccl_heterogeneous_arch_guard: Fail closed when NCCL/RCCL participants span different GPU architectures (HI85)

**Status:** untested
**Group:** core
**Plan item:** HI85

## What it does

Adds a guard before ncclCommInitAll that compares every participating device's compute-capability architecture identifier and aborts with GGML_ABORT (naming the mismatched architectures and pointing at HI85) when they differ, rather than letting the collective launch crash uncatchably later.

## Why

Real dual/multi-GPU testing on Brutus found RCCL's collective kernel dispatch does not correctly handle a communicator whose ranks span different GPU architectures -- comm init succeeds, but the actual collective kernel launch inside ncclGroupEnd() hard-aborts with an uncatchable HIP SIGABRT. Per explicit user direction, silently degrading to META instead is not acceptable; this fails loudly and diagnosably instead.

## Upstream / provenance

Local design, based on real-hardware findings (docs/planning/active/hip-autotune/HI85.md), reproduced against stock llama.cpp too. Message wording later corrected (HI138/HI142) after finding the true root cause is PCIe atomics capability, not architecture per se, though the architecture-mismatch trigger itself is unchanged.
