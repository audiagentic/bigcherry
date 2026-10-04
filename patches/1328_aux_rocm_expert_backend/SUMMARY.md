# 1328_aux_rocm_expert_backend

**Status:** untested
**Plan item:** MET05

## What it does

Adds an opt-in `BIGCHERRY_EXPERT_AUX_DEVICE=<device>` path for Qwen4Exp tensor-split target contexts. The named GPU is registered as an ordinary scheduler backend without being added to the Meta tensor group. Routed-expert tensors placed there with `-ot` execute there, with Meta/aux activations copied through a scheduler-owned pinned host buffer.

For a fully auxiliary routed-expert layer, the routed MoE result is treated as a full mirrored value. Meta reduces the tensor-partial shared-expert branch before the routed/shared add, then adds the mirrored auxiliary result without AllReducing it.

## Safety

Off by default. The path fails closed if the auxiliary device is absent, ambiguous, non-GPU, already in the Meta group, if a selected routed-expert layer is only partially on the auxiliary device, if router/shared-expert tensors are moved there, or if a Meta/aux crossing is not MIRRORED.

## Why

`-ot ...=ROCm3` can allocate target weights in an ordinary ROCm3 buffer, but the target scheduler previously knew only Meta + CPU and aborted during graph reserve. This package makes ROCm3 schedulable without adding it to RCCL/cpu-root AllReduce membership and preserves exact routed/shared reduction semantics.

## Upstream

Local BigCherry feature against llama.cpp pin `0504396140d1c882f5f6ee34466a42db7ae90114`.
