# 1328_aux_rocm_expert_backend

**Status:** untested
**Plan item:** MET05

## What it does

Adds an opt-in `BIGCHERRY_EXPERT_AUX_DEVICE=<device>` path for Qwen4Exp tensor-split target contexts. The named GPU is registered as an ordinary scheduler backend without being added to the Meta tensor group. Routed-expert tensors placed there with `-ot` execute there, with Meta/aux activations copied through a scheduler-owned pinned host buffer.

For a fully auxiliary routed-expert layer, the routed MoE result is treated as a full mirrored value. If the shared-expert branch is still PARTIAL at the merge, Meta reduces it once; if it is already MIRRORED, both inputs are complete replicas and the merge is a plain per-device add. The auxiliary result is never AllReduced.

## Safety

Off by default. The path fails closed if the auxiliary device is absent, ambiguous, non-GPU, already in the Meta group, if a selected routed-expert layer is only partially on the auxiliary device, if router/shared-expert tensors are moved there, or if a Meta/aux crossing is not MIRRORED.

## Why

`-ot ...=ROCm3` can allocate target weights in an ordinary ROCm3 buffer, but the target scheduler previously knew only Meta + CPU and aborted during graph reserve. This package makes ROCm3 schedulable without adding it to RCCL/cpu-root AllReduce membership and preserves exact routed/shared reduction semantics.

## b11474 diagnosis

The first hardware sweep after the enum-tag compile fix aborted at load for every aux-offload layout because the marked merge observed `MIRRORED + MIRRORED`, while 1328 only accepted `MIRRORED + PARTIAL`. At b11474, Meta's split-state contract uses `MIRRORED` for a complete replica; its synchronized matmul state can therefore make the shared branch MIRRORED before buffer initialization. In that state an AllReduce would incorrectly sum duplicate complete values. 1328 now accepts both `MIRRORED + PARTIAL` and `MIRRORED + MIRRORED`, returning MIRRORED in either case, and still aborts on every other marked combination.

Hardware rerun pending.

## Upstream

Local BigCherry feature against llama.cpp pin b11474 (`b9acf138a1e2`).
