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

## Hardware result and decode fix (Brutus 2026-10-08)

Flash-Next, 245760 ctx, 79722-token fill, auxiliary routed-expert layers on ROCm3 (6900 XT):

```text
ub512:  L0 1117.7 t/s (decode 65.3 t/s, accepted 336/525; VRAM GiB 22.7 23.1 31.1 4.1)
        L2 1058.5 (22.3 22.6 29.7 6.4) | L4 1001.8 (21.8 22.1 28.4 8.7) | L6 964.1 (21.3 21.6 26.8 11.3)
ub1024: L0 OOM (cudaMalloc 100 MiB on device 1) | L2 1158.8 (23.4 23.7 30.5 7.1) | L4 1136.7 | L6 1086.6
ub2048: all arms OOM (3958.87 MiB on device 0), including L0
```

At equal ubatch the offload costs about 5% prefill per two layers, but L2 frees enough VRAM for ub1024 and reaches 1158.8 t/s (+3.7% versus the ub512 L0 baseline). Every L2/L4/L6 arm then stopped at the first generation step while L0 decoded normally.

The source-level defect was context scope: 1328 registered the ordinary auxiliary backend and enabled its Qwen4Exp layer semantics for every context sharing the target model, including `LLAMA_CONTEXT_TYPE_MTP`. Qwen4Exp's MTP context owns the MTP block alone, so trunk-layer auxiliary placement must not alter that scheduler topology. 1328 now registers/uses the auxiliary backend only for `LLAMA_CONTEXT_TYPE_DEFAULT`. `BIGCHERRY_PATCH_TRACE` also emits `phase=aux_merge layer=<n> tokens=<n> ctx_type=<n>` immediately before each marked target merge.

Hardware decode rerun pending; state remains untested until the L2+ arm completes generation.

## Upstream

Local BigCherry feature against llama.cpp pin b11474 (`b9acf138a1e2`).
