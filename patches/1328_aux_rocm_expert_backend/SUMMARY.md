# 1328_aux_rocm_expert_backend

**Status:** rejected
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

## Rejected (2026-10-08, pin b11474, Brutus)

Six hardware rounds after the b11474 re-base. With routed-expert layers on the auxiliary 6900 XT (ROCm3) the patch
never produced a generation result:

1. compile error (`ggml_backend_dev_type` enum tag);
2. abort at load (`marked expert merge requires exactly MIRRORED + PARTIAL sources, got MIRRORED + MIRRORED`);
3. loads and prefills, generation request never returns;
4. diagnostic run: the server process is gone 60 s after prefill, no error, assert or backtrace in the log;
5. and 6. (d9370c50, a23c77dc): do not compile (`ggml/src/ggml-backend.cpp:1104:27: error: expected expression`).

Prefill, 79722-token fill, 245760 context (t/s): no offload ub512 1087-1118; two layers offloaded ub512 1059,
four 1002-1004, six 964; no offload ub1024 does not fit; two layers offloaded ub1024 1159-1178, four 1132-1137,
six 1087. So offload costs about 5% prefill per two layers at equal ubatch; the only benefit was fitting ubatch 1024.

Experiment `deploy-v5-plus-1327-aux6900` and `tools/lab/flash-next/queue-expert-aux6900.sh` removed.
