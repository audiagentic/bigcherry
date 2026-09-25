# 1250_nro01_allreduce_q8_wire

**Status:** untested
**Plan item:** NRO01/NRO02

## What it does

Ports nasone e06dcf63 onto b11126: selectable AllReduce wire formats (`GGML_CUDA_AR_WIRE=f32|bf16|q8_0`), a fused residual-add finish that lets the meta backend skip the ADD after a tensor-parallel reduction (`GGML_CUDA_AR_FUSED_RESIDUAL`), and AllReduce profiling/ROCTx tracing. Everything is opt-in; unset, behaviour is the legacy BF16-threshold path.

## Why

Two-GPU `-sm tensor` decode spends measurable time moving F32 partials over PCIe and launching a separate ADD per reduction. Q8_0 wire cuts bytes ~3.8x (lossy); fusion removes a launch per layer (exact).

## Upstream

nasone commit `e06dcf6300718227cb8cfda9e61fb12ccb693418`, stacked on 1252 as in the fork.
