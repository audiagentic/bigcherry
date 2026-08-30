# 1209_rd22_integrated_gpu_host_buffer_backout: Back out integrated-GPU host buffers on HIP (RD22, fork divergence from PR #24233)

**Status:** untested
**Group:** rdna-boosts
**Plan item:** RD22

## What it does

Forces integrated=false inside ggml_cuda_init()'s GGML_USE_HIP branch, backing out upstream PR #24233's restoration of prop.integrated on HIP, which enabled a zero-copy UMA host-buffer path on APUs.

## Why

On the fork author's Strix Halo iGPU, the host-buffer path corrupts full-model results under async execution (PPL 5.9243 -> 8.51+ without HIP_LAUNCH_BLOCKING); forcing integrated=false restores async-safe operation at no discrete-GPU cost, since discrete GPUs never select that path anyway.

## Upstream / provenance

Ported verbatim from stew675-rdna-boosts fork commit 507f2e267 (https://github.com/stew675/llama.cpp), a deliberate divergence away from mainline PR #24233 -- a pin bump does not absorb this, it would re-enable the broken path.
