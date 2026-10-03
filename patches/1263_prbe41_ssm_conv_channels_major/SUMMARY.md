# 1263_prbe41_ssm_conv_channels_major

**Status:** untested
**Plan item:** PRBE41

## What it does

Adds a channels-major input mode to `ggml_ssm_conv` (CPU and CUDA/HIP) and makes the Qwen3.5/3.6 delta-net graph use it, so the conv input is no longer physically transposed before the SSM convolution.

## Why

The transpose (a CONT kernel) cost ~221 ms per 4096-token prefill ubatch in the fork's measurement; channels-major input avoids it.

## Upstream

Port of nasone commit `33611a98a53af8a327f4a0e01a42631eb5fcd576` (AMD). Metal support rejection not ported; the recurrent conv-state layout changes (saved states are not interchangeable across the patch).
