# 1269_prbe55_vk_smalln_dmmv

**Status:** untested
**Plan item:** PRBE55

## What it does

Adds an explicit `BIGCHERRY_VK_SMALLN_DMMV=1` experiment that forces otherwise-MMVQ-eligible AMD/RDNA Vulkan vector matmuls with N=2..8 onto the existing DMMV fallback path.

## Safety

Default routing is unchanged. N=1, N>8, non-AMD, GCN, and calls the baseline selector already rejected for MMVQ are untouched.

## Validation

MTP/speculative verification supplies the positive small-N workload. Correctness is greedy token identity plus backend-op boundary cases; performance is batched MTP server throughput with ordinary decode as control.
