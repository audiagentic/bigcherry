# 1256_nro07_topk_hybrid

**Status:** untested
**Group:** nasone-rdna
**Plan item:** NRO07

## What it does

Adds disabled HIP TOP_K ordered-key and hybrid-selection policy scaffolding beside the stock bitonic path. No runtime dispatch changes yet.

## Why

Current non-CUB HIP sorts the full row even when only a small k is required. Selection-specific kernels may remove unnecessary work, but routing semantics must be fixture-proven first.

## Upstream

Local staged adaptation of nasone commit `7f3e1e4d0b166cb681b2c01503370e610a5b423d`.
