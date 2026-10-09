# 1344_dsv4_hc_grid_index

**Status:** validated
**Plan item:** QFP35

## What it does

The Qwen4Exp hyper-connection PRE and POST kernels (`ggml-cuda/dsv4-hc.cu`) take their coordinates from the launch
grid: PRE runs on a 2-D grid (embedding index in blocks of 256, token), POST on a 3-D grid (embedding index,
destination stream, token). On by default; `BIGCHERRY_HC_GRID_INDEX=0` restores the flat kernels.

## Why

The flat kernels run on a 1-D grid and every thread recovers its coordinates with 64-bit `%` and `/` by run-time
dimensions (two in PRE, three in POST). AMD GPUs have no 64-bit integer divide, so each is an emulated loop executed
once per output element, in kernels that otherwise do a few multiply-adds. An external report on the same model family
measured 230 -> 79 us per call from this change; that number is a hypothesis here until measured.

## Scope

Indexing only. Each thread computes the expression the flat kernel computes for the same element, so the result is
bit-identical. The Q8_1 PRE path of 1311 is not touched. A grid dimension is limited to 65535; a larger batch uses
the flat kernels.

## Activation

`BIGCHERRY_PATCH_TRACE=1` prints `BIGCHERRY_PATCH_HIT patch=1344_dsv4_hc_grid_index path=pre|post` once each.
