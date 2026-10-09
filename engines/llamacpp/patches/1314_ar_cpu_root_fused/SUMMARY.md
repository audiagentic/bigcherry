# 1314_ar_cpu_root_fused

**Status:** evaluated
**Plan item:** QFP13/QFP11

## What it does

With `BIGCHERRY_AR_FUSED=1`, 1291's small-message cpu-root AllReduce enqueues one `bc_cpu_root_fused` kernel per rank
instead of `bc_cpu_root_produce` + `bc_cpu_root_consume`: copy the slice to pinned host memory, publish the arrive
epoch, spin on the CPU's done epoch, copy the result back, all in one launch. Buffers, epochs and the CPU fixed-order
f32 sum are unchanged, so output is identical. Removes one launch per AllReduce per GPU (~30 per Flash-Next token).

## Result (2026-10-04, flashnext-v3-1314b)

Neutral: ~24K 44.3 vs 44.3/44.2 ms/step, ~80K 51.1 vs 51.1/51.1; greedy identical. AllReduce launches 59 -> 29.5/token,
but the CPU-sum wait moves into the fused kernel spin (busy 1.2 -> 2.5 ms/token on XTX). Parked.
