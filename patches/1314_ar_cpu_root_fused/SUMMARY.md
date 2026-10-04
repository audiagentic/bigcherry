# 1314_ar_cpu_root_fused

**Status:** untested
**Plan item:** QFP13 (QFP11)

## What it does

With `BIGCHERRY_AR_FUSED=1`, 1291's small-message cpu-root AllReduce enqueues one `bc_cpu_root_fused` kernel per rank
instead of `bc_cpu_root_produce` + `bc_cpu_root_consume`: copy the slice to pinned host memory, publish the arrive
epoch, spin on the CPU's done epoch, copy the result back, all in one launch. Buffers, epochs and the CPU fixed-order
f32 sum are unchanged, so output is identical. Removes one launch per AllReduce per GPU (~30 per Flash-Next token).
