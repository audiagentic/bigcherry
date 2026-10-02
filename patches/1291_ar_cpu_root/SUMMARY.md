# 1291_ar_cpu_root

**Status:** untested
**Plan item:** QFN01

## What it does

Adds `--allreduce cpu-root` (HIP only). Small f32 contiguous messages (<= `BIGCHERRY_AR_CPU_ROOT_MAX_BYTES`,
default 65536) use a CPU-root one-shot AllReduce: each rank's kernel copies its slice into pinned,
device-mapped host memory and publishes a per-rank epoch; a persistent CPU worker sums in fixed rank order
(exact f32) and publishes a result epoch; each rank's kernel waits for it and copies the result back, all
on the backend streams. Larger or other messages go to RCCL. Measured standalone on 2x 7900 XTX + R9700:
10 KB 13.8 us/call vs RCCL 33 us. Activation marker: `BIGCHERRY_PATCH_HIT patch=1291_ar_cpu_root path=small`.
