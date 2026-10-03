# 1291_ar_cpu_root

**Status:** evaluated
**Plan item:** QFN01

## What it does

Adds `--allreduce cpu-root` (HIP only). Small f32 contiguous messages (<= `BIGCHERRY_AR_CPU_ROOT_MAX_BYTES`,
default 65536) use a CPU-root one-shot AllReduce: each rank's kernel copies its slice into pinned,
device-mapped host memory and publishes a per-rank epoch; a persistent CPU worker sums in fixed rank order
(exact f32) and publishes a result epoch; each rank's kernel waits for it and copies the result back, all
on the backend streams. Larger or other messages go to RCCL. Measured standalone on 2x 7900 XTX + R9700:
10 KB 13.8 us/call vs RCCL 33 us. Activation marker: `BIGCHERRY_PATCH_HIT patch=1291_ar_cpu_root path=small`.

## Hardware result (2026-10-03, flashnext-cpuroot-4)

Flash-Next UD-IQ4_XS, 2x 7900 XTX + R9700, `-sm tensor -ts 4,4,3 -ub 1024`, ABBA vs `auto` (RCCL), 3
requests per arm, greedy output identical in all 8 arms: no MTP decode 38.7/39.1 vs 36.5/36.6 t/s
(+6.4%); MTP3 (draft on the 6900) 70.5/70.4 vs 67.0/67.1 t/s (+5.1%), acceptance 74.6% vs 73.0%;
prefill unchanged (large messages stay on RCCL). Earlier builds produced garbage because ranks whose node
the meta backend left uncomputed must contribute zeros (the RCCL provider memsets them). Not yet
qualified: needs KLD and a balanced contract run.
