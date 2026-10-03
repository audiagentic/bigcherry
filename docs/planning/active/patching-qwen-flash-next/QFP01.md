---
id: QFP01
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:13.613565+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# 1291 CPU-root one-shot AllReduce for small decode messages (3-GPU, no P2P)

## Description

Patch 1291_ar_cpu_root (evaluated, in production profile): --allreduce cpu-root. f32 messages <= 64 KiB use pinned mapped host slots, per-rank device-advanced epochs and a persistent exact-f32 AVX2 CPU sum worker; GPUs spin on a result epoch; RCCL above 64 KiB. Ranks whose node the meta backend skipped contribute zeros. Graph-capture safe on the small path. Large-message host path exists but is opt-in/off: lost to RCCL in-model at every chunk size (1057 vs ~1450 t/s prefill).

## Steps

1. Fold GPT review findings (req_a956c612be5e4284); fix any BUG items with tests. 2. Measure per-AR critical path after changes. 3. KLD/greedy contract evidence for promotion.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

ABBA decode vs RCCL on deployment profile; greedy identical; graph-capture on; no hang over 1h soak.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Evidence: flashnext-cpuroot-4 ABBA vs auto (-ts 4,4,3 ub1024): no-MTP decode +6.4%, MTP3 +5.1%, greedy identical. Census: 96 ARs/token, 10-40 KB, cpu-root consume wait ~121 us per AR = rank arrival skew (R9700 slowest). Microbench tools/lab/rccl/cpu-root-ar.hip: 3 ranks 10KB 13.8 us vs RCCL 33.1. Related: PGC10 (3-GPU adaptive root/RCCL - this patch is its realisation), RNX11 (AR hardening), RV4210. GPT code review in flight: req_a956c612be5e4284 (memory ordering, spin strategy, fewer PCIe round trips, AR+residual fusion).

## Change Log

- 2026-10-03T15:20:13.613565+00:00 (created-by): Created by agent
