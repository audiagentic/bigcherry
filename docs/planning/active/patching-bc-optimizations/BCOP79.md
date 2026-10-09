---
id: BCOP79
order: 79
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T00:10:02+00:00'
created-by: agent
priority: P2
work: S
---

# PRBE54: qualify Q5_1 dequant rounding

## Finding
Pinned b11474 uses float multiply/add for Q5_1 before F16 conversion. Patch 1271 instead uses half2 multiply then half2 add. Host model: 28,261/131,072 Q5_1 values differ; Q5_0 matched all 131,072. Not GPU evidence.

## Owner and disposition
PRBE54/1271 owns Q5 FlashAttention contiguous F16 staging. QFP17 active masked prefill remains untouched. Retain existing selector and qualification infrastructure. No new scheduler or cache.

## Gate
Require Q5_1 float-reference parity or disable Q5_1 selector. Verify actual TILE/MMA staging, then measure gfx1100/gfx1201 critical-path converter share. Close if <5% or theoretical E2E ceiling <3%. Otherwise four sessions, ten paired rounds, full-vocab correctness and <=1% control regression. No hardware test this run.

## References
PRBE54; llama.cpp PR #27140, #29827, #29846.
