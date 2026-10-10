---
id: BCOP120
order: 120
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T04:13:12+11:00'
created-by: agent
priority: P1
work: S
---

# TRVP16: gate Vulkan multi-RDNA collectives on inactive-rank safety and upstream reuse

## Discovery / disposition

1290's evaluated Vulkan host-F32 provider unconditionally reads all rank tensors. Pinned meta's fallback first FILL-zeroes inactive `!GGML_TENSOR_FLAG_COMPUTE` ranks; the provider does not. Source-proven semantic gap, **not** a reproduced GPU fault. A false return after partial writes would also invoke generic fallback on already-mutated data. [llama.cpp #25051](https://github.com/ggml-org/llama.cpp/pull/25051) is open and already supplies a mapped-host/timeline Vulkan provider, making a separate phase-1 implementation duplicative until its current diff is evaluated.

## Authoritative owners and active-work exclusion

PRVP03 owns the 1290 prewrite inactive-rank guard and upstream adoption decision. TRVP16 owns D=2/3/4 mixed-RDNA qualification; RRVP05 owns hardware campaign; RRVP02's Vulkan implementation pause remains in force. No new provider, allocator, scheduler, cache, transport, profiling queue or configuration surface. Last independent TRVP16 change 2026-10-02; no independent TRVP16/PRVP03/1290 change or PR in the last 12 hours. Excluded active QFP41/1356 HIP graph/dispatch, QFP48/49/50 prefill/links, QFP36/1357 router, QFP17/1330, MTP/1348 and Radiance. No protected files/queued experiments touched. Not a repeat of BCOP77 (CM1) or BCOP118 (placement).

## Unresolved action and terminal gate

First, prewrite reject any inactive rank in 1290 or prove zero-source/destination semantics with exact meta fallback parity; prohibit false after mutation. Then validate #25051's actual F32 ordinary mapped-host path (the F16 ring is >=2 MiB/all-COMPUTE only), imported host memory, driverUUID/opaque-FD timeline, no-P2P, D=2/3/4, multi-turn and graph replay. If no correct eligible driver/shape, return `NO_PROVIDER_LANE` and close/defer. Only after correctness and non-overlapping hardware availability may RRVP05 run 4-session/10-ABBA matched controls; promote with CI95-low >=3% E2E and <=1% control regression. Stock meta fallback remains default.

## Evidence / validation

Pinned b11474 `ggml-backend-meta.cpp` (~2333-2360, 2461-2479), BigCherry 1290 patch.py, upstream merged [#29793](https://github.com/ggml-org/llama.cpp/pull/29793), open #25051 head `99becee6`, RRVP05/PRVP03 2026-10-02 dual-XTX receipts. 10/10 source assertions and 6/6 disposable Python host-only unittest methods passed; one fixture assertion typo was corrected before the passing rerun. No repository pytest, HIP/Vulkan compilation, GPU benchmark or new performance measurement.
