---
id: BCOP84
order: 84
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-09T05:07:37+00:00'
created-by: agent
priority: P3
work: S
---

# PRBE47: wave32 DPP cross-row correctness and activation gate

## Discovery / disposition

Pinned b11474 and current upstream retain five-stage float shuffle reductions. PRBE47's historical DPP row_shr sketch is not a butterfly and DPP16 row_xmask cannot exchange lanes across the 16-lane boundary; the XOR-16 stage must remain a full-wave operation. mmvq.cu also has a separate multi-token MUL_MAT_ID kernel with normal/gate reduction sites omitted from the old four-site inventory. 8,195 deterministic host float32 wave32 fixtures proved exact XOR-16 + row_xmask-8/4/2/1 lane-model parity; this is **not** a HIP or performance result.

## Ownership and activity exclusion

PRBE47 is authoritative for Q6_K-only DPP reduction-lowering evidence. PRBE111 owns IQ vec-dot/VDR; PRBE21 and 0600/0650 own small-K MMVQ geometry; 1273 owns IQ tuning. The latest independent DPP-related commit search found only September 10 work; PRBE47's last substantive source review was September 24 and its October 8 triage was outside the 12-hour window. This is a new mechanism audit, not a repeat of BCOP79-83. Active QFP36/41, QFP35, MTP, Meta split-cache, engine migration, Radiance and other hardware lanes were excluded; no active code, queue or plan was modified.

## Bounded next action / terminal gate

First inspect stock gfx1100/gfx1201 ISA and profile reduction's attributable E2E decode fraction; **close without patch** if equivalent DPP already emitted or maximum possible gain <3%. Otherwise prototype only an additive, compile-time-gated Q6_K wave32 helper (native XOR16 + DPP row_xmask 8/4/2/1), test all 32 lanes and both standard/MoE + fused-gate call sites, require bitwise/full-vocab/MTP/graph parity, and qualify 4 sessions x >=10 ABBA pairs with CI95-low >=3% E2E decode and <=1% controls. Revert/retire if incorrect, no ISA benefit or no causal E2E win. No new scheduler/registry/allocator, and Q8_0 widening is a separate future decision.

## References

PRBE47; PRBE111; PRBE21; 0600; 0650; 1204; 1273; AMD ROCm DPP builtin guide; LLVM AMDGPU modifier syntax; pinned llama.cpp b11474 mmvq.cu and common.cuh.
