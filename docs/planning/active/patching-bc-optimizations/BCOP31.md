---
id: BCOP31
order: 31
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T21:45:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Evaluate native AMD/HRX-style backend opportunities

## Description

Determine whether a more AMD-native execution/backend path can materially outperform or simplify BigCherry's current HIP/Vulkan llama.cpp path on RDNA3/RDNA4. This is an evaluation and integration-boundary item, not authorization to fork the backend stack. Prefer upstream or externally maintained primitives where possible.

## Steps

1. Inventory candidate AMD-native runtimes/kernel stacks applicable to llama.cpp-class inference and record support for gfx1100/gfx1201, quantized GEMM/MMQ, attention, MoE, MTP, graph capture, multi-GPU and host-weight execution.
2. Define a minimal representative kernel/workload matrix: dense Q/K/V and FFN matmul, K-quants, Flash Attention, sparse/routed MoE, MTP draft path, PP512/2048 and TG128/512.
3. Compare against the current promoted HIP baseline at equal quant/model/context/batch and VRAM budget. Record kernel time, launch/sync overhead, allocator cost and end-to-end PP/TG.
4. Separate backend capability wins from kernel wins. If a candidate only contributes a kernel primitive, integrate it through existing BigCherry patch/provider ownership rather than introducing a parallel runtime.
5. Require correctness parity and repeatable hardware evidence on both gfx1100 and gfx1201 before proposing production integration.
6. Feed measured per-op/backend costs to RPL01; RPL01 may compare backend candidates but does not own backend implementation.

## Validation

- Same model/quant/context/batch inputs and correctness contract as baseline.
- ABBA >=5 repetitions for promoted candidates.
- gfx1100 and gfx1201 tested independently before heterogeneous conclusions.
- No claim based solely on synthetic GEMM throughput if end-to-end PP/TG does not improve.

## Acceptance Criteria

- Candidate capability matrix and reproducible benchmark evidence exist.
- Any promoted integration provides >=5% end-to-end gain in a material workload or unlocks a currently unavailable capability without >2% regression elsewhere.
- No duplicate model loader, scheduler, residency policy or multi-GPU planner is introduced.
- Unsupported candidates are dispositioned with evidence rather than retained as speculative production paths.
