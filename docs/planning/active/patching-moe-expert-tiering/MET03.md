---
id: MET03
order: 3
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:44:56.833804+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# 1282 qwen4exp_moe_tier_graph: 2-tier POC (primary TP group + CPU) on offline-repacked GGUF

## Description

First experiment that proves value. Offline byte-preserving repack: per layer derive a permutation from placement, reorder router rows (`ffn_gate_inp`), expert axis of `ffn_{gate,up,down}_exps` and any expert-indexed scale/bias, then split into contiguous tier tensors. No dequant/requant. Qwen4Exp gets `build_moe_ffn_tiered()`: per-tier gate/up/down via `ggml_mul_mat_id_range`, lane-wise sum, then existing weighting/reduction. Shared expert remains unchanged and generic `build_moe_ffn` stays the control path.

MET03 also owns CPU-cold-tail execution efficiency for this POC. Do not create a separate CPU-MoE fusion plan. External fused-CPU/OpenVINO results are mechanism evidence only; the BigCherry gate is whether a deliberately small CPU tail is dominated by graph/intermediate/scheduler overhead rather than arithmetic.

## Steps

1. Repack tool (`tools/lab`, gguf-py): permutation + tier split, writes inverse map.
2. Patch: Qwen4Exp loader accepts tier tensors; `build_moe_ffn_tiered` behind placement-present.
3. Four Q6 builds with CPU receiving ~0.5/1/2/5% route mass.
4. Measure decode/prefill versus predicted CPU-active-layer curve.
5. Attribute CPU-tail cost before fusion: selected experts/tokens, gate/up/down matmul, activation, intermediate bytes, combine, scheduler/barrier and total CPU service time. Keep unfused range-op execution as the control.
6. Attempt fused gate+up+activation+down only if non-matmul overhead is >=15% of CPU expert service. Reuse MET02 range semantics; inactive lanes remain exact zero and never remap to expert 0.
7. Do not import NUMA expert sharding on the desktop target. Qualify only on multi-NUMA hardware with measured remote-memory traffic.
8. Compare fused/unfused at identical placement and route mass for plain decode, MTP decode and pp512/4096. Promote only on end-to-end wins; microkernel gains alone are insufficient.

## Detailed Solution & Technical Design

The tier graph intentionally makes CPU work sparse, so the target is low-latency sparse expert service rather than generic high-throughput CPU MoE. `ggml_mul_mat_id_range` remains the semantic primitive and MET01 placement remains authoritative; no second router, expert store, placement layer, cache or transport policy is added here.

Fusion candidate semantics:

```text
for each active (token, slot) whose expert is in CPU tier:
    gate = W_gate[e] * x
    up   = W_up[e]   * x
    h    = silu(gate) * up
    y   += router_weight[token,slot] * (W_down[e] * h)
```

Fuse only stages shown expensive by profiling. Fold router weighting/combine only when accepted numeric-order tolerance is preserved; otherwise retain the current combine. Expected benefit is reduced intermediate traffic and graph/scheduler overhead at low CPU route mass.

Keep decode and prefill gates separate. Published fused-MoE evidence shows prompt-processing gains can be much larger than token-generation gains, so a prefill-only winner may be dispatched only in that regime.

## Code Samples & Guidance

Suggested profiling record:

```text
{layer, cpu_route_mass, active_cpu_experts, active_cpu_tokens,
 gate_up_ms, activation_ms, down_ms, combine_ms, scheduler_ms,
 intermediate_bytes, total_cpu_tail_ms, pp_tps, tg_tps}
```

Promotion: correctness first; then positive repeated end-to-end CI in at least one declared lane and <=1% regression in the other primary lane unless dispatch is explicitly restricted to the winning batch regime.

## Files

`patches/1282_qwen4exp_moe_tier_graph/`; `tools/lab/flash-next/expert-placement/repack.py`; if profiling justifies fusion, ggml CPU `MUL_MAT_ID`/range execution and Qwen4Exp tier graph only. Do not add a placement/cache subsystem.

## Validation

Router permutation unit test; inverse-mapped top-k sets/weights identical; per-layer monolithic vs tiered MoE op; full-model greedy parity + KLD versus same-quant monolithic baseline; plain decode, MTP depth 3, pp512/pp4096 at 8K/64K/192K; record CPU-active layers/token, bytes/tier/token, VRAM peak and MTP acceptance.

CPU-tail fusion: exact same repacked GGUF/placement at ~0.5/1/2/5% CPU route mass; report service ms per active layer, intermediate bytes, scheduler time, pp512/4096 and effective TG/MTP. Test zero/mixed/all-local range cases and Q6_K first.

## Effort & Risk

Medium-high. Tier graph/repack is correctness-sensitive; CPU fusion adds numeric-order and dispatch risk, so it is attribution-gated. NUMA sharding remains out of scope without relevant hardware evidence.

## Standards

One placement owner (MET01), one range semantic owner (MET02), one tier-graph/CPU-tail owner (MET03). Profile before fusion; reuse-before-fork; batch-regime-specific dispatch only with measured gates.

## Acceptance Criteria

- Tiered graph preserves router/expert semantics and passes greedy/KLD gates.
- CPU route-mass curves measured for 0.5/1/2/5% placements.
- CPU-tail attribution separates matmul, activation/intermediate, combine and scheduler cost.
- No fused CPU implementation promotes unless non-matmul overhead is material and end-to-end measurements win.
- Fusion reuses MET02 range semantics and creates no duplicate placement/cache/router/expert store.
- Decode and prefill decisions are independent.

## Notes

Requires MET01 (profile) and MET02 (range op). Decision gate: if throughput tracks the predicted curve, proceed to MET04/MET05.

2026-10-05, folded in from BCOP16 (audit backfill) - fused CPU cold-tail gate: before any CPU-tail fusion, profile gate/up/down matmul, activation/intermediate traffic, combine, scheduler and CPU service time. Attempt fusion only if non-matmul overhead is >=15%. The current Brutus deployment keeps all experts on GPU, so this only matters if a CPU tail returns.

External mechanism references: https://github.com/gamozolabs/llama.cpp ; https://github.com/ggml-org/llama.cpp/releases/tag/b11374

## Change Log

- 2026-10-02T04:44:56.833804+00:00 (created-by): Created by agent
- 2026-10-05: BCOP16 audit backfill added CPU-tail fusion gate.
- 2026-10-05: Transplanted structured attribution, fusion semantics, validation and ownership details from `automation-qfp-indexer-20261004`.
