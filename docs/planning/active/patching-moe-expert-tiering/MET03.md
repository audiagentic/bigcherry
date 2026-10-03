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

First experiment that proves value. Offline byte-preserving repack: per layer derive a permutation from placement, reorder router rows (ffn_gate_inp), expert axis of ffn_{gate,up,down}_exps and any *_exps_s / expert-indexed bias, then split into contiguous tier tensors (blk.N.ffn_*_exps.tierK.weight). No dequant/requant. qwen4exp gets build_moe_ffn_tiered(): per tier gate/up/down via ggml_mul_mat_id_range, lane-wise sum, then the existing weighting/reduction. Shared expert untouched. Generic build_moe_ffn unchanged (control path).

## Steps

1. Repack tool (tools/lab, gguf-py): permutation + tier split, writes inverse map.
2. Patch: qwen4exp loader accepts tier tensors; build_moe_ffn_tiered behind placement-present.
3. Four Q6 builds with CPU receiving ~0.5/1/2/5% route mass.
4. Measure decode/prefill vs predicted CPU-active-layer curve.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

patches/1282_qwen4exp_moe_tier_graph/, tools/lab/flash-next/expert-placement/repack.py

## Validation

Router permutation unit test (inverse-mapped top-k sets and weights identical); per-layer MoE op monolithic vs tiered; full model greedy parity + KLD vs same-quant monolithic baseline; perf: plain decode, MTP depth 3, pp512/pp4096, 8K/64K/192K; record CPU-active layers/token, bytes/tier/token, VRAM peak, MTP acceptance.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Requires MET01 (profile) and MET02 (range op). Decision gate: if throughput tracks the predicted curve, proceed to MET04/MET05.

## Change Log

- 2026-10-02T04:44:56.833804+00:00 (created-by): Created by agent
