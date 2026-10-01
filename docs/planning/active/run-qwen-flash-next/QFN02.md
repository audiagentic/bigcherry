---
id: QFN02
order: 0
plan: run-qwen-flash-next
state: pending
created-at: '2026-10-01T12:36:26.056572+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
---

# Frequency-aware MoE expert placement: hot experts on the tensor-split cards, cold experts on the 6900 XT / RAM

## Description

Owner 2026-10-01: use the 6900 XT (chipset PCIe, no atomics, so no RCCL) for experts, keeping frequently used experts on the fast cards. llama.cpp places each layer's ffn_*_exps tensor (all 512 experts) statically; -ot moves whole tensors and --n-cpu-moe whole layers; there is no per-expert placement or dynamic swapping. Qwen3.8-Flash-Next: 48 layers x 512 routed experts, 10 active per token, expert FFN 640, 55.4 GiB of experts out of ~60 GiB compute weights.

## Steps

1. Routing profile: a diagnostic patch logging top-k expert ids per layer (env-gated, like 1277); run representative prompts (chat, code, long context); compute per-layer usage skew (share of tokens covered by the top-N experts). Go/no-go on step 2+ from the skew.
2. Offline GGUF repack: per layer, permute experts by usage (ffn_gate/up/down_exps along the expert axis and the router ffn_gate_inp rows, plus any per-expert bias/scale) so hot experts are contiguous; output must be bit-identical to the original model.
3. Expert-split patch: split ffn_*_exps along the expert axis at a per-layer boundary; hot slice on the tensor-split cards (rows split as today), cold slice whole on the 6900 XT (or CPU); MUL_MAT_ID runs on both groups with only the ids each holds, outputs summed; activation copies only (no collectives on the 6900 XT).
4. Measure: decode/prefill vs all-resident 3-card tensor split, VRAM freed, MTP fit, hot-hit rate.
5. Compare with the simple -ot whole-layer offload (probe v3, flashnext-probe-3).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Greedy output identical to the unsplit layout; balanced A/B decode/prefill; per-layer cold-hit rate recorded.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Depends on QFN01 tensor split (1278) and the 6900 XT staying out of RCCL groups unless PGC14's hostcall-free RCCL build works.

## Change Log

- 2026-10-01T12:36:26.056572+00:00 (created-by): Created by agent
