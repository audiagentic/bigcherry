# 1283_qwen4exp_expert_parallel

**Status:** validated
**Plan item:** MET04

Kind: enhancement, flag `BIGCHERRY_MOE_EP` (default 0). Requires 1281.

Whole-expert parallelism inside the tensor split. With the flag the routed expert weights are split across the
devices along the expert index, so each device holds whole experts and computes the selected ones it holds (range
MUL_MAT_ID of 1281, exact zeros for the others). The AllReduce is delayed through the block - gate, up, GLU, down -
and then by the existing delay to the block output: one AllReduce per MoE block, the same as today's row split.

It does not reduce collectives. What it changes is the kernel work per device (a few whole experts instead of a
slice of every selected expert) and placement freedom for quants whose experts do not all fit. The risks are load
imbalance (a step is as slow as the device holding most of a token's experts) and the fused expert kernels, which
must be range-aware before decode can benefit.

The delay is a strict pattern match with fallback to an immediate AllReduce; expert biases, the merged gate_up
tensor and scale MULs are not matched.

## Evidence and placement boundary

- Historical `README.md` documents validated b11402 `b-moeep-b11402r` 8K/24K ABBA, 24-probe fidelity, and `tools/tests/patch/test_1283_qwen4exp_expert_parallel.py` offline mechanics. 1283 is a **validated default-off enabler**, not blocked on unimplemented 1281 phase B. No new b11552 GPU qualification or measured placement gain is claimed here.
- 1283 matches separate `ffn_gate_exps`, `ffn_up_exps` and `ffn_down_exps` expert-axis weights; merged `ffn_gate_up_exps` is outside its strict pattern. MET08/BCOP241 requires the offline reordered-GGUF tool to reject unsupported/incomplete tensor families **before** publishing derived artifacts. Placement manifest presence is not proof of executed range nodes.
- MET09 owns B1 first-divergence attribution; MET11 owns eventual B2 mapping. No new kernel, cache, dispatcher or scheduler is proposed.
