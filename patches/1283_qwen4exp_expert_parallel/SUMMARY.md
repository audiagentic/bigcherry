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

## Evidence

- Offline mechanics test: pending. Hardware: blocked on 1281 phase B (range op on the GPU).
