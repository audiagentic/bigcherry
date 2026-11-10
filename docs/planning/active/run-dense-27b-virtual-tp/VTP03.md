---
id: VTP03
order: 2
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Three-device collective, local R9700 combine and end-to-end experiment

## Experiment hypothesis

Three physical contributions (XTX0, XTX1, locally combined R9700)
can execute one logical four-shard transformer layer faster than
a serial XTX TP2 + R9700 layer pipeline **at the same Q6/Q8 quality
and resident-context target**, despite the R9700 Gen4 x4 upstream link.
Must outperform the best qualified control, not just an equal-share
synthetic computation graph. No such runtime is implemented today.

## Isolated bench matrix and stop gates

**Stage A — card-free**: VTP01 shape/buffer/epoch audit and VTP02
memory feasibility. For FFN GEMM compare both single-device native
kernel libraries and compute-only full-shape time at real M=1/4/8/64/
128/512/2048; include quantize/dequantize overhead. Capture measured
per-output-tile cost, not solely peak TFLOPS.

**Stage B — exclusive GPU lease; physical transport microbench only**:
reuse r3-arbench, host-link-bw and PGC/RAD50 link findings where
compatible, otherwise make a new bounded bench in the existing
worktree. Measure exact-F32 and explicitly quality-qualified optional
wires, local R9700 two-slice reduction and a three-physical-participant
host-mediated collective across decode, MTP verify and prefill row bands.
Place the R9700 both idle and doing local GEMM work; measure PCIe bytes,
kernel launch/service time, GPU wait, root link contention, p50/p95,
and deadlock/replay/cancel safety. Compare a single R9700 combined
packet versus **two** independent host packets. NO fake four-physical
participants or unsound spin-wait synchronisation.

**Stage C — candidate physical compute partition**:
30/30/20/20, 35/35/15/15 and 37.5/37.5/12.5/12.5; use rounded legal
operator widths, separate attention/KV head map (1,1,1,1).
On the R9700 compare (i) one fused 40% or 30% or 25% local workload,
(ii) two sequential sub-shards, (iii) two event-synchronised
streams with proper nonblocking event dependencies. If two streams
contend or regress, select one fused physical executor.

**Stage D — opt-in engine prototype only after Stage B go**:
model Q6/Q8 + matching BF16-origin reference, four logical shards
over 3 physical cards, repeatable prompt corpus, 1/4/6/8 concurrent
streams, short (8K), medium (32/64K), long (128/256K native) context.
Control arms: XTX TP2 same quant/KV; XTX TP2 + R9700 full-layer
stage (once implemented), and where memory permits independent
R9700 replica. Same binary/pin, tokenizer, sampling, batch budget,
draft depth, resident KV and model tensor encoding; do not mislabel
Flash-Next MoE results as dense 27B controls.

## Evidence and decisions

Capture TTFT, client-output tok/s **and** engine output separately,
per-stream latency and p95 step, steady-state tokens/step,
acceptance, cold/warm prefill, energy optional, physical VRAM split,
KV allocation/spill, GPU GEMM %, host copy %, collective exposed
wall time (not summed across devices), GPU clocks and all four-stream
quality. Run balanced paired ABBA order with at least 3 repeats and
different startup orders; increase only to resolve noise.

- Reject virtual TP4 if the R9700 is still a dominant straggler after
  candidate ratio tuning, if collective overhead overwhelms compute
  savings, or if four logical head/state ownership cannot be correct.
- Reject unsafe local-GPU spin waits, graph epoch reuse, stale host
  visibility and same-GPU deadlocks regardless of throughput.
- Proceed to production admission work only if quality passes,
  4+ stream throughput and p95 beat best feasible Q6/Q8 baseline,
  and measured resident KV headroom improves, with no unsafe OOM.
  No invented percentage speed target or production default change.

## Status

All stages specified only. No GPU lease taken, no new benchmark executed,
no live kernel/collective prototype and no claims of runtime support.
