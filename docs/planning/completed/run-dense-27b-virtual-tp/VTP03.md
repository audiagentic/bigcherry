---
id: VTP03
order: 2
plan: run-dense-27b-virtual-tp
state: superseded
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Quality-qualified Q6/Q8 virtual TP4: full 27B A/B acceptance

> Archived 2026-11-12 from branch `exp/xtx27-prefill-opt` (git tag `archive/exp-xtx27-prefill-opt`), which never merged. Its lab programs (tools/lab/radiance/xtx27_*), tests and early patch drafts are in that tag. Superseded by VTP14 (native Radiance TP3, completed 2026-11-12), which kept these programs as reference oracles only.

## Scope

Only after VTP01 has a reviewed physical-GPU execution contract,
VTP02 has real per-device memory receipts, VTP04 has qualified small
physical-device tests, and XDP03 has proven native Q6 or Q8 dispatch,
implement opt-in Radiance virtual TP4 over two XTX + one R9700.

This is the **end-to-end experiment** plan, not the owner of synthetic
GPU communication/local-shard microbenches (VTP04), of model conversion
(XDP03), or of full-layer R9700 staging (XDP02/HD2701).

## Hard preconditions

- Four logical weight partitions with legal per-operator alignment,
  routed through three physical executors, R9700 ranks 2 and 3
  sharing a single VRAM allocator, KV ownership and GPU context.
- A real heterogeneous gfx1100/gfx1201 kernel dispatch for every
  dense projection, complete attention and recurrent/GDN operator.
  The present Radiance launcher prohibits mixed engine architectures;
  no tricked physical-device enumeration is acceptable.
- One on-GPU sum for R9700 local partials, and one safe exact
  three-physical-GPU no-P2P reduction (or independently proven
  quality-safe compressed option). No GPU-side spin-wait on sibling
  logical ranks; cancellation and graph replay epochs tested.
- Quality-qualified original-BF16-derived INT8 W8A8 and/or IQ6R or
  separately compatible GGUF Q6_K/Q8_0. Conversion support alone
  does not establish native GPU kernels. Do not substitute Q4
  to make the experiment fit.
- Actual current 27B native 262K maximum per request, with four
  heads of full attention. 1M aggregate over four 256K streams
  is distinct from a future single-request 1M model.

## A/B experiment design

Compare on the SAME checkpoint, quant, KV precision, build, prompt,
sampling, MTP setting and resident token budget:

- Existing dual-XTX TP2 control.
- Virtual four logical shards mapped 30/30/20/20, 35/35/15/15,
  37.5/37.5/12.5/12.5, with measured legal integer rounding.
- TP2-XTX plus R9700 full-layer stage if a safe prototype exists.
- Independent R9700 replica if the same Q6/Q8 model fits and the
  same total GPU resources are accounted for.

Stages: first M1/4/8 GEMM+collective and full layer with 4/6/8
independent streams at 8K; next pp512/2048/8192;
then 32K/64K/128K/256K native context and 4/6/8 streams.
Where memory fails, report explicit admission refusal rather than
host-spilled tokens or lowering precision. Reuse KAB01/rad-kbench
captured native operator fixtures before any whole-model run.

Balanced paired ABBA repeats at least three times, with multiple
fresh starts and rotated mode order. Record p50/p95 per-stream
step, real client output tokens/s, cold/warm TTFT, MTP accepted
tokens/step and quality/KLD/top1, physical GPU weight/KV/GDN/state
and scratch allocations, exposed per-step collective, measured PCIe
bytes/latency and no unplanned RAM spill. Do not use summed
GPU kernel time as exposed wall time.

## Required competing three-physical-GPU design (2026-11-11)

Before committing to a four-logical-rank engine, compare the **one
physical rank per GPU, asymmetric TP3** design in [VTP12](VTP12.md)
and [VTP13](VTP13.md). Both have 1/1/2 physical KV-head ownership and
three collective endpoints; TP3 handles both R9700 heads in one
physical GPU context and concurrently with XTX compute, rather than
running a serial TP2-plus-R9700 layer stage. Do not presume two
virtual slices offer extra R9700 compute. Keep fused virtual TP4 as
an A/B candidate if a single unequal rank's kernel shapes regress.

Add to each matched candidate a **uniform W8A8 on all three** control
and a provisional **XTX INT8 / R9700 FP8** arm when XDP10's variant
loader and VTP12's cross-rank activation/output/numeric contract are
proved. IQ6R remains a separate BF16-calibrated, quality-qualified
arm. With per-format weight sizes, test actual VRAM/KV tradeoffs and
both FFN and attention/GDN slowest-worker barriers. A faster local FP8
GEMM does not by itself prove a faster model; measure quantisation,
layout, PCIe and all-reduce exposed time. Only run the 4+ stream full
model campaigns after the small kernels and exact three-way operator
harness pass.

## Go/no-go

Reject virtual TP4 if any legal tensor/attention/GDN split or
correctness/lifetime check fails; if the R9700 dominates steps
after shape/rank tuning; if three-device communication plus
synchronisation erases the compute gain; or if no resident-context
benefit remains at Q6/Q8 after real memory accounting.

Promote only after end-to-end 4+ stream quality, throughput and
p95 latency beat the best **feasible** same-precision control with
useful resident KV headroom and safe restart/abort/replay. Document
a negative conclusion rather than promising an unmeasured speedup.

## Status

PENDING. No virtual-TP4 engine graph, production layout, model
conversion or model-serving benchmark has been created. Small
GPU experiments are independently scoped and leased under VTP04.

## Change Log

- 2026-11-12T22:13:15.023202+00:00 (state-transition): State: pending → superseded
