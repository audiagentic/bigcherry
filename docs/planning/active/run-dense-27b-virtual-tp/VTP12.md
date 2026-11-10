---
id: VTP12
order: 11
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-11T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Native heterogeneous INT8/FP8 three-GPU and asymmetric operator feasibility

## Why separate from VTP04

Answer whether native gfx1201 FP8 (E4M3 or a specific supported block
FP8 layout) makes the R9700 fast **at its assigned work**, while two
gfx1100 XTX cards use W8A8 INT8. All THREE physical GPUs must compute
simultaneously on every distributed layer. Do not substitute a
TP2-to-R9700 sequential layer pipeline: that is XDP02.

The candidate physical rank mapping is XTX0: one full-attention KV head,
XTX1: one, R9700: two, and 6/6/12 query heads (24 total). Natural
GDN key/value head candidates are 4/4/8 of 16 and 12/12/24 of 48;
conv/state and attention projection details still require a separate
legal per-operator layout proof. Initial FFN splits are 30/30/40,
35/35/30 and 37.5/37.5/25 **per physical GPU**, tile-rounded;
two logical R9700 halves under VTP04 are a matched alternate.

## Phase A: cheapest falsifiable GPU evidence

1. Card-free inventory of source BF16 tensor shapes, original stored
   quant encodings, actual gfx11/gfx12 FP8 and INT8 GEMM registry entries,
   activation requirements, K/N alignment, scales and fused operators.
   Record a deterministic per-op rank ownership and legality table for
   full attention, GDN/recurrent state, FFN, embeddings, output, MTP.
   Label unsupported combinations, not speculative capabilities.
2. Before writing any new runtime, run independent single-card
   `rad-kbench`/native selftests (reuse KAB01/VTP05 harness and VTP07/09
   oracles) for matched original-27B gate/up/down **and** full-attention
   and GDN shapes. Compare R9700 native W8A8 versus every proven
   E4M3/block-FP8 kernel, with both XTX W8A8 controls. M=1/4/8/64/
   128/512/2048; cold/warm, clocks, kernel row, layout conversion and
   A8/FP8 activation production costs reported separately. Use the
   native model tensors where available; synthetic weights must be
   clearly labelled and cannot prove model quality.
3. Use a bounded three-physical-GPU concurrent harness: dispatch
   XTX0/XTX1 INT8 and R9700 FP8 **at the same time** on identical
   activation input, with the intended legal unequal partitions.
   Compare uniform W8A8 all-three and VTP04 fused virtual-TP4 control
   at identical geometry. Require the R9700's two heads to launch and
   process concurrently in one context, not two serial host stages.
4. Exact F32 host-mediated no-P2P physical three-way reduction: convert
   partials to a declared common output ABI; measure quantizer, launch,
   slowest-device barrier, R9700 local-combine where relevant, D2H,
   CPU accumulation, H2D broadcast and exposed wall time. The result
   must match a trusted BF16/reference computation within explicitly
   calibrated tolerance; do not compare FP8-quantised partial output
   to INT8 bitwise or mask numerical differences with wire quantisation.
5. Balanced forward/reverse ABBA at least three fresh starts, with
   device clocks, physical VRAM and PCIe transfer receipts; measure
   per-GPU p50/p95, R9700 critical-path frequency and `max(device
   completion)+collective` rather than summed GPU kernel durations.
   No hardware runs without exclusive agent-lease; avoid an expensive
   whole-model GPU run until this small gate is complete.

## Decision table to report

- Is R9700 native FP8 really faster than its **same-shape, same-residency**
  W8A8 kernel after activation preparation and layout costs?
- Does the three-GPU INT8+FP8 combination beat all-three W8A8?
- Which legal physical FFN split minimizes the measured slowest
  participant? VTP05 preliminary **warm W8A8 M4** at 30/30/40
  measured XTX 134.5/138.3 us vs R9700 102.7 us; do not reuse the
  bandwidth-only 37.5/37.5/25 preference without comparing data.
- Is R9700 two-head attention or GDN slower than XTX one-head operators,
  and can native gfx1201 kernels offset this?
- Does the gain survive no-P2P three-way communication; does it remain
  at 4, 6 and 8 streams (later VTP13)? Is FP8 model quality acceptable?
- What physical weight/KV/graph/activation bytes remain? FP8 versus INT8
  does not by itself produce a substantial weight-capacity saving.

## Acceptance / fail-closed outcome

Report complete precision/shape/arch dispatch matrix, checked output
error, representative kernel and 3-card physical barriers, and a ranked
measured candidate with confidence intervals or a documented NO-GO.
Do NOT claim engine support, full-model accuracy or a speedup from the
microbenchmark alone. Block integration if per-rank mixed activations,
attention/GDN ownership, output ABI or gfx1100/gfx1201 dispatch cannot
be proved. Findings feed XDP10 variant selection and VTP13 engine,
not VTP03 virtual-rank integration by assumption.

Dependencies: VTP01/02/04/05/07/08/09, XDP03, XDP10, VTP06, KAB01.
