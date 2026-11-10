---
id: VTP02
order: 1
plan: run-dense-27b-virtual-tp
state: active
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Q6/Q8 memory and attention-KV ownership: weighted TP4 envelope

## Purpose

User requires **Q6 or Q8, not Q4**, with maximum **resident** context and
4+ streams. Avoid extrapolating Q4/MXFP4 results to higher-bit encodings.
For first matched model use Qwen3.8-27B BF16 checkpoint, already converted
Radiance W8A8/INT8 and matching Q8_0 GGUF control as **distinct** formats.
Q6_K GGUF and calibrated Radiance IQ6R are likewise distinct. XDP03 owns
quant conversion, native gfx1100/gfx1201 dispatch and BF16 KLD validation.

## Key arithmetic (lower-bound, not full VRAM budget)

Physical capacities 24/24/32 GiB = 30/30/40% of 80 GiB. Four
logical sharded-weight fractions 0.30/0.30/0.20/0.20 therefore distribute
**shardable weights** 30/30/40% physical, but replicated embeddings/head,
norms, vision, drafter, GDN/conv and scratch may invalidate proportional
total weights. Require actual tensor census / per-device allocator receipts.

At four logical full-attention KV ranks, one KV head per rank means
physical **25/25/50%** of full-attention KV, independent of the weighted
FFN/matrix slices. At 1,048,576 *aggregate cached tokens* across all
streams: FP8 attention KV = 32 GiB model-wide, physical 8/8/16 GiB.
For `k6_v4`, layout coefficient is 22/32 vs ordinary FP8, thus 22 GiB
model-wide, physical 5.5/5.5/11 GiB (logical bytes only, not page
allocation and not qualified on gfx1100). `fp8_v4` is 25/32.
The current checkpoint officially supports up to 262144 **per request**;
1M aggregate may be 4×256K, while a single 1M request is hypothetical
for a future supporting checkpoint.

30/30/20/20 is a **capacity** initial candidate, not a speed optimum.
Bandwidth-only physical work balance favours ~37.5/37.5/25%, equivalent
to 37.5/37.5/12.5/12.5 logical. Additional candidates:
35/35/15/15. All require integer/tile-aligned per-operator splits.
For FFN width 17408, 128-wide tiles total 136; the nearest candidate
for 30/30/20/20 is (41,41,27,27) tile groups, not exact percentages.
Do not apply that tile rule blindly to attention heads, GDN or output head.

## Card-free experiment

`tools/lab/radiance/xtx27_virtual_tp4.py` computes logical KV exactly
and reports physical weight/KV budgeting for candidate fraction vectors,
provided *explicit* measured weight allocations or a clearly labelled
synthetic total. The script never queries or uses GPUs and NEVER approves
a production model plan. It supports aggregate cached tokens and validates
that fractions sum to one, there are four ranks, two on the R9700, and
reserves are physically charged once.

Preview: `python3 tools/lab/radiance/xtx27_virtual_tp4.py --demo`.
It labels all example weight/reserve figures synthetic and refuses
implicit production-valid conclusions. Validation:
`python3 -m unittest tools.tests.lab.test_xtx27_virtual_tp4`.

## Hardware fit and numerical gates (future, exclusive GPU lease)

1. Capture real per-tensor/model .rad Q8 and Q6 weights, quant scale
   metadata, static/replicated bytes, actual allocator headroom and BF16
   quality. Do not use extrapolated 30 GiB model sizes as admission.
2. Test actual 4/6/8 streams at 8K/32K/128K and up to native 256K;
   ensure GDN/checkpoint/vision/embedding memory is charged. Test
   combined-context admission and one long-sequence admission separately.
3. Qualify physical per-device KV and `k6_v4`/`fp8_v4` model/architecture
   availability and quality; no claim that a libr4d kernel exists in libr3
   without a real dispatch/shape/profiling proof.
4. Require 10% or more unclaimed **physical** VRAM per device after all
   allocations as an initial experimental safety floor, adjustable only
   with measured driver/headroom evidence; not a generic product limit.
5. Fail if model/runtime cannot map four logical KV heads correctly,
   if actual allocated KV exceeds plan, if host spill is hidden, or
   if quality/KLD/MTP acceptance differs unacceptably from exact control.

## Physical TP3 and mixed-precision extension (2026-11-11)

Keep **two independent ownership axes**: FFN/shardable weight ratio
30/30/40 (and measured alternatives 35/35/30, 37.5/37.5/25), versus
full-attention KV heads 1/1/2 (25/25/50% KV) on physical XTX0/XTX1/
R9700. The latter is unchanged whether R9700 is one unequal TP3
rank or two virtual TP4 ranks. Operators may have distinct legal tile
rounding and replication rules; physical combined VRAM is not equal
to four equal logical ranks. Reuse [VTP08](VTP08.md) exact W8A8 FFN
floor and extend to real FP8 layouts rather than applying assumed bpw.

A multi-variant disk bundle (XDP10) consumes VRAM only for its
**selected** INT8/FP8/IQ6R shards. Count actual per-format scale,
activation staging, GDN/conv recurrent state, replicated/output
weights, KV, graphs, allocator padding and 10% provisional reserve.
FP8 and INT8 are both eight-bit weight codes, so do not credit a
capacity win without byte receipts. Assess 4/6/8 streams up to
native 262144 tokens each and verify R9700's two-head KV/compute
work neither stalls the XTX nor spills. VTP13 owns runtime admission;
VTP12 owns mixed-precision operator measurements.

## Completion criterion

Measured, per-physical-GPU Q6/Q8 *resident* weight+KV+state+scratch
budgets and maximum **quality-qualified** context at 4+ streams for each
candidate; explicit out-of-memory and unsupported-format refusals.
No engine integration before these receipts.

### Original checkpoint census cross-check (VTP06, 2026-11-10)

The structural BF16 header/index audit reads the actual 18-shard
Qwen3.8-27B source and finds **34,225,520,640 BF16 bytes** in its
192 text-FFN gate/up/down projection tensors. This is 31.875 GiB
**before quantization**. The other 21,337,335,264 BF16 bytes include
attention/GDN, embeddings/head, MTP and vision, each requiring its
own explicit quantization and physical-placement plan. This census
does not make Q8 weights equal to BF16/2: actual scales, alignment,
side tables, replicated tensors and MTP/vision handling must be
accounted for by the real converted .rad artifact (VTP06).
