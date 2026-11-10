---
id: VTP13
order: 12
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-11T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Opt-in asymmetric physical TP3 runtime and full 27B performance A/B

## Goal

Provide a competing implementation to VTP03 virtual TP4: **one worker,
one GPU context and one collective endpoint for each of XTX0, XTX1
and R9700**, but unequal per-operator tensor ownership. All three
compute every distributed layer concurrently. The R9700 owns two
of four full-attention KV heads in ONE physical worker; this must not
be a serial whole-layer stage. The project is primarily to improve
real 4+ stream throughput **and** resident KV capacity relative to
dual-XTX TP2, not merely make the model fit three GPUs.

## Staged implementation (new managed Radiance feature; no fake --tp 3)

1. Card-free formal ownership/shape table: full-attention KV 1/1/2,
   query 6/6/12, GDN key 4/4/8 and value 12/12/24 as candidate
   mappings; prove actual Q/K/V, GDN conv, state, FFN gate/up/down,
   embedding/head, MTP/drafter, scale, normalisation and residual
   boundaries. All tensor offsets nonoverlapping and covering exactly
   the full original dimensions, with legal kernel/quant tile rounding.
   Example FFN 17,408-wide 128-tile 30/30/40 split: 41/41/54 tiles.
   Weight share is NOT tied to 1/1/2 KV ownership.
2. Extend the managed layout/engine startup ABI to express explicit
   `physical-rank -> device UUID, arch/plugin, per-op offsets, KV-head
   ownership, collective endpoint`. Do not bypass current unique
   device or uniform-world-size guards by aliasing two ranks to R9700.
   Verify mixed gfx1100 libr3 and gfx1201 libr4d plugin compatibility
   per op, not just separate process starts.
3. Implement one asynchronous physical contribution per layer, with
   two R9700 attention heads scheduled **together** via a compatible
   native kernel or overlap-safe launches. No sibling logical-rank
   spin-wait, repeated CPU round-trip per head, or extra rank barrier.
   Preserve GDN recurrent state per physical owner and legal KV head
   owner, graph/epoch identity, scratch lifetime and cancellation.
4. Reuse XDP10's pinned **selected variant** for each tensor shard:
   XTX W8A8 and R9700 approved FP8 only when VTP12 proves the common
   activation/output ABI. Uniform all-three W8A8 is mandatory baseline
   and safest first engine milestone. IQ6R is optional, only after
   VTP06 calibration, supported kernels and independent quality checks.
   Never load all alternative variants to VRAM.
5. Implement three-physical-GPU no-P2P exact collective using one
   published partial per physical rank, F32 combine, event ordering,
   host-transfer accounting, guarded graph replay and abort/restart.
   Add fail-closed admission for actual physical VRAM weights,
   embedding/head, GDN, KV, scales, temp scratch, host staging,
   graphs, fragments and a measured reserve; no hidden host spill.
6. Incremental proof: synthetic exact shard ABI -> FFN -> GDN ->
   full attention/KV -> mixed encoder/weights -> fixed greedy
   short-model replay -> full Qwen3.8-27B inference. Where a cheap
   smaller architectural fixture is legal, use it only for scheduler
   and ownership correctness, never as a substitute for 27B throughput.

## Matched 3-GPU experiment and promotion gate

Against (A) dual-XTX TP2 existing exact and approved wire controls,
(B) physical TP3 with W8A8 on every GPU, (C) physical TP3 with
XTX INT8 / R9700 FP8, and (D) VTP03 fused virtual TP4 **if built**,
run the same BF16-derived model, quality-qualified quant families,
prompt corpus, KV dtype, sampling, MTP policy, resident tokens, engine
pin and CPU/power settings. ABBA runs >=3 balanced fresh starts, 4/6/8
streams at 8K; pp512/2048/8192; then 32K/64K/128K/256K (native
per-request limit), fail-closed on infeasible resident allocations.
Report per-stream p50/p95 latency, real output tok/s, TTFT, p95
slowest-GPU barrier, prefill/decode, all-reduce/PCIe time and physical
weight/KV/GDN/scratch memory. Measure KV maximum with actual
per-physical allocator receipts, not projected head fractions.

Quality gate: held-out BF16 comparison (VTP06), KLD/PPL/top-1,
short/long greedy replay, concurrent deterministic checks, speculation
off initially then separately qualified on. Mixed dtype is not
acceptable merely because each local kernel passes an oracle.

**Accept** as a runtime option only if no ABI/quality/cancellation/
deadlock/admission failure and the end-to-end 4+ stream performance
and resident KV benefit beat the **best feasible** same-quality
two-XTX control, with a clear p95/throughput and capacity report.
If uniform W8A8 wins mixed FP8, retain it; if TP4 wins physical TP3,
document the measured crossover. Negative results close the design
question rather than silently enabling a slower engine configuration.

Dependencies: VTP01/02/04/08/12, XDP03/XDP10, VTP06;
VTP03 is an independent competing path. No production default change,
no vendor edits, no PR/push until explicitly authorised.
