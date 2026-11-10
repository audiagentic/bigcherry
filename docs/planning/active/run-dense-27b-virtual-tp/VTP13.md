---
id: VTP13
order: 12
plan: run-dense-27b-virtual-tp
state: active
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

## 2026-11-11 TP3 physical FFN design input from VTP12

VTP12 measured two 42.65%-R9700 FFN allocations. A symmetric
4992/4992/7424 physical split is strongly decode-regressed by
128-aligned XTX down-K shapes. An unequal **5120/4864/7424**
physical split keeps all three down-K sizes 256-aligned and the
ideal warm-weight FFN barrier within 2.1% of 30/30/40 uniform
INT8 across M=1/4/8/64. This is not yet real simultaneous TP3.

VTP13 must permit **nonidentical XTX partition widths** as well
as R9700 1/1/2 KV-head ownership; different legal FFN, GDN and
attention offsets are explicit per-operator dimensions, not a
common physical ratio. Require the actual three-GPU concurrent
exact reduction, GDN and full-attention numeric and performance
acceptance before enabling any execution path. Native INT8 kernel
dispatch is the control; the FP8 loader remains XDP10-gated.

## 2026-11-11 card-free physical-TP3 checkpoint ownership evidence

`tools/lab/radiance/xtx27_vtp13_ownership.py` now audits the
**original Qwen3.8-27B BF16 source safetensors metadata**, checking
all indexed tensor names against bounded, dtype/offset-validated
headers without reading 55.6 GB of payload. Exact output is an
immutable JSON receipt in
`${BC_WORK_ROOT}/xtx27-prefill/vtp13-physical-tp3-ownership-header-only-20261111.json`.
The config/index/header hashes provide **metadata** provenance only;
payload SHA-256 and converted artifact provenance are still open
(VTP06). All output remains opt-in research, not runtime admission.

The formal 3-physical-rank candidate keeps:
- FFN widths **5120 / 4864 / 7424**. Each gate/up rows are sliced;
  down projection uses the same *input column* offsets, resulting in
  partial hidden-5120 outputs to reduce exactly.
- Full attention **6/6/12 Q heads, 1/1/2 KV heads**. Qwen's
  `self_attn.q_proj.weight` is **[12288,5120]**, not [6144,5120]:
  query and output-gate values are interleaved per 256-wide head. The
  three query/gate segments therefore have 3072/3072/6144 output
  rows; K/V have 256/256/512 rows each; O projection has
  1536/1536/3072 input columns. Q/K norm vectors are shared
  256-wide constants, **not** split like heads.
- GDN **4/4/8 key heads, 12/12/24 value heads**, with packed
  `in_proj_qkv` groups Q=[0,2048), K=[2048,4096) and
  V=[4096,10240); R9700 gathers Q=[1024,2048),
  K=[3072,4096), V=[7168,10240). The same three disjoint
  ranges apply to `conv1d.weight`, rather than slicing the
  10,240-channel weight contiguously. Source tensors
  `in_proj_a/b`, `in_proj_z`, `A_log`, `dt_bias`
  and `out_proj` follow value-head ownership, retaining three
  value heads per key head. The GDN shared 128-wide norm is
  replicated. The existing Radiance GDN packer accepts
  Q|K|V|a|b as five separate groups; conversion and the on-device
  state protocol must still implement this per-physical layout.

The card-free source census covers **640 sharded tensor sources**
(192 FFN + 64 full-attention + 384 GDN); 96 of them
(48 GDN QKV + 48 GDN conv) require **non-contiguous source
extraction/repacking**. It checks rank coverage and rejects
overlap, gaps, illegal 128-wide tiles, missing/changed q/gate
rows or GDN coefficients, wrong layer order, and attempts to
overwrite receipts. It does *not* assume a fully qualified model
or assign the vision/MTP/embedding/head allocations.

**Next stage, still open:** implement and independently numerically
validate the native rank-local GDN packed producer, conv/recurrent
state, full Q+gate attention, per-op kernel variant layout,
shared activation/partial-output ABI, vocab/embedding/MTP ownership,
real no-P2P three-GPU collective and physical memory admission.
The legacy Radiance `divide_or_fail(...g.world...)` uniform
head/kernel contract and one-architecture-per-engine restriction
remain blockers. No artificial `--tp 3` workaround or default.

## 2026-11-11 physical collective hardware result

VTP04 now has an explicit one-partial-per-physical-GPU TP3 host-transport mode.
It compiled for gfx1100+gfx1201 and passed exact GPU-result F32 oracles
in **three independent exclusive-lease hardware starts** (including a
reverse-order control). Median across per-start p50 times: physical
TP3 76.35/96.28/627.52 us at 1/4/64 rows versus two logical
R9700 contributions locally combined then sent as one packet:
80.41/102.41/645.70 us. One extra R9700 host packet was still slower.
These are 3 physical endpoints, NOT 3-GPU serving or native quant
GEMMs; the physical TP3 has one source contribution rather than two.
Do not call it an equal-work speedup. See VTP04 for raw receipts.

Stage 1 tensor ownership now has a concrete BF16-header-only checker
(`xtx27_vtp13_ownership.py`, 640 sharded tensor sources, 96 GDN
noncontiguous packed tensors) and stage 5 transport has a synthetic
three-physical-partial oracle. **Still blocked** for runtime on
nonuniform engine-rank dimensions, gfx11/gfx12 mixed kernel registry,
GDN Q|K|V/conv repacking and recurrent state, attention Q+gate pairing,
real three-GPU GEMM-to-collective numerical integration, and measured
KV/VRAM admission. Production and quant defaults remain unchanged.

## 2026-11-11 — VTP13 simultaneous INT8 chain: compile proven, GPU pending

New `tools/lab/radiance/xtx27_vtp13_ffn.hip` is a 3-device
5120/4864/7424 FFN + host-TP3 numerical microharness, called via
`xtx27_vtp13_chain.py`. Unlike VTP12's separately timed native rows,
the three physical GPU streams enqueue the **same dependency chain**
before a single shared barrier, then perform 3 D2H transfers, CPU exact
ordered F32 sum, and 3 H2D broadcasts. A CPU reference independently
recomputes INT8 and BF16-source outputs; each GPU result is checked.
The reference is synthetic BF16-representable data, **not Qwen checkpoint
payloads**; the scalar-warp INT8 kernels are **not native optimized
Radiance dispatch**. No end-to-end speed comparison is justified.

The HIP file compiled under `hipcc -O3 -std=c++17` for gfx1100 and
gfx1201. Card-free planning, source hashing and lease tests passed.
No GPU case ran: the lease was held by another agent doing power tuning
and `rocm-smi --showpids` listed four KFD processes. Start under lease
with `python3 tools/lab/radiance/xtx27_vtp13_chain.py --phase gpu --run
--rows 1,4 --iterations 3`; repeat reversed; use M8/M64 only after
the cheap correctness gate. Receipts go outside Git under
`BC_WORK_ROOT/xtx27-prefill/vtp13-ffn-chain/runs/`.

Still open: checkpoint-BF16 numerical operator proof, native per-arch
W8A8 (and optional common-ABI FP8), two-head full-attention and GDN
Q|K|V/conv/recurrent state, VRAM admission, integrated Radiance TP3,
matched TP2/TP4 and 4/6/8-stream full-model acceptance.
