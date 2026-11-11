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
At initial staging no GPU case ran: the lease was held by another agent doing power tuning
and `rocm-smi --showpids` listed four KFD processes. Start under lease
with `python3 tools/lab/radiance/xtx27_vtp13_chain.py --phase gpu --run
--rows 1,4 --iterations 3`; repeat reversed; use M8/M64 only after
the cheap correctness gate. Receipts go outside Git under
`BC_WORK_ROOT/xtx27-prefill/vtp13-ffn-chain/runs/`.

Still open: checkpoint-BF16 numerical operator proof, native per-arch
W8A8 (and optional common-ABI FP8), two-head full-attention and GDN
Q|K|V/conv/recurrent state, VRAM admission, integrated Radiance TP3,
matched TP2/TP4 and 4/6/8-stream full-model acceptance.

## 2026-11-11 — executed synthetic INT8 FFN/host-TP3 (limited PASS)

After the compile-only staging note, the lease cleared for **two**
bounded fresh starts. First: M1; second: M1/M4/M8. Every GPU returned
the same host-reduced vector and all four cases passed the
independent INT8 CPU oracle (max-abs 0). The three physical devices
actually computed the full gate/up/SwiGLU/down chain concurrently;
this goes beyond the previous separately timed native selftests and
synthetic-only F32 partials. The host transfer path uses 3 D2H and
3 H2D, with one physical worker/context per GPU. Source digest
`c4a9d96253e3c824`, HIP ROCm 7.2.4, checkout `28b116e3e`.

M1 wall median 852.84/863.48 us across two fresh starts; M4 1622.85 us,
M8 2957.80 us (one start each). Per-device M1 503/480/745 us
(XTX0/XTX1/R9700); M4 1525/1429/812 us; M8 2494/2512/1498 us.
The device bottleneck **changes with M** on this simple kernel.
The raw receipts are in VTP12; none uses original model payloads
or optimized Radiance GEMMs. Report **NO QUALIFIED THROUGHPUT GAIN**.

A three-sample p95 reporting defect (p95 accidentally selected the
median) was corrected to nearest rank and recompiled as source digest
`49408233ba43eac2`; its hardware p95 remains unmeasured. A
reverse-order attempt failed closed when the power A/B agent reacquired
the lease. Next: a fresh reverse-order and repeated matched starts,
checkpoint-BF16/native-W8A8 operator-oracle, then real R9700 two-KV-head
attention and GDN key/value/conv/state execution, before engine TP3.

## 2026-11-11 — checkpoint BF16 FFN and GDN packing: real payload evidence

**VTP13 layer-0 FFN hardware proof (leased):** the extended
`xtx27_vtp13_chain.py --checkpoint ${BC_CHECKPOINT}
--layer 0` reads all 3 actual original BF16 gate/up/down source tensors
(3 x 178,257,920 bytes), applies per-output-channel INT8 scales to the
unequal 5120/4864/7424 shards, and computes the full FFN chain
concurrently on 2 gfx1100 + gfx1201. The 3 D2H / CPU FP32 / 3 H2D
physical collective passed the independent quantized CPU oracle with
reported max-abs difference **0**. Against a separate *original BF16
weights, synthetic BF16 activation* reference, M1 RMSE **0.001222** and
max absolute difference **0.03611**; M4 reverse-order RMSE **0.001284**
and max absolute difference **0.03611**. M1 chain p50 **669.17 us**,
p95 **778.13 us** (3 samples); M4 reverse p50 **1932.68 us**,
p95 **2119.10 us** (5 samples). These timings use simple warp-per-output
INT8 kernels, **not native optimized Radiance dispatch**, and say
nothing about real tokens/s or BF16 model-quality thresholds.

Immutable run receipts:
`BC_WORK_ROOT/xtx27-prefill/vtp13-ffn-chain/runs/
20261110T235921961889Z-ecca30e678/` (M1) and
`20261111T000158963464Z-ecca30e678/` (M4 reverse). Each records
source/compiler fingerprints, full selected BF16 tensor payload
SHA-256s, three physical endpoints and CPU/GPU checks.

**VTP13 original-GDN Q|K|V+conv packing (card-free):**
`xtx27_vtp13_gdn_pack.py --checkpoint $BC_CHECKPOINT --phase pack --run --layer 0` produced
real BF16 rank-local files for all 10,240 GDN QKV rows
(104,857,600 source bytes) and convolution
(81,920 source bytes), preserving exact original bytes. XTX0 and
XTX1 each own 2560 Q|K|V channels; R9700 owns 5120. For
both source tensors, reconstructing the original Q|K|V ordered
payload from all three rank files produced a *matching entire
source SHA-256*. Receipt:
`BC_WORK_ROOT/xtx27-prefill/vtp13-gdn-pack/
layer0-20261111T000916481127Z/summary.json`.
**This is pack/layout correctness, not a valid GDN recurrent compute
or entire 48-layer converter.**

New `xtx27_vtp13_heads.py` and `xtx27_vtp13_heads.hip`
compile for gfx1100+gfx1201. The opt-in experiment stages:
(1) actual 24Q/4KV GQA softmax with R9700 12Q/2KV in ONE launch,
(2) original 16K/48V 128-dimensional GDN-shaped persistent
128x128-per-value-head state, with R9700 8K/24V in ONE launch.
Both have independent CPU numerical oracles and measure per-GPU
wall/critical path, but use synthetic BF16-representable
Q/K/V. **GDN deliberately uses a simplified recurrence and omits
original conv, A_log and dt_bias, so full Qwen GDN correctness
remains open.** Initial GPU runs were blocked by another agent's
exclusive power-regression lease; these remain unmeasured, not a pass.

Next: lease-bound attention/GDN head M1/M4 then M8/M64, original BF16
GDN packed-producer/conv/state oracle, true native gfx11/gfx12
INT8 GEMM/FP8 compatible ABI and M4/M8 serving admission; no
FP8 variant loader or production path promoted.

## 2026-11-11 — head-local R9700 attention and simplified GDN: GPU PASS

The opt-in `xtx27_vtp13_heads.py` compiled for gfx1100 and gfx1201
and completed **three separate exclusive-lease starts**, source digest
`db658fc9cc0711ff`, all GPU/CPU numerical oracles passing:
`BC_WORK_ROOT/xtx27-prefill/vtp13-head-ops/runs/`
`20261111T001533333281Z-db658fc9cc` (M1/4; 256-token context),
`20261111T001618569980Z-db658fc9cc` (M8/64; reverse order),
`20261111T001704701578Z-db658fc9cc` (M1/4/8; context 1024).
All tests used three physical contexts/streams and **no P2P**.

**Actual GQA softmax compute**: XTX0 and XTX1 each execute 6 query/1
KV heads; R9700 executes 12 query/2 KV heads **in one GPU kernel launch**,
not two serial layers or separate physical workers. Q/K/V inputs are
deterministic synthetic BF16-representable (no original attention
projection weights). Per-GPU median kernel us:

| Context, M | XTX0 | XTX1 | R9700 | Limiter |
| --- | ---: | ---: | ---: | --- |
| 256, 1 | 480 | 478 | 201 | XTX0 |
| 256, 4 | 475 | 476 | 201 | XTX1 |
| 256, 8 | 478 | 476 | 203 | XTX0 |
| 256, 64 | 492 | 493 | 565 | **R9700** |
| 1024, 1 | 1845 | 1838 | 841 | XTX0 |
| 1024, 4 | 1845 | 1839 | 845 | XTX0 |
| 1024, 8 | 1840 | 1850 | 841 | XTX1 |

Independent CPU softmax maximum absolute error was <=
`6.52e-9` at context 256 and <= `3.38e-9` at context 1024.
**R9700 handles both KV heads concurrently and is not the slow
participant at M1/M4/M8 for this kernel. At M64/256 it IS slower.**
These are naive, synthetic Q/K/V kernels, NOT native gfx11/gfx12
optimized Radiance registry rows, causal full-model attention, or
long-context real KV admission.

**Simplified GDN state update**: 4/4/8 key heads, 12/12/24
value heads in one kernel per physical rank. Each value head owns
a persistent 128x128 F32 state. R9700 state is 1,572,864 bytes
(1.5 MiB) **per synthetic layer and request** in this prototype.
XTXs are 0.75 MiB each. Measured per-GPU median us:

| M | XTX0 | XTX1 | R9700 | Limiter |
| --- | ---: | ---: | ---: | --- |
| 1 | 41.36 | 41.36 | 61.52 | R9700 |
| 4 | 95.80 | 95.36 | 107.92 | R9700 |
| 8 | 165.40 | 165.32 | 172.08 | R9700 |
| 64 | 1118.14 | 1120.35 | 1072.98 | XTX1 |

All simplified state-update oracles passed (max absolute error
<=`7.52e-6`). **Crucial limitation:** this is NOT the exact
Qwen GDN state transition: convolution, learned A_log/dt_bias,
normalisation and beta/z gating are not represented. The per-layer
state figure cannot be used as a verified serving allocation budget.
GDN numerical, topology, exact causal replay and long-sequence
admission remain open. The real BF16 original QKV and conv rank packs
are independently byte-validated but not yet consumed by this
simplified GPU kernel.

Next order: (1) actual original BF16 GDN packed input producer +
convolution/recurrent state math checked against independent
reference, (2) attention source Q/gate head pairing and native
optimized gfx1100/gfx1201 shape dispatch, (3) matched head-local
M64/long-context ABBA and then full TP3-vs-TP4 operators,
(4) only after quality and admission evidence enable engine TP3.
No extrapolation to 262,144-token or 4/6/8-stream real serving.


## 2026-11-11 — next bounded slice: original-BF16 GDN math, GPU pending

New opt-in files tools/lab/radiance/xtx27_vtp14_gdn_exact.{py,hip}
and xtx27_vtp14_gdn_gpu.py stage the Qwen GDN **decode**
mathematics, based on pinned Radiance libref/ref_gdn.cpp and
arch/common/rad_block_gdn.h. Production and vendor code are unchanged.

Original layer-0 BF16 conv1d.weight, A_log, dt_bias and norm.weight
are actually consumed, validated against checkpoint index and
safetensors headers, and payload-SHA256 checked. The three physical
rank convolution packs from VTP13 are checked against their original
recorded hashes. Projected Q/K/V/a/b/z are still deterministic
BF16-representable SYNTHETIC inputs, **not** original in_proj GEMMs.

Card-free reference equations: width-4 causal conv -> SiLU -> BF16
round; L2 q/k norm with 1/sqrt(128) query scale; learned
g=-exp(A_log)*softplus(a+dt_bias), beta=sigmoid(b);
S=exp(g)*S + beta*(v-(exp(g)*S)k)*k^T; o=Sq; and gated
RMSNorm per 128-value head with the checkpoint gain times SiLU(z).
Two independent CPU state orientations and original-vs-repacked
convolution compare all rank outputs and recurrent states.

Real layer-0 M2 and M4 card-free receipts PASS. Convolution comparison
has max absolute 0, M2 state max 8.68e-19, output max 5.21e-18.
M4 immutable receipt under
BC_WORK_ROOT/xtx27-prefill/vtp14-gdn-reference/layer0-20261111T004006431113Z/summary.json.
The independent oracle also exports per-rank FP32 reference binaries
and original learned coefficients with SHA256 provenance.

The actual HIP kernel compiled card-free for gfx1100 and gfx1201;
source digest 304efc77dce14da1. It performs convolution,
normalisation, learned decay/beta delta recurrence and gated norm
simultaneously on three physical GPU streams, comparing every
output with the independently created reference. It does not rely
on P2P. **At initial staging, GPU execution was not yet proven**: an unrelated RHP19
4-GPU calibration owns the exclusive lease and has live KFD processes.

When hardware is free, run from the experimental worktree:
python3 tools/lab/radiance/xtx27_vtp14_gdn_gpu.py --phase gpu --run
--reference "$BC_WORK_ROOT/xtx27-prefill/vtp14-gdn-reference/layer0-20261111T004006431113Z/summary.json"
--pack-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp13-gdn-pack/layer0-20261111T000916481127Z/summary.json"
--iterations 5

The wrapper acquires its own exclusive lease, rejects inherited
GPU masks/foreign processes and writes immutable GPU receipts.
Require rc=0 and independent oracles on every GPU, then repeat
M1 and M8 with separately generated reference receipts.

Still required before engine/quality admission: actual QKV/a/b/z
projection GEMMs, per-sequence warm convolution histories,
exact KKT prefill, state-cache and speculative rollback contracts,
native optimized gfx1100/gfx1201 dispatch, matched TP2/TP4
4/6/8-stream real model throughput and resident KV admission.
The result is NOT full model Qwen GDN, tokens/s or 262k context.


### 2026-11-11 — VTP14 bounded GDN GPU hardware acceptance: PASS

Following the prior card-free staging, the GPU lease became free and
the KFD inventory empty. Three **fresh exclusive-lease GPU starts**
ran the original-BF16-convolution + learned-decay Qwen GDN math on
2x gfx1100 XTX and gfx1201 R9700 concurrently. All three oracles
PASS (rc=0) against independently generated per-rank CPU FP32
outputs; there was no rank serialisation, fourth physical endpoint
or GPU P2P. Source digest: 304efc77dce14da1 (HIP ROCm 7.2.4).

| Rows | End-to-end GPU-stage wall p50 | XTX0 device | XTX1 device | R9700 device | Max absolute GPU-to-CPU error |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 (5 reps) | 123.01 us | 101.04 us | 98.56 us | 78.00 us | 6.99e-10 |
| M4 (3 reps) | 302.79 us | 277.88 us | 276.92 us | 167.08 us | 2.10e-9 |
| M8 (5 reps) | 571.97 us | 545.89 us | 543.36 us | 296.20 us | 6.52e-9 |

Bounded receipts under BC_WORK_ROOT/xtx27-prefill/vtp14-gdn-gpu/runs:
20261111T004542862724Z-304efc77dc (M1),
20261111T004434800289Z-304efc77dc (M4),
20261111T004613287331Z-304efc77dc (M8).
The source, CPU reference, original BF16 packs, coefficient packet,
driver and compiler metadata are recorded/checksummed. Every
run released its lease and exited without residual KFD processes.

**Result:** the R9700 can run 24 value heads + 8 key heads together
with original BF16 convolution and learned Qwen GDN recurrence
faster than either XTX's 12/4 in this specific *unoptimised*
preprojected-input test. This is an operator correctness/relative
latency result only. Actual original QKV/a/b/z GEMMs, input/output
norm/projections, KV/GDN cache admission, full causal prefill KKT,
warm-history continuation, speculative replay and full-model serving
are NOT validated. Do not compare these timings to old simplified
GDN, native Radiance kernels, or tokens/s as if matched.

## 2026-11-11 — VTP15 original BF16 GDN projection/core integration

**Original checkpoint projection proof:** one Qwen3.8-27B GDN layer's
original BF16 input_layernorm, QKV, a, b, z weights have now been
index/header/payload validated, byte-exactly partitioned across the
physical 4/4/8 key and 12/12/24 value head ownership, and computed
concurrently on the two gfx1100 cards plus gfx1201 R9700. Exactly
16,480/16,480 BF16 projection outputs matched an independent CPU
double-accumulation reference, M1 maximum FP32 absolute error 9.54e-7.
GPU wall p50 494.69 us; XTX0 293.68 us, XTX1 288.60 us,
R9700 465.72 us. The R9700 was slower for its larger projection
share with the **simple, unoptimised BF16 GEMV**. Immutable hardware
receipt:
BC_WORK_ROOT/xtx27-prefill/vtp15-gdn-gpu/runs/20261111T010538697678Z-1f43875bcd/summary.json.

**Original-checkpoint GDN CPU integration:** the independent CPU
reference now consumes the *real BF16 checkpoint projection outputs*,
original BF16 convolution, learned A_log/dt_bias, and norm. It
calculates width-4 causal convolution -> BF16 SiLU -> L2-normalised
Q/K -> learned gated-delta recurrence -> gated RMSNorm with two
independent global/rank state layouts; validates the source and
projection byte hashes and rank ownership. M1 and M4 pass.
M1 global/rank recurrent-output max abs 4.44e-16; M4 2.22e-15;
convolution max difference 0. Immutable receipts:
BC_WORK_ROOT/xtx27-prefill/vtp15-gdn-core/layer0-20261111T011811618321Z/summary.json
BC_WORK_ROOT/xtx27-prefill/vtp15-gdn-core/layer0-20261111T012015196547Z/summary.json

**Integrated HIP kernel compiled, hardware not yet run:** opt-in
xtx27_vtp15_gdn_integrated.hip + xtx27_vtp15_gdn_integrated_gpu.py
perform on *each* GPU in one dependent stream real checkpoint BF16
QKV/a/b/z GEMVs -> original BF16 convolution -> Q/K norm ->
learned delta recurrence -> gated RMSNorm. All three physical streams
are launched before the barrier; no P2P or serial layers. Native
compile passed for gfx1100+gfx1201 with source digest
ed3dd929b34571fc.... The runner validates all independent CPU
reference and source-receipt hashes, acquires its own exclusive GPU
lease and refuses pre-existing KFD processes. A competing agent
held a 6-stream/cold-TTFT lease at the last check, so the integrated
GPU correctness gate is explicitly **PENDING**, not PASS.

First acceptance command (only once GPU lease is available and KFD
inventory empty), from the experimental worktree:

    python3 tools/lab/radiance/xtx27_vtp15_gdn_integrated_gpu.py --phase gpu --run --iterations 3 \
      --weights-receipt ${BC_WORK_ROOT}/xtx27-prefill/vtp15-gdn-projection/layer0-20261111T005819109215Z/summary.json \
      --projection-receipt ${BC_WORK_ROOT}/xtx27-prefill/vtp15-gdn-ref/layer0-20261111T010143705406Z/summary.json \
      --core-receipt ${BC_WORK_ROOT}/xtx27-prefill/vtp15-gdn-core/layer0-20261111T011811618321Z/summary.json \
      --learned-receipt ${BC_WORK_ROOT}/xtx27-prefill/vtp14-gdn-reference/layer0-20261111T003313797426Z/summary.json

Then the M4 matched controls: source projection pack
vtp15-gdn-projection/layer0-20261111T011938129228Z,
CPU projection reference vtp15-gdn-ref/layer0-20261111T011953188932Z,
CPU GDN core vtp15-gdn-core/layer0-20261111T012015196547Z.

**Remaining before any model serving/speed claim:** block output
projection + residual, exact causal state/conv cache continuation,
Qwen chunk-prefill KKT, full-attention, all-layer model loading,
native optimised gfx1100/gfx1201 kernels, speculative rollback,
BF16-relative model quality, and matched dual-XTX TP2 vs TP3
real 4/6/8-stream throughput and resident KV/GDN memory admission.
The incoming hidden activation is still synthetic. Do not interpret
the measured microkernel speed as 27B model throughput.


## 2026-11-11 — integrated GDN hardware M1 PASS; VTP16 output and residual card-free PASS

**VTP15 integrated three-physical-GPU GDN original BF16 M1 PASS:**
one device stream per XTX0/XTX1/R9700, real original checkpoint BF16
QKV/a/b/z GEMVs -> original BF16 causal convolution -> L2 Q/K
normalization -> learned A_log/dt_bias delta recurrence -> gated
RMSNorm, verified against independent CPU references at each stage.
M1 GPU p50 wall 416.49 us; XTX0 387.04 us; XTX1 383.28 us;
R9700 336.40 us. Max GEMV-to-reference FP32 error 9.54e-7;
max GDN recurrent output absolute error 3.58e-7. All PASS.
GPU immutable receipt: BC_WORK_ROOT/xtx27-prefill/vtp15-integrated/
runs/20261111T014815593848Z-ed3dd929b3/summary.json.
The M4 hardware follow-up was refused because another agent
acquired the exclusive lease, and is still PENDING.

**VTP16 GDN output projection output and residual CPU proof:**
original checkpoint layer-0 out_proj.weight BF16 [5120,6144],
original source 62,914,560 bytes, is validated and byte-exactly
reconstructed from physical input-column shards [5120,1536] /
[5120,1536] / [5120,3072]. The BF16 rank-local input tensor
is derived from the independent original-weight GDN CPU-core
reference, reordered [token, head, 128] and BF16-rounded once.
A separate C++ double-accumulation control computes both the
unsharded projection and the three rank-local partials, then
performs explicitly ordered FP32 host reduction, BF16 projection
rounding and BF16 residual add. No GPU is needed for this proof.

| Original BF16 output and residual | M1 | M4 |
| --- | ---: | ---: |
| Max unsharded vs 3-rank projection FP32 abs | 1.49e-8 | 5.96e-8 |
| Exact BF16 output-projection elements | 5120/5120 | 20480/20480 |
| Exact BF16 post-residual elements | 5120/5120 | 20480/20480 |

Original M1 source/output and CPU receipts:
BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output/layer0-20261111T014452115048Z/summary.json
BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output-ref/layer0-20261111T014742582287Z/summary.json.
M4:
BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output/layer0-20261111T014909377366Z/summary.json
BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output-ref/layer0-20261111T014923805172Z/summary.json.

**Caveats:** deterministic synthetic incoming hidden activation,
not real prior-layer output. The core GPU M1 proof ends before
out_proj and all_reduce; the VTP16 projection/allreduce/residual
is currently an independently tested card-free oracle, not
on-GPU or native Radiance runtime. Model-wide 64-layer GPU loading,
long-context cache, prefill KKT, speculative rollback, real 4/6/8
stream throughput and resident context are all open. No
production default or vendor changes.


### 2026-11-11 — VTP16 original-BF16 out-proj three-GPU acceptance staged

Opt-in xtx27_vtp16_gdn_output.hip and
xtx27_vtp16_gdn_output_gpu.py are now compiled for gfx1100+gfx1201
(source digest 0e579a3508054a90...). One physical GPU context
per XTX0/XTX1/R9700 loads only its own exact-original BF16
5120x1536/1536/3072 output-projection weight columns and
the independently computed BF16 GDN-core output slice, launches
all three physical GEMVs before one barrier, sends three
[M,5120] F32 partials D2H, sums in explicit physical-rank order,
BF16-rounds and adds the original BF16 residual, then broadcasts
the result H2D to every GPU. All three output buffers are
compared against the host result; every partial, reduced F32
vector and final BF16 vector is numerically compared against
the independent VTP16 original-BF16 whole-matrix CPU reference.
This is pageable host staging; no P2P, no claimed production
overlap, no GPU-layer-wide integration with VTP15 core yet.

**Hardware status:** GPU acceptance PENDING, because another
agent owns the exclusive 4-GPU lease for ongoing Flash-Next
serving experiments. The harness is opt-in, obtains a fresh
exclusive agent-lease, rejects pre-existing KFD processes and
writes source-digested immutable receipt files. No attempt
to run inside the competing lease.

M1 when lease is free (using original CPU M1 oracle regenerated
after the lint-only writer change):

    python3 tools/lab/radiance/xtx27_vtp16_gdn_output_gpu.py --phase gpu --run --iterations 5 \
      --packet-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output/layer0-20261111T014452115048Z/summary.json" \
      --reference-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output-ref/layer0-20261111T015557796594Z/summary.json"

For M4 substitute the matching original output packet at
vtp16-gdn-output/layer0-20261111T014909377366Z/summary.json
and CPU oracle receipt at
vtp16-gdn-output-ref/layer0-20261111T014923805172Z/summary.json.
Require rc=0, all three original-weight partials, host FP32
reduction, BF16 residual and H2D broadcast numerical oracles
PASS, no residual KFD process, per-GPU/wall timings recorded.
Do not confuse this separated output-stage proof with the
already passed VTP15 on-device input-to-recurrence M1 kernel,
or claim all operations share a single live engine context.

**Next graph integration:** feed each VTP15 GDN GPU recurrent
BF16 output directly into this GPU output-GEMV and host
collective, then attach BF16 residual. That makes the
whole core block one GPU dependency graph; the subsequent
gates remain original long-token recurrence history, exact
prefill KKT, full attention, all 64 layers, native optimized
quant GEMMs, serving/quality, and real 4/6/8-stream and
resident context comparisons against TP2 and fused TP4.


### 2026-11-11 — VTP16 hardware PASS and VTP17 pinned asynchronous host transport optimisation

The previously card-free VTP16 original-BF16 GDN output-projection
hardware proof has now passed M1 and M4 under fresh exclusive
three-physical-GPU leases. Both use actual original BF16
5120x6144 output projection weights column-sharded 1536/1536/3072,
and check each GPU partial, ordered F32 D2H host sum, BF16 residual
addition and H2D broadcast against the independent whole-matrix
reference. All 5120/20480 BF16 final outputs matched exactly,
with no P2P and clean process/lease release.

Opt-in VTP17 uses the SAME original weights, oracle, kernel geometry,
physical ranks and output protocol, but exchanges pageable host
staging for persistent HIP portable pinned partial/final buffers.
It queues D2H transfers behind each GPU GEMV on each GPU stream,
waits only after all three transfers are enqueued, performs the
same ordered host F32 combine, then queues H2D BF16 broadcast.
Actual M1 and M4 leases both completed with numerical PASS.

| Original BF16 output stage | VTP16 pageable p50 | VTP17 pinned async p50 | Indicative stage reduction |
| --- | ---: | ---: | ---: |
| M1 | 214.53 us | 193.40 us | 9.8% |
| M4 | 309.59 us | 246.94 us | 20.2% |

VTP17 M1 measured XTX0 kernel 66.44 us, XTX1 66.52 us,
R9700 38.24 us, first-D2H-complete stage p50 98.83 us,
host-reduce stage 15.89 us and H2D stage 78.16 us.
M4 device kernels 173.60/172.24/113.80 us; first-D2H-complete
stage 211.21 us, host reduce 11.60 us, H2D stage 20.90 us.
The first stage includes kernel execution, enqueues and transfers:
**never label it PCIe-only D2H time**. The 3 stage medians are
independent marginal p50 values and do not add exactly to
the total wall median. Timings are separated fresh-lease starts,
not balanced/thermal-matched ABBA, and not model tok/s.
Treat speedup as *preliminary*, not a production configuration.

Immutable VTP16 GPU receipts:
BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output-gpu/runs/
20261111T020106329323Z-0e579a3508 (M1) and
20261111T020201155538Z-0e579a3508 (M4).
VTP17 pinned GPU receipts:
BC_WORK_ROOT/xtx27-prefill/vtp17-gdn-output-pinned/runs/
20261111T020414350676Z-950d69e809 (M1) and
20261111T020439712421Z-950d69e809 (M4).

Acceptance still needed before claiming a robust improvement:
same-session ABBA or BAAB alternating VTP16/VTP17 M1 and M4
with identical GPU inventory, no competing leases, equal
kernel build, power settings and deterministic original weight
inputs. Validate all full CPU oracles every arm and measure
per-stage p95 as well as p50. VTP17 is opt-in research only,
not a Radiance native collective or full inference.


### 2026-11-11 — VTP18 proposed row-fused BF16 GEMV reuse, GPU acceptance pending

A second strictly opt-in optimisation prototype
(VTP18 xtx27_vtp18_gdn_output_rowreuse.hip and driver)
builds on the already hardware-validated VTP17 pinned
asynchronous host collective. The non-optimised VTP16/VTP17
output GEMV launches separate per-row warps, rereading each
original BF16 [5120,K_rank] weight shard for every input row.
At M4, that is four matrix reads per physical rank, even
though weights do not change across those rows. The VTP18
template specializes M=2, M=4 and M=8. Each output-neuron
warp now loads an original BF16 weight once, multiplies it
against its 2/4/8 BF16 input rows and accumulates per-row
FP32 results in registers, with static unrolled loops.
M1, M3, M5, M6 and M7 deliberately keep the baseline kernel.

**Hypothesis, NOT an observed speedup:** for M4/M8 this can
reduce repeated output-GEMV weight loads by up to the number
of batch rows and launches only one grid per output matrix.
The actual speedup may be smaller or negative due to register
pressure, fewer independent workgroups, GPU cache reuse
and reduced occupancy. The original three-GPU 1536/1536/3072
column partition, independent original BF16 CPU oracle,
FP32 ordered host sum, BF16 residual and pinned-async
broadcast are unchanged.

HIP compiler acceptance on gfx1100+gfx1201 PASSED;
source digest a12e0015a14d954c..., all card-free
tests pass. **No GPU run has been conducted yet**
because another agent holds the exclusive lease.

After lease is free, run M4 and M1 with same original
checkpoint/output references and compare against VTP17
in a balanced ABBA or BAAB sequence. M4 VTP18 example:

    python3 tools/lab/radiance/xtx27_vtp18_gdn_output_rowreuse_gpu.py --phase gpu --run --iterations 9 \
      --packet-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output/layer0-20261111T014909377366Z/summary.json" \
      --reference-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output-ref/layer0-20261111T014923805172Z/summary.json"

Require all three per-rank original-weight outputs and final
BF16 residual to agree against the independent source-hashed
CPU oracle. Measure M1/M4/M8 p50 and p95, device kernel
timings and all host stages. Reject or revert the variant
if M4 or M8 fails numerical or latency gates; never
enable either VTP17 or VTP18 as a production default.


### 2026-11-11 — VTP19 same-lease three-way A/B/C comparison scaffold

The VTP19 runner xtx27_vtp19_output_abccba.py is the next gate
for deciding whether VTP17 pinned transfers and VTP18
batch-row weight reuse truly improve the original-BF16 GDN
output stage. It compiles and pins all three exact sources
from a single compiler build manifest: A=VTP16 pageable,
B=VTP17 pinned asynchronous, C=VTP18 row-fused+pinned.
For M1 and/or M4 original checkpoint output input packets
it generates a strictly ordered A/B/C/C/B/A palindrome,
using unique receipt filenames per arm under ONE fresh
exclusive GPU lease. Each process has the same three
physical GPUs, original source pack, BF16 residual, held
CPU reference and 5120-wide F32 host-reduced output.
Every arm must pass the original unsharded+residual CPU
oracle, including exactly equal BF16 final outputs;
failed/partial results remain incomplete and refuse
performance admission.

The card-free all-three-variant multiarchitecture
compile PASSED for gfx1100+gfx1201 (digest
946c6ad1f88fd5d6...), and the five VTP19 plan,
ordering, missing-source, canonical output parsing,
corruption and fail-closed checks PASS. Actual
same-session hardware A/B/C is **NOT yet run**:
another owner holds the GPU lease. VTP18 still has
no hardware performance or numerical acceptance.

Once the lease is free, run from the experimental worktree:

    python3 tools/lab/radiance/xtx27_vtp19_output_abccba.py --phase gpu --run --iterations 9 \
      --packet-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output/layer0-20261111T014452115048Z/summary.json" \
      --reference-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output-ref/layer0-20261111T015557796594Z/summary.json" \
      --packet-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output/layer0-20261111T014909377366Z/summary.json" \
      --reference-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output-ref/layer0-20261111T014923805172Z/summary.json"

The final receipt parses each arm's wall p50/p95,
individual GPU kernel p50, max output/reduction
numerical errors and exact BF16 residual count.
It calculates per-variant median-of-two-arm wall
latencies and indicative percentage changes. Power,
thermal state and production graph execution are not
enforced: separate fresh process starts within one
lease only partially control clock drift. Compare
the A/C numerical outputs against their original
independent CPU oracle before any promotion.

The acceleration objective remains real 4/6/8-stream
Qwen3.8-27B serving on three physical GPUs. These
standalone original-weight output-GEMV/host-collective
experiments are NOT native mixed-architecture
quantised Radiance kernels or end-to-end tok/s.

### 2026-11-11 — VTP19 blocked by power sweep; VTP20 full original BF16 GDN block staged

**VTP19 benchmark:** The stale manual lease was released and a new runner-managed lease attempted, but bigcherry-1 acquired the exclusive GPUs for an XTX power sweep first. The VTP19 request refused safely and produced no GPU benchmark receipt. The balanced original/pageable, pinned async and row-fused A/B/C/C/B/A GPU comparison is still pending, and VTP18 has no measured speedup.

**VTP20 connected GDN block, staged:** The previously separate original BF16 input projection-to-GDN-core and GDN output projection now have one experimental three-physical-GPU execution chain, under tools/lab/radiance/xtx27_vtp20_gdn_full_block.hip and xtx27_vtp20_gdn_full_block_gpu.py. Each GPU performs original QKV/a/b/z GEMVs, 4-tap depthwise convolution, L2 Q/K normalization, original learned delta recurrence and gating, then *on-device* [head, token, 128] F32 to [token, head, 128] BF16 transpose/rounding and original BF16 out_proj column-shard GEMV (1536/1536/3072 input channels). The three streams are enqueued before the barrier; GPU F32 partials move to portable pinned host memory, combine in ordered XTX0->XTX1->R9700 F32, BF16-round and add the original BF16 residual, and broadcast to each GPU.

The runner matches all original VTP15 and VTP16 checkpoint/source/output CPU receipts by SHA256, layer, and M1/M4 row count. GPU checks include all 12 projection results, original GDN core state/output, three out_proj F32 partials, host reduced output, final BF16 residual and each GPU's broadcast. Every input uses original checkpoint BF16 weights; the incoming hidden activation is synthetic. It fails closed on any missing or inconsistent reference.

**Card-free gates PASS:** gfx1100 + gfx1201 compilation, read-only M1 plan, all 123 VTP-focused card-free tests, Ruff and Git diff checks. **VTP20 GPU execution PENDING** (the power sweep owns the lease). No text-generation, 64-layer, cache-continuation, long-context, native quantisation, real throughput or model-quality claim.

M1 hardware acceptance once the exclusive GPU lease is genuinely free, in the experimental worktree:

    python3 tools/lab/radiance/xtx27_vtp20_gdn_full_block_gpu.py --phase gpu --run --iterations 5 \
      --weights-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp15-gdn-projection/layer0-20261111T005819109215Z/summary.json" \
      --projection-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp15-gdn-ref/layer0-20261111T010143705406Z/summary.json" \
      --core-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp15-gdn-core/layer0-20261111T011811618321Z/summary.json" \
      --learned-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp14-gdn-reference/layer0-20261111T003313797426Z/summary.json" \
      --output-packet "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output/layer0-20261111T014452115048Z/summary.json" \
      --output-reference "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output-ref/layer0-20261111T015557796594Z/summary.json"

M4 substitutes the VTP15 weight pack layer0-20261111T011938129228Z, projection CPU reference layer0-20261111T011953188932Z, core CPU reference layer0-20261111T012015196547Z, VTP16 output packet layer0-20261111T014909377366Z and VTP16 output CPU oracle layer0-20261111T014923805172Z; learned-parameter receipt remains the same.

**Priority next:** First run VTP19 A/B/C once a free lease is available, then VTP20 M1 and M4 full-block numerical acceptance. Afterward stop isolated microbenchmarks: implement exact GDN/conv cache continuation and chunk-prefill KKT, full attention and model-wide 64-layer loading, real TP2/TP3 C4/C6/C8 text throughput and resident context tests. Only a full end-to-end run qualifies speedup.


### 2026-11-11 — VTP21 original-BF16 GDN conv/recurrent state continuation CPU PASS

A new strictly card-free reference, tools/lab/radiance/xtx27_vtp21_gdn_state_continuation.py, consumes the independently validated original Qwen3.8-27B layer-0 BF16 projected QKV/a/b/z, original BF16 causal conv1d weights, learned A_log/dt_bias/norm gains, and the previous full M4 original-checkpoint CPU GDN oracle. On all three physical TP3 partitions (4/4/8 key heads, 12/12/24 value heads), it compares monolithic four-token execution with 1+3, 2+2, 3+1, and 1+1+1+1 token chunks. Every boundary round-trips the 3-row causal convolution BF16 projection history and a key-dimension-major, value-dimension-minor GDN recurrent state serialized as little-endian F64. This is a CPU reference-state handoff: the GPU state needs an explicit separate FP32 residency/transport contract.

**PASS:** bit-identical final SHA256 hashes of the recurrent state and causal-convolution history on all five partitions of four tokens and all three cards; the original independent monolithic M4 output agrees to:

| Physical rank | Maximum output difference |
| --- | ---: |
| XTX0 | 7.72e-8 |
| XTX1 | 8.74e-8 |
| R9700 | 5.24e-8 |

A deliberate state reset at token four on the same projected input produces nontrivially different output (max abs differences XTX0 0.809, XTX1 0.770, R9700 0.319), proving the state carry is actually needed. Malformed snapshot geometry, nonfinite state, incomplete chunks and broken original provenance refuse.

Receipt: BC_WORK_ROOT/xtx27-prefill/vtp21-gdn-state-continuation/layer0-20261111T033702426695Z/summary.json.

**Limit:** synthetic incoming hidden, not the previous real model layer. CPU F64 cached state is the exact oracle, NOT the actual GPU FP32 state format, not decode-speed hardware acceptance, chunk-KKT prefill, KV residency, cancellation/rollback, model-wide 64 layers or actual tokens/sec. GPU acceptance must prove the same chunk boundaries with resident on-GPU FP32 state, then connect real previous-layer output and full attention.

VTP19/VTP20 GPU acceptance remains blocked by another agent's exclusive XTX power sweep. Do not launch into it or claim a measured VTP18 row-reuse speedup.


### 2026-11-11 — VTP19 balanced original-BF16 output-stage GPU ABCCBA PASS

The VTP19 runner acquired a clean exclusive GPU lease and completed
all 12 arms of the source-hashed balanced original/pageable (A),
pinned-async (B) and row-reuse+pinned (C) comparison at M1 and M4.
All arms passed original-BF16 output-projection numerical checks,
including exact BF16 residual output and identical broadcasts to
all three GPUs. The run captured read-only before/after GPU clock,
temperature and power observations. Receipt:
$BC_WORK_ROOT/xtx27-prefill/vtp19-original-BF16-output-abccba/runs/
20261111T041059364071Z-946c6ad1f8/summary.json.

Each value below is the median of two per-arm p50 wall latencies from
one lease, with A/B/C/C/B/A order for each batch size:

| Output-stage p50 (us) | Pageable A | Pinned B | Row-reuse+pinned C |
| --- | ---: | ---: | ---: |
| M1 | 233.34 | 188.745 | 188.96 |
| M4 | 310.67 | 242.08 | **146.50** |

M4 C saves **52.84%** relative to original A and **39.48%**
relative to pinned B. M1 pinned B saves **19.11%**; row-reuse
falls back to the baseline GEMV at M1 and does not improve speed.
M1 row-reuse p95 across arms is **340.05 us** versus pinned
192.73 us, so do not promote the row-reuse variant at M1.
M4 row-reuse p95 155.545 us versus pinned 266.415 us.

Candidate experimental selection: pinned asynchronous host transfer
for all row counts; row-fused weight reuse only at supported M2/4/8
after numerical parity, never assumed faster for M1.
This is not a production default, native Radiance kernel or full
model tok/s result. Power profile and clocks were not enforced
during individual runs, and adjacent XTX power sweeps may matter.

### 2026-11-11 — VTP22 original-BF16 full-attention layer-3 CPU oracle PASS

The original Qwen3.8-27B has 48 GDN and 16 full-attention layers.
VTP22 progresses to a missing full-model operator rather than
continuing isolated GDN kernels.

- xtx27_vtp22_attention_pack.py reads the REAL original
  BF16 checkpoint layer-3 full-attention tensors with validated
  safetensors shapes, offsets, config/index and SHA256. It
  packs interleaved query/output-gate source rows into Q/G
  3072/3072/6144, K/V 256/256/512, and O projection
  columns 1536/1536/3072, reconstructing all original source
  tensors byte-for-byte. Original 256-wide Q/K norms and
  layer-3 input RMSNorm are replicated, not head-sliced.
  M1 and M4 original payload packs PASS.
- xtx27_vtp22_attention_ref.cpp and the Python CPU oracle
  use original BF16 Q/K/V/O weights, head-local zero-centred
  RMSNorm, first-64-dim partial RoPE at text-only position ids,
  causal grouped-query 24Q:4KV softmax, per-head sigmoid output
  gating, original BF16 output-projection column shards,
  XTX0/XTX1/R9700 ordered F32 sum, BF16 residual.
  Both real-checkpoint M1 and M4 CPU reference runs PASS.
  This is a standalone numerical interpretation with explicit
  BF16 rounding boundaries; full Transformers model numerical
  parity is not claimed.

Original weight packages:
$BC_WORK_ROOT/xtx27-prefill/vtp22-original-full-attention/
layer3-m1-20261111T041209305980Z/summary.json,
layer3-m4-20261111T041210935826Z/summary.json.

CPU original full-attention oracle receipts:
$BC_WORK_ROOT/xtx27-prefill/vtp22-original-full-attention-cpu/
layer3-m1-20261111T041245628121Z/summary.json,
layer3-m4-20261111T041247681924Z/summary.json.

All input activations remain explicitly deterministic synthetic BF16
hidden states, not preceding-layer output. No GPU full-attention
acceptance yet, and no native KV-cache implementation or real text
generation. Next operator work: validate the physical full-attention
GPU path against these original-source CPU outputs and integrate
GDN/attention/FFN with embedding, head and 64-layer residual
handoff. VTP20 connected GDN block GPU acceptance remains
pending due to another agent's exclusive power sweep lease.


### 2026-11-11 — VTP23 original-BF16 full-attention GPU chain staged

VTP23 is the actual original-checkpoint full-attention GPU
counterpart to VTP22's layer-3 M1/M4 CPU reference, moving
beyond the earlier synthetic VTP13 head-kernel microbenchmark.

tools/lab/radiance/xtx27_vtp23_attention_original.hip runs on
one stream on each *physical* XTX0/XTX1/R9700. All original
BF16 Q/G, K, V and O projection tensors use the byte-exact
checkpoint weight packs from VTP22. The chain performs:

- Checkpoint BF16 Q/G (6/6/12 interleaved query/gate heads),
  K/V (1/1/2 original KV heads) GEMVs with the same
  original BF16 weighted pre-RMSNorm input on every GPU
- Original replicated Q/K 256-wide zero-centred RMSNorm,
  first-64-dimension partial RoPE for text-only positions,
  causal six-query-per-KV grouped softmax with 12 R9700
  query heads sharing TWO local KV groups in one launch
- Actual sigmoid Q-projection output gate, original
  BF16 O projection input-column shards 1536/1536/3072,
  one F32 partial per physical GPU, pinned asynchronous D2H
  host reduction in XTX0/XTX1/R9700 order, BF16 residual
  addition and broadcast to all three devices

The GPU runner xtx27_vtp23_attention_gpu.py enforces immutable
real checkpoint/source payload fingerprints and independent
VTP22 source-hashed CPU-oracle receipts before acquiring
a fresh exclusive GPU lease. It compares stage outputs
(Q/G raw, K/V raw, normalized/rotated Q/K, causal/gated
attention, three O partials, reduced output and final
BF16 residual) against original BF16 CPU tensors. It
reports exact BF16 final residual matches, stage maximum
errors, per-GPU and full wall p50/p95 and refuses on any
numerical/lease failure.

The source **compiles for gfx1100 and gfx1201**, digest
bf8430b1407dec2e..., and card-free validation plus focused
VTP tests pass. **GPU hardware acceptance is still pending**
because the other agent owns the exclusive XTX power-sweep
lease. VTP23 has NOT yet demonstrated real model output,
correct physical KV retention, streaming text tokens,
native quant kernels or full-model superiority over TP2.

M1 GPU acceptance when owner releases the GPUs:

    python3 tools/lab/radiance/xtx27_vtp23_attention_gpu.py --phase gpu --run --iterations 5 \
      --packet-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp22-original-full-attention/layer3-m1-20261111T041209305980Z/summary.json" \
      --cpu-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp22-original-full-attention-cpu/layer3-m1-20261111T041245628121Z/summary.json"

For M4 use layer3-m4-20261111T041210935826Z packet and
layer3-m4-20261111T041247681924Z CPU receipt.
Both need numeric GPU PASS plus source receipt; no
model-wide performance claims from this stage.

After original full GDN block VTP20 and original
full attention VTP23 pass GPU oracles, the next
work is the actual model graph/engine: FFN, residual
handoff, embed/head, persistent GDN/conv and KV
state across decode/prefill, 64-layer original
model and 4/6/8 stream TP2/TP3 tok/s. Do not
enable experimental kernels as production defaults.


### 2026-11-11 — VTP20/VTP23 original-BF16 physical-TP3 GPU acceptance (hardware measured)

The previously staged full connected GDN (VTP20) and full
causal-attention (VTP23) single-layer GPU paths have **now run**
on the two gfx1100 XTX GPUs and one gfx1201 R9700 under
separate exclusive, runner-managed leases. Each lease was
cleanly released, with no remaining KFD process at the
post-run inspection. All four single-layer cases passed their
original-checkpoint, independent CPU-referenced **numeric
tolerance** gates. The input hidden tensors are deterministic
and synthetic, not preceding-layer model activations.
Original BF16 checkpoint projection weights were used.

| Test | Wall p50 | Wall p95 | Final exact BF16 | Largest final BF16 abs error |
| --- | ---: | ---: | ---: | ---: |
| VTP20 connected GDN M1 | 647.84 us | 772.49 us | 5,120/5,120 | 0 |
| VTP20 connected GDN M4 | 1,110.39 us | 1,124.43 us | 20,371/20,480 | 0.001953125 |
| VTP23 original full attention M1 | 471.03 us | 674.93 us | 5,119/5,120 | 0.0009765625 |
| VTP23 original full attention M4 | 745.36 us | 853.02 us | 20,186/20,480 | 0.015625 |

**These PASS results are not all bit-exact.** The GDN M4
109 differing final BF16 outputs and full-attention M1/M4
1/294 differing outputs remain within the *existing*
stage-specific numerical tolerances; the GPU reference
must not be described as fully BF16-bit-exact. In particular
the full-attention M4 maximum observed final BF16
absolute discrepancy was 0.015625. The VTP23 validator's
bounds are relatively generous; promotion requires
quality checks and independent full-framework parity.

Immutable hardware receipts:

- VTP20 M1: $BC_WORK_ROOT/xtx27-prefill/vtp20-full-gdn-block/runs/20261111T060113662007Z-f383946315/summary.json
- VTP20 M4: $BC_WORK_ROOT/xtx27-prefill/vtp20-full-gdn-block/runs/20261111T060157393289Z-f383946315/summary.json
- VTP23 M1: $BC_WORK_ROOT/xtx27-prefill/vtp23-original-attention-gpu/runs/20261111T060218973632Z-bf8430b140/summary.json
- VTP23 M4: $BC_WORK_ROOT/xtx27-prefill/vtp23-original-attention-gpu/runs/20261111T060240861624Z-bf8430b140/summary.json

Each receipt case has rc=0 and oracle_pass=true.
Stdout gives p50/p95, per-device times, maximum numeric
differences, exact value counts and explicit limitations.
No GPU power cap, clock or undervolt was applied by
these runners. A power sweep preceded these runs; wall
timings are single-run observations and not matched
stock-clock throughput benchmarks.

The **next actual milestone** is no longer an independent
GDN or full-attention kernel oracle. It is a model-wide
physical-TP3 execution path wiring FFN plus 48 GDN
and 16 full-attention operators with per-layer
pre-/post-RMSNorm, residual handoff, real token embeddings,
output head/logits and token sampling; persistent
per-request recurrent/conv and KV state; and measured
per-GPU VRAM admission for 4/6/8 streams and long context.
Only real end-to-end text generation, quality replay and
matched TP2/TP3 throughput can qualify a production option.
Neither VTP20 nor VTP23 constitutes full model inference.


### 2026-11-11 — VTP24/VTP25: original-BF16 FFN after GDN, plus full-64-layer ownership lower bound (VTP26)

The next **model execution blocker**, rather than another independent
GDN microbenchmark, is full original-weight FFN and real residual handoff.
VTP24 has packaged and executed the *original checkpoint* layer-0
BF16 FFN gate/up/down using physical rank widths **5120/4864/7424**
and original post_attention_layernorm. Its FFN input is the
source-hashed, original-GDN CPU reference **post-residual** from
VTP16 (M1 and M4), not a new synthetic generated activation.
This verifies actual residual -> norm -> FFN placement but **not yet
live GPU-to-GPU GDN/FFN forwarding**.

VTP24 files:
- tools/lab/radiance/xtx27_vtp24_ffn_pack.py: verified original
  safetensors source offsets, original row-sharded gate/up and
  input-column-sharded down; exact byte reconstruction of all
  17408-wide original FFN source tensors. Original model layer-0
  post-attention weighted RMSNorm, original VTP16 GDN residual
  source verified by hashes. The output is immutable.
- tools/lab/radiance/xtx27_vtp24_ffn_ref.cpp and
  xtx27_vtp24_ffn_oracle.py: standalone C++ double-accumulation
  original BF16 gate/up, BF16 SwiGLU and down projections,
  ordered 3-device F32 reduction, BF16 residual and the resulting
  **complete original layer-0 post-FFN hidden**, suitable as
  layer-1 input. This CPU correctness reference is deliberately
  separate from the GPU implementation.

M1 source / CPU receipts:
- $BC_WORK_ROOT/xtx27-prefill/vtp24-original-ffn/layer0-20261111T061700090195Z/summary.json
- $BC_WORK_ROOT/xtx27-prefill/vtp24-original-ffn-cpu/layer0-m1-20261111T061945282150Z/summary.json

M4 source / CPU receipts:
- $BC_WORK_ROOT/xtx27-prefill/vtp24-original-ffn/layer0-20261111T061946347706Z/summary.json
- $BC_WORK_ROOT/xtx27-prefill/vtp24-original-ffn-cpu/layer0-m4-20261111T062028159412Z/summary.json

Both original-weight CPU references pass their declared source and
execution checks. **Not a full-model-framework numerical parity
qualification**; BF16 intermediate rounding boundaries remain
independently chosen for this research oracle.

VTP25 is a distinct new three-physical-GPU **original BF16 FFN**
implementation under xtx27_vtp25_ffn_original.hip, supervised
by xtx27_vtp25_ffn_gpu.py. Original checkpoint
gate/up GEMVs -> BF16 SwiGLU -> original down projection,
parallel one stream on each XTX0/XTX1/R9700 -> pinned async
D2H and ordered F32 host reduction -> original BF16 residual
and three-device broadcast. Every stage is checked
against VTP24's source-hashed CPU oracle. Multiarch HIP
compilation gfx1100+gfx1201 PASS and card-free checks
PASS. **VTP25 has not yet run on GPU**, because the other
agent's exclusive GPU lease is held for power/text identity.

VTP25 hardware acceptance once GPUs are free:

    python3 tools/lab/radiance/xtx27_vtp25_ffn_gpu.py --phase gpu --run --iterations 5 \
      --packet-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp24-original-ffn/layer0-20261111T061700090195Z/summary.json" \
      --cpu-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp24-original-ffn-cpu/layer0-m1-20261111T061945282150Z/summary.json"

M4: replace the two receipts with
layer0-20261111T061946347706Z/summary.json and
layer0-m4-20261111T062028159412Z/summary.json.

VTP26 exact-original-BF16 **64-layer tensor and state resource
ledger** is now card-free:
tools/lab/radiance/xtx27_vtp26_full_model_inventory.py.
It consumes the original 18 safetensors headers, validates all
640 sharded source tensors across 48 GDN, 16 gated causal attention
and 64 5120/4864/7424 FFN blocks, including the real replicated
input/post-attention and Q/K/GDN norms. It accounts for original
source BF16 weights and per-request recurrent state, 3-row
causal convolution history, and full causal attention KV
retention. The first text-only correctness path leaves the
original embeddings, full language head and final norm in
host RAM (about 5.09 GB before workspace); visual/MTP
tensors are intentionally excluded and separately counted.

**Minimum BF16 physical GPU bytes, not launch admission:**

| Resident target | XTX0 | XTX1 | R9700 |
| --- | ---: | ---: | ---: |
| 4 x 8192 tokens | 14,378,367,232 | 13,875,050,752 | 23,218,913,792 |
| 1 x 262144 tokens | 18,021,005,568 | 17,517,689,088 | 30,504,190,464 |

Immutable source-header-backed model ledger receipts:
- $BC_WORK_ROOT/xtx27-prefill/vtp26-model-64-layer-4streams-8192-20261111.json
- $BC_WORK_ROOT/xtx27-prefill/vtp26-model-64-layer-1stream-262144-20261111.json

These are strictly **lower bounds**: allocator/graph/scratch,
collective staging, conversion scales, fragmentation and
measured safety reserve are absent. Neither 8K x4 nor 262K x1
has been admitted or run, and headers alone are not a
full-payload integrity proof. R9700 memory at 262K x1
has little theoretical space left for those missing costs.

**What is still genuinely open before real 27B TP3 inference:**
numerically validate VTP25 on GPUs; wire actual VTP20 GPU
post-GDN output directly to the FFN without a CPU-oracle
reinjection; convert the per-layer operators into one real
64-layer runtime with preserved GDN conv/recurrent and full-KV
cache state, original embedding/head, per-layer norm, residual,
sampling and measured memory admission; then quality and
end-to-end TP2-vs-TP3 4/6/8-stream tests. Do not claim real
model output or native 262K context from these staged proofs.


### 2026-11-11 — VTP27 first connected GPU GDN→FFN handoff staged (M1 only)

The VTP20 full original BF16 GDN and VTP25 full original
BF16 FFN kernels were previously tested or compiled in
isolation. VTP27 stages the first execution path in which
the actual GPU-produced BF16 post-GDN output is exported,
checked, then consumed by the original checkpoint FFN.
This is a real model hidden-state boundary, not a fresh
synthetic FFN activation.

Two **separate copies** of the accepted original-kernel
HIP sources are added without changing their previous
receipts: xtx27_vtp27_gdn_export.hip and
xtx27_vtp27_ffn_import.hip. They retain VTP20/25
original-weight CPU stage-oracle checks, adding
fail-closed immutable BF16 output files, an exact
GPU-produced GDN input check at the FFN boundary,
and the FFN's post-layer0 output. They compile on
gfx1100+gfx1201. Supervision:
xtx27_vtp27_gpu_layer0_bridge.py:
- Validates all original VTP15/VTP16 GDN checkpoint
  sources, original VTP24 FFN checkpoint sources and
  independent CPU reference receipts, requiring identical
  GDN→FFN parent SHA-256 lineage
- Under **one owned exclusive GPU lease**, runs the
  full GDN GPU and full FFN GPU stages sequentially,
  with no other resident KFD processes
- Requires the produced M1 GDN BF16 residual to equal
  the original source-verified CPU FFN input bit-for-bit,
  so the previously computed original post-attention
  normalized BF16 input is actually valid
- Writes immutable post-FFN GPU BF16 and checks it
  numerically against the independent original-source
  VTP24 complete-layer CPU result. Failure marks an
  incomplete receipt, never numerical success

The **M1 restriction is intentional**: prior VTP20 M4
GPU-versus-CPU BF16 residual had 109 differing values.
It would be invalid to silently reuse the M4 FFN CPU
input for that different GPU residual. General M4
requires recomputing post-attention normalization
and its original-weight FFN reference on the *actual*
M4 GPU output, then requalifying the downstream
numeric tolerance.

The bridge stages **only layer 0** with synthetic
initial hidden. Each original operator relaunches
three GPU processes and hands off BF16 through
a file, not a persistent GPU context/graph. It
cannot generate text, reuse recurrent/KV state,
or claim the full 64-layer model.

VTP27 plan and dual-architecture compilation PASS;
card-free exact-handoff tests PASS. **GPU execution
still pending** because another agent owns the GPU
lease for power/text identity controls.

M1 acceptance when GPUs are free:

    python3 tools/lab/radiance/xtx27_vtp27_gpu_layer0_bridge.py --phase gpu --run --iterations 5 \
      --weights-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp15-gdn-projection/layer0-20261111T005819109215Z/summary.json" \
      --projection-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp15-gdn-ref/layer0-20261111T010143705406Z/summary.json" \
      --core-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp15-gdn-core/layer0-20261111T011811618321Z/summary.json" \
      --learned-receipt "$BC_WORK_ROOT/xtx27-prefill/vtp14-gdn-reference/layer0-20261111T003313797426Z/summary.json" \
      --output-packet "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output/layer0-20261111T014452115048Z/summary.json" \
      --output-reference "$BC_WORK_ROOT/xtx27-prefill/vtp16-gdn-output-ref/layer0-20261111T015557796594Z/summary.json" \
      --ffn-packet "$BC_WORK_ROOT/xtx27-prefill/vtp24-original-ffn/layer0-20261111T061700090195Z/summary.json" \
      --ffn-reference "$BC_WORK_ROOT/xtx27-prefill/vtp24-original-ffn-cpu/layer0-m1-20261111T061945282150Z/summary.json"

Next after a passing VTP27: make GDN/FFN
layer-agnostic and share one actual GPU context and
resident weights/states, extend full attention with
persistent causal KV cache, implement actual original
embed/head, run all 64 layers then a full decoder
step, and compare TP3/TP2 quality and throughput.


### 2026-11-11 — VTP25/VTP27 original BF16 layer-0 FFN and GPU GDN→FFN hardware acceptance

**VTP25 physical-TP3 original checkpoint FFN GPU ACCEPTANCE:**
The new original layer-0 5120/4864/7424 BF16 SwiGLU FFN
ran on 2x gfx1100 RX 7900 XTX + 1x gfx1201 R9700.
It takes the original post-GDN CPU-reference residual,
applies source-hashed original post-attention weighted
RMSNorm, executes original BF16 gate/up/down weights,
three concurrent per-rank contributions, pinned-host
ordered F32 reduction and BF16 residual, then broadcasts
the output. Both M1 and M4 GPU runs have rc=0 and
independent checkpoint-derived CPU-oracle tolerance PASS.

| VTP25 case | Wall p50 | Wall p95 | Final BF16 exact | Largest absolute error |
| --- | ---: | ---: | ---: | ---: |
| M1 | 875.92 us | 1,005.88 us | 5,107 / 5,120 | 0.00390625 |
| M4 | 1,564.19 us | 1,898.33 us | 20,455 / 20,480 | 0.00390625 |

Immutable receipts:
- $BC_WORK_ROOT/xtx27-prefill/vtp25-original-ffn-gpu/runs/20261111T064141720549Z-4afdb501e8/summary.json
- $BC_WORK_ROOT/xtx27-prefill/vtp25-original-ffn-gpu/runs/20261111T064157636260Z-4afdb501e8/summary.json

**VTP27 first connected original-weight GPU GDN→FFN M1 ACCEPTANCE:**
Under ONE runner-managed exclusive GPU lease, the actual
VTP20-based GDN GPU checkpoint execution exported its
post-GDN BF16 residual. The independent VTP25-derived
FFN GPU process imported that exact GPU-produced file.
The imported GDN output matched its source-hashed CPU
reference 5,120 / 5,120 BF16 words bitwise, with both
SHA256 digests
7b51c63c34a341a3ee5cc1872a6ef6dfdab6fd4259fd2bf30803af76414.

The complete actual original checkpoint-derived GPU
layer-0 FFN residual, ready as an input hidden tensor
for layer 1, matched 5,107 / 5,120 BF16 words exactly.
The remaining 13 were within the current numerical
tolerance, maximum absolute difference 0.00390625.
**Not bit-exact complete layer-0 output**. Both internal
GPU stage subprocesses recorded rc=0, oracle_pass=true.
Immutable receipt:
- $BC_WORK_ROOT/xtx27-prefill/vtp27-layer0-gpu-gdn-ffn-bridge/runs/20261111T064222931275Z-864e59e3e5/summary.json

Actual GPU-produced file:
- $BC_WORK_ROOT/xtx27-prefill/vtp27-layer0-gpu-gdn-ffn-bridge/handoffs/20261111T064222611374Z-864e59e3e5/gpu-after-GDN-residual.bf16

Actual complete layer-0 GPU-derived hidden state:
- $BC_WORK_ROOT/xtx27-prefill/vtp27-layer0-gpu-gdn-ffn-bridge/handoffs/20261111T064222611374Z-864e59e3e5/gpu-after-layer0-FFN.bf16

**Exact limitation:** VTP27 handoff runs each model
operator as a separate 3-GPU process, serialised with a
host-file BF16 bridge, **NOT** resident GPU-to-GPU
state transfer or a native model runtime. First layer
input is deterministic synthetic BF16 hidden, not an
actual token embedding. VTP27 is only M1; prior VTP20
M4 post-GDN output had 109 BF16 differences, so blindly
reusing the precomputed M4 CPU normalized FFN input would
not represent the actual GPU hidden tensor.
No model text generation, no 64-layer execution, no
persistent 16-layer causal KV or 48-layer GDN history,
no measured VRAM admission, no end-to-end TP3 vs TP2
performance claim and no production activation.

All three GPU acceptance runs released their lease;
at the post-run check no lease or resident KFD process
remained. The tests do not alter GPU clock/power settings.
Next highest priority: layer-agnostic GPU operator runtime
within one process, with original embedding/head and
per-layer residual dispatch, physically resident weights
and recurrent/KV state; only then actual tokens/s and
resident context qualification. Keep previous exact
VTP19/GDN/attention results separate from model output.


## 2026-11-11 — Native Radiance integration, VTP30 opt-in source patch (VTP13)

After the standalone original-BF16 VTP20/23/25/27 numerical
prototypes, **stop constructing a separate Python/HIP serving
runtime**. Radiance v1.17.0 already has the Qwen3.5/3.8 BF16
64-layer graph (GDN, gated causal attention, SwiGLU FFN),
embeddings/head/sampling, native residual forwarding, persistent
KV/GDN/conv state, a per-rank scheduler and world-size-three exact
all-reduce with ascending-rank F32 accumulation. The experiment
must become a Radiance feature, not a competing engine.

First native source patch:
`engines/radiance/patches/rad0016_vtp30_native_tp3_ownership`,
available **only** in the independent
`source.radiance-vtp30-native-tp3` recipe and only with
`BC_VTP_NATIVE_TP3=1`. Uses Radiance's already available
`rad_weight_shard_span` /
`rad_weight_shard_span_parts` interfaces to express exact
original-Qwen BF16 physical ownership:

- Attention: Q+gate heads 6/6/12 with interleaved query/gate,
  K/V 1/1/2 heads, matching 1536/1536/3072 input columns
  of the native attention output projection
- GDN: key heads 4/4/8, value heads 12/12/24;
  exact five fused Q|K|V|a|b row parts, 3-part convolution,
  48-wide a/b/A_log/dt_bias heads, Z, and output columns
- FFN: 5120/4864/7424 gate/up rows and down input columns;
  the original full 17408-wide checkpoint is read by the
  existing loader, not a VTP-produced weight package

All other architectures and regular model deployments retain the
existing source declaration. Wrong model geometry and non-3-rank
activation refuse rather than quantise/slice incorrectly.
No Radiance scheduler/state/cache/collective/kernel execution
implementation is replaced.

**Card-free**: managed patch matches the pinned v1.17.0 source,
all eight edits apply idempotently and atomically to five files,
the native patched Qwen BF16 C++ plugin passes C++23
`g++ -fsyntax-only` and an isolated native C++23 `-fPIC -shared` plugin build against upstream headers (not installed), and targeted
tests exercise the ownership/source-drift contract. No GPU
source patch has been validated for mixed-architecture
execution; patch remains `untested`.

**Separate mandatory blocker before any TP3 hardware smoke**:
the core `usable_devices` filter currently discards a card
whose GPU architecture differs from the first one; core/plugin
loader tests fatbinary code compatibility against only the
first device. Removing the filter alone is **unsafe**: device
kernels may be selected for a card with no compatible code.
Next native slice must implement **per-rank device-code/kernel
eligibility and fail-closed selection** across gfx1100/gfx1201,
then compile a truly native experimental build and validate
graph/kernel coverage, memory admission, startup, numerical
quality and live token generation on all three physical GPUs.
Only after real tokens and quality pass run matched TP2/TP3
C4/C6/C8, resident KV and power A/B. Do not claim these
from VTP19 microbench figures or the standalone VTP27 file
handoff.
