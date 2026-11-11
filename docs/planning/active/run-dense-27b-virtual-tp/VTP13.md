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
