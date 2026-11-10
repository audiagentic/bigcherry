---
id: VTP12
order: 11
plan: run-dense-27b-virtual-tp
state: active
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

## 2026-11-11 — first VTP12 execution: existing-native-row reanalysis

**Status: card-free analysis executed; no new leased GPU work.**
The GPU lease was held by another agent (R9700 PM study). We used the
36 immutable VTP05 same-invocation `--perfw8` receipts instead of
interrupting that lease. Source receipts:
`${BC_WORK_ROOT}/xtx27-prefill/quant-shape/placement-w8a8-20261110/runs/`.
Verified offline machine report:
`${BC_WORK_ROOT}/xtx27-prefill/vtp12-fp8-int8-prior-20261111/report.json`.
Reproduce with `tools/lab/radiance/xtx27_vtp12_compare.py --runs-dir
${BC_WORK_ROOT}/xtx27-prefill/quant-shape/placement-w8a8-20261110/runs`.
The tool validates stored receipt geometry/ABI/provenance, paired
actual FP8 and INT8 kernel rows, reports missing/partial cases rather
than extrapolating, and makes no hardware calls. Ten focused card-free
tests plus Ruff lint pass.

| Candidate; M | Three-device FFN ideal W8A8 barrier | XTX INT8 + R9700 FP8 ideal barrier | R9700 gate+down INT8 -> FP8 |
| --- | ---: | ---: | ---: |
| 30/30/40; 1 | 138.6 us | 138.6 us | 103.9 -> 97.3 us |
| 30/30/40; 4 | 138.3 us | 138.3 us | 102.7 -> 97.7 us |
| 30/30/40; 8 | 138.6 us | 138.6 us | 106.5 -> 102.1 us |
| 30/30/40; 64 | 314.5 us | 314.5 us | 197.3 -> 140.4 us* |
| 35/35/30; 4 | 149.2 us | 149.2 us | 63.3 -> 61.7 us |
| 37.5/37.5/25; 4 | 173.6 us | 173.6 us | 54.6 -> 52.8 us |

*At M64 the provisional FP8 narrow row is compared against VTP05's
INT8 selected tiled row; independent INT8 narrow timing also exists
(161.5 us for R9700 gate+down) and should be evaluated for actual
registry dispatch. **All stages remained XTX-limited in the ideal
no-communication FFN lower bound, so this matrix predicts 0%
three-GPU barrier improvement from changing ONLY the R9700 format.**
None of these numbers includes activation quantisers, per-rank
mixed-mode compatibility, overlap, collective, actual weight residency
or model quality, nor is the FP8 runtime row selector proven.

**Next discriminating GPU experiment**: measure new 27.5/27.5/45
(tile-rounded) and 25/25/50 physical FFN partitions, initially at
M1/4/8/64, both real FP8 and INT8 R9700 rows and matching XTX INT8.
The existing tested 30/30/40 remains the control. Run one card at a
time with the exclusive lease, paired ABBA repeats and original
binary/plugin digests. Only after local balance improves should a
single **simultaneous** three-card exact reduction be built and timed.
Account for R9700's **two** KV heads: 50% FFN allocation leaves less
32-GiB VRAM for four long 256K contexts, GDN and scratch; use VTP08
strict lower bounds before proposing any resident serving admission.
Do not declare the FP8 variant faster end-to-end from this screen.

### Source-contract blockers to resolve before within-layer mixed format

Pinned upstream `libr4d/r4d_rows.cpp` exposes
`gemm_fp8a8_nt_m16` / `gemm_fp8a8_tiled` with E4M3 activation,
F32 per-row/128 activation scales and BF16 per-block weight scales;
the INT8 rows use different activation and weight scale layouts.
`libr4d/r4d_selftest.cpp` `w8_perf` times both from the same
synthetic input, but **does not** prove an INT8 rank and FP8 rank
consume the same correct upstream activation or can combine with
one engine graph. `arch/qwen35_fp8/qwen35_fp8.cpp` divides GDN
heads by uniform world size, so native `--tp 3` is also unqualified.
XDP10/VTP13 must address these as explicit fail-closed ABI gates.

### Card-free next-shape geometry/capacity gate

For 256-wide FFN tiles (17,408 total intermediate), candidate
27.5/27.5/45 rounds to physical widths **4864/4864/7680**
(27.94/27.94/44.12%); 25/25/50 gives **4352/4352/8704** exactly.
All FFN widths are 256-aligned, allowing native GEMM-shape probes;
this does not prove other attention/GDN/legal tensor boundaries.
At 4 x 256K aggregate resident tokens with FP8 KV, the R9700
50%-FFN choice has a strict **~8.093 GiB W8A8 FFN + 16 GiB KV**
minimum, leaving **<7.91 GiB** of nominal 32 GiB for everything
else; this is a *floor*, not an admission approval (VTP08).

## 2026-11-11 — leased R9700-heavy native kernel ABBA experiments

**Executed:** three matched, fresh, exclusive-GPU-lease campaigns,
24 standalone `r4d_selftest --perfw8` processes per campaign
(**72 total**, two ABBA samples per arm for each of three physical
GPUs and two FFN projections). All 72 processes exited 0, all paired
actual native INT8/FP8 kernel rows parsed, zero missing/rejected cases,
and every lease was released automatically. Independent final check:
`agent-lease status` => no leases; `rocm-smi --showpids` => no KFD PIDs.

Code: `tools/lab/radiance/xtx27_vtp12_pilot.py`, which requires the
exclusive `agent-lease gpu` wrapper, fails on inherited device masks,
requires the pinned selftest binary and gfx11/gfx12 kernel plugin
provenance, validates each measured shape and writes independent
immutable per-case receipt/stdout/stderr. Builds, converter, model
weights and production service were not touched. M=4 and M=64,
9 selftest iterations/row, **warm synthetic constant weights** only.
Exact full manifests, instrument results and numerical-unchecked
receipts:

- `${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-45-m4-20261111/`
- `${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-45-m64-20261111/`
- `${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-50-m4-20261111/`

### M64 dispatch correction, raw receipts preserved

The original VTP12 pilot `summary.json` at M64 used the narrow INT8
kernel for **all three** cards. At M64 this is not the VTP05
`selected_row` convention: VTP05 chooses `gemm_i8a8_tiled` for
M>16; the XTX tiled kernels are much faster than narrow there.
Do **not** quote the original M64 summary as the selected INT8
engine-control result. Added a separate card-free read-only verifier
`xtx27_vtp12_replay.py`, which checks each case/shape, source
stdout SHA and all measured native rows, then writes
`summary-replay-v1.json` **without rewriting the raw cases or the
original summary**. It reports both VTP05-declared INT8 dispatch
and a separate *optimistic best-measured INT8 row per GPU* lower bound.
Neither is proof the actual engine graph will select that row.

### Experimental result — ideal FFN-only barriers, microseconds

| Physical FFN split; M | INT8 control 30/30/40 | Challenger INT8, VTP05-selected | Challenger XTX INT8 + R9700 FP8 | Best observed INT8 challenger |
| --- | ---: | ---: | ---: | ---: |
| ~28/28/44; 4 | 138.95 | 156.00 | 152.30 | 156.00 |
| 25/25/50; 4 | 139.00 | 192.45 | 186.80 | 192.45 |
| ~28/28/44; 64 | 312.70 | 317.00 | **304.35** | **304.35** |

`~28/28/44` rounds intermediate tensor widths to
**4864/4864/7680** (27.94/27.94/44.12%); 25/25/50 is
**4352/4352/8704** exactly. All figures are sums of two serial
FFN dependency-stage *ideal* barriers: for each projection, take
max(three GPU **independently measured** kernel times). They **exclude**
overlap, activations, no-P2P wire/CPU combine, GDN/attention, actual
mixed model layout and quality. The ABBA has only two samples/arm
per projection; repeat with >=3 fresh campaigns before promotion.

At M4 the 44% candidate raises the gate/up R9700 FP8 time to
99.3 us, beyond XTX1's 83.4 us. The 50% candidate raises R9700 gate/up
FP8 to 139.0 us against XTX1's 66.45 us: **a severe R9700 critical
path, not a balanced three-card distribution**. Exceeding ~41%
physical FFN weight share is a poor **current decode-kernel** choice.
A change from 7168 to 7680 intermediate columns produces a much
larger latency jump than the width ratio alone suggests. Source
`libr4d/r4d_gemm_fp8a8.hip` narrow_mw/ksplit policy does not
obviously change at M4 with these aligned N values; check weight
residency, cache, waves and actual kernel counters before proposing a
firm mechanism.

At M64 the same 44% split produces a modest **2.67% ideal FFN
barrier reduction** versus the 30/30/40 control if R9700 uses FP8.
However, R9700's **best measured native INT8 narrow** kernel yields
the *identical* 304.35-us bound, while using the VTP05-selected tiled
INT8 row gives 317.0 us (slower than baseline). **Optimised
per-architecture/per-shape dispatch may recover all this apparent
mixed-format win without producing a multi-variant model package.**
Verify dispatch with the real registry and matched numerics, not a
timing-only selftest.

### Decision and next bounded work

- Keep 30/30/40 as the M1/M4/M8 decode baseline; reject 25/25/50
  for current M4 kernels. No full-engine promotion.
- First tune/check the gfx1201 gate/up latency cliff across
  neighboring 256-aligned widths near 7168..7680 and assess cold
  weights/pinned clocks, and review gfx1100 M64 tiled-vs-narrow and
  gfx1201 M64 narrow-vs-tiled dispatch. Reuse VTP05 oracles,
  card-free checks and an explicit kernel policy test.
- Next run legally sized GDN and full-attention **two-head R9700
  versus one-head XTX** tests and prove a shared mixed activation/output
  ABI, then a bounded simultaneous three-device exact reduction
  through VTP04 harness. Those gates may outweigh any FFN-only gain.
- Preserve XDP10's multi-variant format as a candidate, **not a default**.
  Only undertake loader/producer changes after a meaningful
  *net* benefit over optimised all-three INT8 dispatch AND full-model
  BF16-relative quality. Retain VTP08 FP8-KV/R9700 32-GiB admission
  constraints; FP8 weights are not automatically smaller than INT8.
