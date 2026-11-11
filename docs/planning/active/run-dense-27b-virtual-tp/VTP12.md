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

## 2026-11-11 — finer physical R9700 split prepared (card-free, hardware pending)

Candidate `r9700-43`: **4992 / 4992 / 7424** physical FFN intermediate
columns (28.68 / 28.68 / 42.65%). Its four logical benchmark
partitions are **4992 / 4992 / 3712 / 3712**, each 128-aligned.
The prior 30/30/40, ~28/28/44 and 25/25/50 candidates retain their
original 256-tile geometry and unchanged frozen VTP12 manifests. The
new split is a microbenchmark **shape** only, not an implemented
physical-TP3 or fused virtual-TP4 execution path.

The existing pinned selftest's `--perfw8` accepts N multiple 256 and
K multiple 128. XTX gate/up: N=9984, K=5120; R9700 gate/up: N=14848,
K=5120. Down: N=5120, K=4992 (XTX) or K=7424 (R9700). Matching
native narrow/tiled GEMM row constraints allow these *shapes*, not
yet a correct engine or mixed quant graph. Updated the existing
`xtx27_vtp12_pilot.py` and `xtx27_vtp12_replay.py` to include the
new candidate, preserving lease checks, provenance and immutable
ABBA receipts. 19 focused tests + Ruff passed; all 72 prior cases
replayed with unchanged schedules.

**At initial preparation (before later GPU runs):** the initial check found an XTX
-125 mV power A/B GPU lease and KFD processes. At the final check,
the lease was free but other Radiance/MoE KFD processes were still
resident *outside any lease*. Never stop/preempt their processes.
Do not acquire a measuring lease or start a benchmark until both
`agent-lease status` and `rocm-smi --showpids` are clear.

When `agent-lease status` says no leases and `rocm-smi --showpids`
says no KFD processes, the reproducible hardware command is:

```bash
# From the existing VTP12 worktree, with BC_WORK_ROOT configured.
: ${BC_WORK_ROOT:?Set BC_WORK_ROOT in environment.local.toml}
BUILD="${BC_WORK_ROOT}/builds/f17f21fda44979e9abe84f13e96b8896/08b5dd81b4874390bf0cbf489db0d7df"
OUT="${BC_WORK_ROOT}/xtx27-prefill"
python3 tools/lab/radiance/xtx27_vtp12_pilot.py --phase run --run --challenger r9700-43 --m 4 --iters 9 --build-dir "$BUILD" --output-dir "$OUT/vtp12-abba-43-m4-UNIQUE" --timeout-s 30
python3 tools/lab/radiance/xtx27_vtp12_replay.py --campaign-dir "$OUT/vtp12-abba-43-m4-UNIQUE" --output "$OUT/vtp12-abba-43-m4-UNIQUE/summary-replay-v1.json"
```

The runner acquires/releases the exclusive lease, refuses inherited
GPU masks and pre-existing KFD processes, and rejects changed pinned
binary/plugin hashes. Do not double-acquire its lease. After a passing
M4 test, use unique directories for M1/M8/M64. M64 must use the
separate corrected replay (VTP05-declared INT8 tiled at M>16 versus
optimistic best-native INT8 row). Require >=3 fresh ABBA campaigns,
actual 3-card simultaneous communication, real GDN/attention
two-head tests, numerical oracles, BF16-relative full-model quality
and physical VRAM/KV admission before any serving promotion.

## 2026-11-11 — complete 42.65%-R9700 ratio sweep (GPU measured)

**Executed, not merely planned.** Two physical layouts were tested at
M=1,4,8,64 with pinned same-build `r4d_selftest --perfw8`: eight
24-case ABBA campaigns = **192 GPU selftest invocations** across XTX0,
XTX1 and R9700, nine timing iterations per row. All 192 cases exited
successfully, all eight corrected read-only replay reports validate the
full case/log hash census, zero failure receipts. Every exclusive GPU
lease auto-released; final `agent-lease status` and KFD PID checks clear.
These are independently timed warm, synthetic-value GEMMs and
mathematically combined **ideal** FFN dependency-stage barriers, not
simultaneous TP, model tensors, language accuracy, KLD, or throughput.

Physical width assignments:
- **Symmetric** `r9700-43`: 4992 / 4992 / 7424, physical share
  28.68/28.68/42.65%; 128-column tile support required for both XTX
  down K=4992.
- **Unequal XTX** `r9700-43-skew`: 5120 / 4864 / 7424, physical
  share 29.41/27.94/42.65%; all three down-K widths divisible by
  256. The R9700 computes **exactly the same** gate/down shapes in
  both layouts; only the XTX split changes. Both are per-operator
  *candidate shapes*, not full legal attention/GDN/TP contracts.

### Corrected paired results (microseconds; lower is better)

M64 uses the independent replay's `vtp05_declared_i8_dispatch`
(tiled INT8 M>16), **not** the earlier pilot's narrow-INT8
`summary.json`. For each case the R9700-FP8 arm keeps INT8 on both
XTX cards. Each row has its own same-campaign 30/30/40 INT8 control.

| M | Symmetric control INT8 | Symmetric 42.65% INT8 | Symmetric 42.65% mixed FP8 | Unequal control INT8 | Unequal 42.65% INT8 | Unequal 42.65% mixed FP8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 139.00 | 181.40 | 179.80 | 138.95 | 139.95 | **138.80** |
| 4 | 139.10 | 188.55 | 184.85 | 139.05 | 142.40 | **142.00** |
| 8 | 139.15 | 187.05 | 183.30 | 139.30 | 144.80 | **141.35** |
| 64 | 313.20 | 312.15 | 312.15 | 312.65 | 319.10 | **309.90** |

Exact source folders with raw per-case receipts, console logs,
immutable original `summary.json` and validated
`summary-replay-v1.json`:
`${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-43-m1-20261111/`,
`${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-43-m4-20261111/`,
`${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-43-m8-20261111/`,
`${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-43-m64-20261111/`, and
`${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-43-skew-m1-20261111/`,
`${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-43-skew-m4-20261111/`,
`${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-43-skew-m8-20261111/`,
`${BC_WORK_ROOT}/xtx27-prefill/vtp12-abba-43-skew-m64-20261111/`
(all prefixes are `${BC_WORK_ROOT}/xtx27-prefill/`).
These 8 runs are distinct from the prior VTP05 records and
VTP12's earlier 30/30/40-vs-44% and -50% measurements.

### Interpretation and decision

1. **Avoid symmetric 128-wide XTX down shards at decode.** At M4,
   the XTX down rows jump from ~54-56 us (K=5120 control) to
   ~93-96 us (K=4992 symmetric). This occurs despite *less*
   weight work. `libhalo/halo_gemm_fp8a8.hip` `fp8_sk` explicitly
   chooses FK=128 when K is not divisible by 256 (versus FK=256
   otherwise). This is a **plausible causal mechanism** for a slower
   kernel path, not a proven explanation absent counters/pins.
2. Unequal XTX physical 5120/4864 keeps both down K sizes divisible
   by 256 and recovers almost the full loss. At M4 the ideal mixed
   FFN barrier is **142.00 us**, versus **139.05 us** control
   (2.12% slower), rather than symmetric **184.85 us**
   (~32.9% slower than its control). At M1 the unequal mixed arm
   is virtually tied at 138.80 vs 138.95 us, and at M8 it is
   141.35 vs 139.30 us (1.47% slower).
3. At M64 unequal mixed FP8 reaches **309.90 us** vs control
   312.65 us (~0.88% lower), but the independent
   **best-observed native INT8 rows** reach the *same*
   309.90-us ideal barrier (the declared INT8 kernel policy gives
   319.10 us). That makes native per-GPU shape-specific dispatch
   a mandatory no-new-model-format control before XDP10.
4. At M4, the unequal R9700 FP8 gate/up takes **88.05 us**,
   versus XTX1 **83.35 us**; the R9700 becomes the critical
   gate/up contributor. The next speed experiment should target
   its native gfx1201 gate/up at N=14848 K=5120 M=1/4/8
   and optimise the launch/split/grid before adding yet more
   FFN work. Check actual activation preparation and whether
   the claimed native FP8 selector is used by the engine.
5. Do not assume added GPU VRAM increases resident KV. Full
   attention KV-head ownership (1/1/2), GDN state/conv,
   duplicated head/embedding/MTP, graph scratch, no-P2P PCIe
   transport and host accumulation may change the conclusion.
   8K concurrent 4/6/8 streams and long-context admission
   remain mandatory. The current data do **not** establish a
   faster three-card model or a production-ready mixed-precision
   runtime.

**Go/no-go:** retain 30/30/40 as the native INT8 decode baseline;
advance the **unequal physical 5120/4864/7424** layout as a
provisional competitor, reject the symmetric 4992/4992/7424
decode proposal with current kernels. Do not promote a
multi-variant model bundle yet. Next steps: (a) bounded
shape-specific numeric and dispatch oracle for K=4992/4864/5120
on gfx1100 and R9700 FP8 N=14848, (b) simultaneous three-card
FFN + exact F32 host combine with measured D2H/CPU/H2D
and engine-ordering correctness, (c) 1/1/2 two-head full
attention and 4/4/8 GDN key, 12/12/24 value concurrent
operator acceptance, (d) repeated full-model 4/6/8 stream
BF16-reference quality and throughput/KV/PCIe/TTFT/latency
A/B across both architectures. Three new independent ABBA
campaigns are still required for hardware promotion; the
single pilot per M with two ABBA points is screening only.

## 2026-11-11 — one-physical-rank TP3 transport baseline delivered

VTP04's opt-in 3-physical-GPU exact F32 host transport now passes
independent GPU output oracles across three fresh lease-bound runs.
For 1/4/64 rows, the single R9700 physical-partial median wall times
were 76.35/96.28/627.52 us; the R9700 two-virtual-partial local
combine was 80.41/102.41/645.70 us. **This is an already-computed
F32 partial transport**, not native three-card W8A8/FP8 GEMM or
an equal-work comparison: VTP12's simultaneous mixed quant GEMM,
common activation/output ABI, attention/GDN and model quality
acceptance remain OPEN. See VTP04/VTP13 for receipts and contracts.

## 2026-11-11 — concurrent INT8 FFN pilot staged (card-free only)

`xtx27_vtp13_chain.py` adds an opt-in experiment to VTP04's
source-hashed compile/lease/receipt machinery, using
`xtx27_vtp13_ffn.hip` (compiled card-free for gfx1100+gfx1201).
Geometry is **5120/4864/7424**, full hidden size 5120. Three physical
streams perform genuine INT8 GEMM gate/up, SwiGLU, requantisation,
down GEMM, then three D2H packets, CPU F32 reduction, and H2D broadcast.
An independent CPU oracle checks all output rows and physical ranks.
The source BF16 values are *synthetic and exactly representable*;
original model payload bytes and optimized Radiance GEMMs are **not** used.
The timing will not compare directly with native VTP12 selftests.

`python3 tools/lab/radiance/xtx27_vtp13_chain.py --phase gpu --run --rows 1,4 --iterations 3`
automatically requires the exclusive lease and empty KFD inventory;
a reversed-order fresh start is the immediate control. During staging
another agent held an XTX power A/B lease and four KFD processes were
present; **no VTP13 hardware execution had yet occurred at staging**.
Next: GPU numerical proof, real BF16-checkpoint-derived shard weights,
native INT8/FP8 ABI and R9700 attention/GDN head-local measurements.

### Candidate source review: Spry.ai gfx1201 SGLang (not copied)

Reviewed `Drgr33nSenior/Spry.ai-rdna4-serving` public `sglang-main`
patches at 2026-11-11: 0022 register-fed prepacked MXFP4 Gluon
verify GEMM (reported 18.6->12.1 ms per round, M<=16), 0018
single-launch fused Split-K, 0019/0020 fused norm/SwiGLU + MXFP4
quantisation, 0021 wider single-request long-context verify-attention.
These suggest separate R9700 operator probes **after** the uniform INT8
chain. Patch 0014 requires actual two-gfx1201 PCIe P2P and changes
system-scope release/acquire flag ordering; BigCherry has **no usable
P2P**, so it cannot replace the CPU-staged three-rank collective.
The source's Nov 9 README says its current SGLang-main profile is
unqualified and historical op234 runtime cannot be reproduced as
documented. No MXFP4 quality, speed or production choice is imported.
Source: https://github.com/Drgr33nSenior/Spry.ai-rdna4-serving .

### 2026-11-11 — first real concurrent three-GPU FFN hardware observations

Two **exclusive-lease starts** on the original prototype
source digest `c4a9d96253e3c824` completed. Raw immutable receipts:
`BC_WORK_ROOT/xtx27-prefill/vtp13-ffn-chain/runs/20261110T215450941137Z-c4a9d96253/`
and `20261110T215529616898Z-c4a9d96253/`.
The first ran M1; the second M1/M4/M8, with 3 timed repetitions per case
plus warm-ups. All four cases had `rc=0`, an independent CPU quantized
oracle with reported max-abs error 0, and identical three-GPU broadcast
outputs. The full **INT8 FFN chain did run concurrently**.

| M | FFN+host TP3 median wall | XTX0 / XTX1 / R9700 device execution |
| --- | ---: | --- |
| 1 | 852.84 us (first); 863.48 us (second) | 503 / 480 / 745 us (first) |
| 4 | 1622.85 us | 1525 / 1429 / 812 us |
| 8 | 2957.80 us | 2494 / 2512 / 1498 us |

These are intentionally **non-native scalar-warp INT8 kernels with
synthetic exact-BF16 weights**, and cannot be compared as production
speed or BF16-checkpoint validation. The limiting device switches
from R9700 at M1 to an XTX at M4/M8 under this kernel. The original
three-sample p95 was incorrectly a second median; do not use it.
Nearest-rank p95 was repaired and compiled card-free as digest
`49408233ba43eac2`, still awaiting a GPU re-run. A reversed-order
start was correctly refused when the power agent reacquired the lease.
No third matched start, native GEMM A/B or model text generation.

## 2026-11-11 — follow-up: real checkpoint W8A8 and head-local staging

The opt-in uniform-INT8 physical TP3 harness now loads actual
Qwen3.8-27B layer-0 original BF16 gate/up/down tensors, validates
index/header offsets and hashes complete selected payloads, then uses
per-output-channel W8 scales and performs a concurrent 3-device
FFN->host collective. Both M1 and reverse M4 passed independent CPU
INT8 oracle (max abs 0) with BF16-weight RMSE ~0.00122/0.00128
on deterministic BF16 input. These are not native Radiance GEMM or
serving timings. Raw runs and details are in VTP13.

`xtx27_vtp13_heads.py` stages actual head-local 1/1/2 GQA
attention softmax and a simplified 4/4/8 key, 12/12/24 value
recurrent state proof; the GPU experiments await an exclusive lease.
Card-free original BF16 GDN QKV/conv rank pack reassembles exactly
to the source hashes. **Still mandatory before FP8/TP3 promotion**:
native gfx11/gfx12 optimized operator and common activation/scale ABI,
full original GDN conv/decay/state correctness, matched TP2/TP4
latency, real 4/6/8-stream throughput, KV admission and model quality.
