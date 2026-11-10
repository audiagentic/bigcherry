---
id: VTP05
order: 4
plan: run-dense-27b-virtual-tp
state: active
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Q6/IQ6R and Q8/W8A8: actual Radiance model-shaped kernel screening

## Purpose and order

User goal: Q6 or Q8 (not Q4), maximum resident KV, 4+ streams, two gfx1100 XTX and one gfx1201 R9700. This plan measures **existing quantised GPU kernels** before building a virtual TP4 graph.

1. Card-free shape and kernel audit. Qwen3.8-27B hidden 5120, FFN intermediate 17408, combined gate/up N34816 K5120, down N5120 K17408. V4 IQR runs impose 256-alignment on both N and K. Split the *intermediate* in whole 256-element tiles, and use twice each logical width for gate/up N, one width for down K. Rounded percentages are not exact. Attention/KV/GDN splits require separate proof.
2. Reuse Radiance's real selftest performance hooks, not another pretend Q6 kernel: --perfiqr N K M 6 [iters] for the native IQ6R V4 trellis decoder and GEMM; --perfw8 N K M [iters] for the native W8A8 and additional 8-bit kernels. The selected rows must be matched to the intended dtype and M. For IQ6R, --cold 256 requests cache flushing in the underlying perf_row helper. Radiance --perfw8 has its own warmed HIP-event loop and **ignores --cold**; W8A8 selftest results are warm only. Use checked rad-kbench with --bench-flush for cold Q8 evidence.
3. One bounded case per exclusive lease invocation: physical card XTX0, XTX1 or R9700; gate/up or down; capacity (30/30/20/20), intermediate (35/35/15/15) or bandwidth (37.5/37.5/12.5/12.5); M=1,4,8 then 64,128,512. Run small probes before increasing. Compare R9700 one combined physical GEMM with its two separate logical shard GEMMs (sequential; not falsely treated as concurrent).
4. Repeat stable candidates under matched clock/cold/warm conditions. Use KAB01/rad-kbench for numerical checks against libref on the existing Q8 fixture, and selftest IQR correctness suites. Timing-only selftest does not prove numerical fidelity. No model weights are loaded.
5. Combine physical GEMM times with separately measured VTP04 three-GPU communication lower bounds to reject clearly bottlenecked ratios. Only then consider VTP01 virtual-rank implementation; VTP06 owns complete Q6/Q8 model quality/resident KV.

## Reproduction

The new tools/lab/radiance/xtx27_quant_kernel.py defaults to a fully card-free --phase plan. Its --phase run --run requires an explicit build directory with a matched r4d_selftest binary and matching libr3/libr4d plugins. It obtains the global GPU lease itself, executes ONE case and writes hashed source/binary/plugin provenance, stdout/stderr and an honest success/failure receipt to scratch under BC_WORK_ROOT.

Examples (from this worktree; BUILD must point to a verified matching build):

    python3 tools/lab/radiance/xtx27_quant_kernel.py --phase plan

    python3 tools/lab/radiance/xtx27_quant_kernel.py --phase run --run \
      --build-dir "$BUILD" --quant iq6r --candidate capacity \
      --gpu r9700 --projection gate_up --m 4 --iters 3 --cold-mib 256

The --perfiqr 6 probe is a specific synthetic V4 six-bit form, not the full mixed-bit calibrated recipe and not GGUF Q6_K. --perfw8 includes FP8 and legacy W8 rows: only the native INT8 row is admissible evidence for our W8A8 objective. Neither probe replaces model KLD/continuation quality.

## Source finding

Pinned Radiance has libr4d/r4d_iqr.h, r4d_gemm_iqr.hip and calibrated Qwen3.8-27B IQ6R recipes (about 6.06–6.94 bpw). The source IQR document reports prior quality measurements, but they are not independent current BigCherry results. Both intended GPU architectures must be verified from actual dispatch and runtime ABI.

## Acceptance

Record selected real kernel, physical device, N/K/M and precision, repeated p50/p95 kernel times at supported shapes, plus a physical-stage cost model with measured three-GPU exchange. Fail closed on unsupported dispatch, absent timing, ABI mismatch, numerical oracle failure or lost GPU lease. No production changes or speculative throughput claims.

## 2026-11-10 progress

- Confirmed current Radiance source implements actual 6-bit IQR-V4
  decoder/GEMM and a separate native W8A8 INT8 GEMM; the existing
  matched r4d_selftest and libr3/libr4d plugins are present on disk.
- Implemented shape-aligned, card-free plan and guarded per-case runner.
  The capacity ratio rounds to intermediate widths
  5120/5120/3584/3584 (physical 29.41/29.41/41.18 percent).
- All 8 new card-free VTP05 tests and all 91 XDP tests PASS; Ruff
  passes. No numerical oracle or real quantized-kernel timings yet.
- Hardware gate blocked: exclusive GPU lease held by the live
  flashnext-4gpu placement recalibration. Do not preempt it.
- VTP05 next one-case step when free: IQ6R M=4, gate/up,
  30/30/20/20 candidate first on gfx1100 and gfx1201, followed
  by W8A8 in matched build and M=1/8/64. Every cell is separately
  leased, hash-receipted and numerical-unchecked until dedicated
  oracle/kbench passes.

### Native Q6 executable compatibility correction

An older local build contained --perfw8 but NOT --perfiqr. It is
unsuitable for IQ6R benchmarking despite having historical W8A8
fixtures. Audited newer prebuilt source-slice f17f21fda44979e9abe84f13e96b8896,
build plan 08b5dd81b4874390bf0cbf489db0d7df (under BC_WORK_ROOT/builds).
Its selftest has both --perfiqr and --perfw8, and its gfx1100 libr3
and gfx1201 libr4d plugins both have IQ6R narrow/tiled and W8A8
narrow/tiled rows. This is symbol/ABI-file preflight ONLY: actual
GPU correctness and latency have NOT been run under the held lease.
The runner now fails closed on missing mode/row capability and
records build metadata hash in each timing receipt. 9 dedicated
card-free tests and 92 XDP tests pass after the correction.

## CPU-only prebuild: resumable VTP05 campaign

The companion `tools/lab/radiance/xtx27_quant_campaign.py` builds
a **fixed per-case manifest** and tests ABI/architecture provenance
against the already-built Radiance binary and plugins. No GPU is
touched by `plan`, `prepare`, `next` or `status`.
Its `run-next --run` performs exactly one case through the
exclusive-lease `xtx27_quant_kernel.py` runner, verifies the matching
receipt and keeps unsuccessful attempts visible. It never silently
skips failures or launches an uncontrolled multi-hour sweep.

Candidate matched prebuilt (replace with a newer verified build
only if its kernel/compiler/ABI evidence agrees):

```sh
BUILD="${BC_WORK_ROOT}/builds/f17f21fda44979e9abe84f13e96b8896/08b5dd81b4874390bf0cbf489db0d7df"
CAMPAIGN="${BC_WORK_ROOT}/xtx27-prefill/quant-shape/campaigns/pilot-v1"

python3 tools/lab/radiance/xtx27_quant_campaign.py --phase plan --tier pilot
python3 tools/lab/radiance/xtx27_quant_campaign.py --phase prepare \
  --tier pilot --build-dir "$BUILD" --campaign-dir "$CAMPAIGN"
python3 tools/lab/radiance/xtx27_quant_campaign.py --phase next --campaign-dir "$CAMPAIGN"

# Only once the GPU is free, explicitly run ONE case at a time:
python3 tools/lab/radiance/xtx27_quant_campaign.py --phase run-next \
  --run --campaign-dir "$CAMPAIGN"

python3 tools/lab/radiance/xtx27_quant_campaign.py --phase status --campaign-dir "$CAMPAIGN"
```

The five *independent*, progressively justified tiers are:
`pilot` 12 cases (3 physical cards × 2 precisions × 2 FFN
projections at M4); `decode` 24 (M1 and M8);
`placement` 72 (three legal ratios at M4/M8);
`prefill` 48 (two ratios at M64/M512); and `r9700-local`
48 (combined versus each logical R9700 shard).
Do **not** start the latter tiers before reviewing the pilot
kernel selection and numerical correctness. No tier launches on
preparation or automatically continues after a lease failure.

The read-only build preflight verifies the selected binary supports
both `--perfiqr` and `--perfw8`; both physical plugin families
contain the required narrow/tiled kernels; main CMake records
`RAD_GPU_TARGETS=gfx1201`; component records
`R3_TARGETS=gfx1100`; and the RADIANCE_SRC path. It records
artifact SHA-256 digests and rejects mismatched builds, modified
scripts or corrupt/duplicated per-case receipts.

After each physical rank has *both* FFN projections for the
same quant, ratio and M, `status` computes an explicitly labelled
**ideal FFN-only barrier lower bound** as max(gate/up device time)
+ max(down device time). It also reports the per-card serial workload
and, only when separately measured, R9700 physical-combined
versus logical-two-shard *sequential* results. It is not an engine
latency, interleaving speedup or tokens/s estimate.

**CPU-only gate:** 99 XDP tests passed, including 7 new
campaign tests and 9 VTP05 harness tests; Ruff clean. No
new Q6/Q8 GPU timings were recorded during the active
Flash-Next exclusive GPU lease.

### GPU-free runner safety audit (2026-11-10)

Before hardware access, the campaign runner was hardened: on Linux it
uses **ROCR_VISIBLE_DEVICES alone**, rejects *any* inherited visibility
mask (ROCR/HIP/CUDA/GPU_DEVICE_ORDINAL), and resolves the physical index
from the full unmasked GPU inventory only after acquiring the lease.
AMD ROCm 7.2 documentation recommends the ROCr mask on Linux; setting
HIP_VISIBLE_DEVICES to the same physical ID after ROCr remapping is
ambiguous. The selftest output must now include an exact N/K/M header
matching the requested geometry as well as the selected timing row.
Large plugin capability checks use read-only mapped-file searches
rather than copying ~200 MiB of binary into the Python heap.

Six new mock-GPU receipt/negative-case tests ensure target identity,
strict output geometry and no accidental GPU access. The full XDP
card-free suite passes **105 tests** and Ruff remains clean. These
checks do not qualify numerical output or guarantee a successful live
HIP run; actual hardware remains lease-blocked.

The original pilot-v1 manifest is intentionally invalidated by the
new runner SHA and must not be reused. Recreate a distinct immutable
**pilot-v2** manifest *after* committing/rebasing these safety changes,
and never overwrite pilot-v1 or accept its receipts in pilot-v2.

### Hardware screening update

The first M4 runs were completed after the exclusive lease cleared.
**W8A8 executes on gfx1100 and gfx1201**; **IQ6R-V4 fails at
relayout on both gfx1100 cards** while working on gfx1201.
The actual timings, source receipts, failure status and isolated
remediation are recorded in [VTP07](VTP07.md). IQ6R cannot advance
to 3-card virtual TP4 without fixing this gfx11 blocker.

## W8A8 three-physical-card screening: matched 9-iteration runs (2026-11-10)

All 36 additional model-shaped W8A8 timing cases below passed the
bench tool's N/K/M/kernel-row validation, using the pinned original
libr3/libr4d build and exclusive per-case GPU leases. Receipts are
under `BC_WORK_ROOT/xtx27-prefill/quant-shape/placement-w8a8-20261110/runs`.
These are **warm-weight synthetic INT8 kernels only**, not Qwen3.8
checkpoint conversion, numerical quality, complete 4-stream execution,
KV allocation or end-to-end throughput.

Physical GEMM split, FFN intermediate tiled to multiples of 256,
gate/up and down projections run independently, M=4:

| Logical ratios | Approx. physical weights | max(gate/up) us | max(down) us | ideal FFN-only barrier us |
| --- | --- | ---: | ---: | ---: |
| capacity 30/30/20/20 | 29.4/29.4/41.2% | 83.1 | 55.2 | **138.3** |
| intermediate 35/35/15/15 | 35.3/35.3/29.4% | 83.6 | 65.6 | **149.2** |
| bandwidth 37.5/37.5/12.5/12.5 | 38.2/38.2/23.5% | 97.3 | 76.3 | **173.6** |

The original capacity split is the preferred **W8A8 kernel-only M4**
starting point, *not* a validated runtime configuration. The R9700
has useful compute headroom at this split; moving FFN weight off it
makes the XTXs the slower barrier. Attention KV has a different
physical-head allocation and may change the best feasible choice.

Capacity split, all 3 GPUs and both FFN projections, 9 iterations:

| Model rows M | Ideal FFN-only barrier | Physical summed kernel us XTX0 / XTX1 / R9700 |
| --- | ---: | --- |
| 1 | 138.6 us | 134.9 / 138.6 / 103.9 |
| 4 | 138.3 us | 134.5 / 138.3 / 102.7 |
| 8 | 138.6 us | 135.0 / 138.6 / 106.5 |
| 64 | 314.5 us | 305.2 / 314.5 / 197.3 |

M=64 uses the tiled kernel; M1/4/8 use the narrow decode kernel.
Each sample is a short single benchmark, not a stable p95.
**Only compare within the W8A8 warm-weight format**; IQ6R-V4's
256-MiB cold-flush timing measures different weight residency.

VTP07 records the gfx11 IQ6R host-source omission and corrective
scratch build. Full 27B model/KV validation remains under VTP06.

## W8A8 native kernel numerical-reference qualification (2026-11-10)

In addition to the 36 native W8A8 shape timings above, both
`gemm_i8a8_nt_m16` and `gemm_i8a8_tiled` were checked using
Radiance's independent numerical selftests on **both** gfx1100 XTXs
and the gfx1201 R9700. The gfx1100 cards ran BigCherry's
gfx11-rewritten `r3_selftest` against the original full `libr3.so`;
gfx1201 ran the matching pinned `r4d_selftest` against the original
`libr4d.so`. Each case acquired and released the global GPU lease.
The scratch wrapper records argv, selected physical index, object
SHA-256, return code, and complete stdout/stderr under
`${BC_WORK_ROOT}/xtx27-prefill/iqr-layout-fix/w8-oracle-runs/`.

| Engine test | XTX0 checks | XTX1 checks | R9700 checks |
| --- | ---: | ---: | ---: |
| `gemm_i8a8_nt_m16` | 13 pass / 0 fail | 13 / 0 | 13 / 0 |
| `gemm_i8a8_tiled` | 8 pass / 0 fail | 8 / 0 | 8 / 0 |
| **Total** | **21 / 0** | **21 / 0** | **21 / 0** |

**All 63 numeric selftest checks passed** with zero exit status.
The selftest includes several shapes per kernel, including
intermediate matrices, but this is not a 27B model's full per-layer
BF16-vs-quantised quality/accuracy comparison. We have not established
KLD, maximum KV, multi-rank numerical or fused full-FFN parity;
those gates remain in VTP06 and VTP03. IQ6R cold and W8A8 warm
comparisons remain intentionally non-equivalent.
