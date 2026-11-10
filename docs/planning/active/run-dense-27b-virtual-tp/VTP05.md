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
2. Reuse Radiance's real selftest performance hooks, not another pretend Q6 kernel: --perfiqr N K M 6 [iters] for the native IQ6R V4 trellis decoder and GEMM; --perfw8 N K M [iters] for the native W8A8 and additional 8-bit kernels. The selected rows must be matched to the intended dtype and M. Optionally --cold 256 for realistic cold-weight timing.
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
