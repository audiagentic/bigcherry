# RAD50 real FP16 CPU combine into GPU-local DMA-BUF — Gate 3b

Date: 2026-11-09. Existing RAD50 branch `exp/rad42-cpu-bar-push`. Lab-only evidence, no engine change.

## What was exercised

The persistent separate CPU-only child now calls the **production**
`combine_row_f16` (AVX2/F16C, 2 BF16 helpers, width 2,560) directly into
the selected destination. The existing baseline combines into shared HIP-
registered host RAM, which the GPU reads over PCIe. The candidate combines
**directly** into an exported GPU-local DMA-BUF mmap, with
`DMA_BUF_IOCTL_SYNC START/END` around writes; the GPU reads local VRAM.

A *separate device-side bitwise oracle* checks **every FP16 element** and
the first-word `kCombineFallback=0x7e01` sentinel. Four deterministic input
profiles rotate by row/generation: 1+2, helper mask selecting only helper 1,
overflowing 65536+1, and canonical NaN. These are synthetic BF16 inputs
but the CPU combine function is the real production code. Both paths use
the same persistent child, mapping/FDs, alternating order and GPU checker.
Each row is 5,120 bytes. The same destination is reused for every
generation; all tests exited 0 after GPU verification.

## Completed 30-iteration, five-size direct-combine sweeps

Each figure is median serial producer + GPU verification completion (microseconds),
30 measured iterations plus 3 warmups, per path. **Not actual inference.**

| Rows | Bytes | XTX0 pull / push (µs) | XTX0 ratio | XTX1 pull / push (µs) | XTX1 ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 5,120 | 73.81 / 75.24 | 0.981× | 71.58 / 73.21 | 0.978× |
| 16 | 81,920 | 84.92 / 78.08 | 1.088× | 83.18 / 76.84 | 1.083× |
| 64 | 327,680 | 107.83 / 93.09 | 1.158× | 105.99 / 91.15 | 1.163× |
| 256 | 1,310,720 | 265.23 / 203.58 | 1.303× | 263.62 / 201.94 | 1.305× |
| 768 | 3,932,160 | 1265.74 / 998.91 | 1.267× | 684.85 / 703.53 | 0.973× |

**Important:** the 768-row 30-sample result varies considerably between
cards and runs. This is not evidence of a stable ~27% gain; XTX1 in this
run is slightly slower. XTX0's p95 also regressed for this particular
30-sample run. Preserve these observations as diagnostics, not outliers
to hide.

## Independent 100-iteration 768-row confirmation

| GPU | Host-mapped pull p50 | Direct GPU-local push p50 | Difference | Ratio |
|---|---:|---:|---:|---:|
| XTX0 | 700.87 µs | 621.96 µs | 78.91 µs | 1.127× |
| XTX1 | 703.44 µs | 622.16 µs | 81.28 µs | 1.131× |

Both 100-iteration runs passed **every GPU FP16 word** through 100 distinct
generation values. XTX0 p95 was 944.33 µs vs 667.22 µs; XTX1 p95 was
926.59 µs vs 672.04 µs. These are short, uncontended synthetic run
distributions, not serving p95/p99.

At 768 rows, 100-iteration median CPU combine time was 313.50/314.49 µs
writing host RAM versus 363.45/362.76 µs writing GPU VRAM (about 49–50 µs
extra CPU write latency). The GPU's wall-clock read/verification completion
was 380.12/381.49 µs from mapped host versus 251.74/252.60 µs from GPU
VRAM. The net serial difference was ~79–81 µs, not the ~300 µs from
synthetic scalar writes. DMA-BUF SYNC START+END medians totalled
~0.39/0.45 µs.

## Staged-combine controls

The separate staged variant calls the same production combine into ordinary
host staging memory and then copies the whole result to the consumer
destination. At 768 rows, 30-iteration GPU-push versus pull was 1.131×
on XTX0 (961.71 vs 1087.99 µs) and 1.073× on XTX1
(949.67 vs 1019.29 µs). Staging preparation alone costs ~300 µs;
the subsequent 3.75 MiB DMA-BUF copy costs ~361–362 µs. Comparing
different run medians is **not** a controlled direct-vs-staged
optimization result; a same-session paired method test remains necessary.

## Interpretation and next gates

- Small single-row chunks still lose roughly 1–2 µs to CPU-push; 16+ row
  groups show benefits in the serial test. A safe deployment might need a
  shape-dependent fallback, but only after live critical-path modeling.
- Repeated-generation **serial** reuse is demonstrated, but stale-data under
  a concurrently running GPU consumer, crash/restart semantics, FD lifetime
  and per-rank exact-once publication are **not** established.
- CPU combine runs against deterministic BF16 profiles, not captured Qwen
  MoE activations. The GPU kernel is a full-word verifier, not Radiance's
  fused add/gather.
- No claim of a 1.13× inference speedup. Actual RAD42 overlap of next CPU
  combine with current GPU pull may reduce or reverse the net benefit.
- Next: bounded GPU-already-waiting publication canary and multi-process
  chunk/ready protocol; then controlled contended A/B and actual TP2
  prefill/decode/KL parity. Keep the existing host-GPU-pull fallback.
- Treat partial or failed logs as **provisional observations**, not errors
  to discard. `ab_analyze.py --provisional` keeps per-path timing and
  failure markers; strict mode requires post-cleanup `status=0`.
- No new worktree, production/vendor changes, push or PR.

## Raw evidence

The first 1-row 10-iteration, 768-row 10-iteration, five-size
30-iteration direct/staged sweeps and 768-row 100-iteration runs for
each XTX are preserved in `evidence/2026-11-09/` as
`rad50-combine-xtx*.log`, with full analyzer output in
`rad50-combine-o2-four-sweeps-analyzed.txt`. Original raw command
captures remain in `${BC_WORK_ROOT}/builds/rad42-bar-push/`.
