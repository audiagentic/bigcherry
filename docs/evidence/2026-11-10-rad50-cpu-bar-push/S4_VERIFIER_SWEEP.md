# RAD50 S4 verifier-grid sweep (2026-11-09)

**Branch:** exp/rad42-cpu-bar-push. **Worktree:** existing RAD50 project worktree. **Scope:** isolated lab-only S4 GPU verifier sizing, no production/vendor/serving changes, no P2P.

## Reproducing the measurement

Rebuilt the single existing allowlisted test binary at OPT=-O2 for gfx1100 and gfx1201. HIP 7.2.53211 / ROCm 7.2.4, GPUs 0 and 1 AMD Radeon RX 7900 XTX gfx1100. Binary SHA256:
`b86c655d21a0976eb659e3a36abe6738e379c756f454c81b4fc15cb9321b1e45`.

S4 adds a bounded `--verifier-blocks 1|32|64|128|256` selector to `dmabuf-live-parallel` and `dmabuf-live-relay`; **256 remains default**. It changes only the verifier grid, not the one-lane already-running GPU waiter, production AVX2/F16C FP16 CPU combine, DMA-BUF stores/synchronization, generation ticketing, GPU checks or cleanup protocol. Verifier launch uses 256 threads/block. The runner rejects selector use in unrelated modes and unsupported/duplicate values. Log context/sample/summary record the grid, and strict analysis requires them to match (older complete 256-block logs remain readable). For example:

`rad50-tester --test dmabuf-live-parallel --device 0 --iterations 100 --rows 768 --verifier-blocks 256 --lease-confirmed`

Only run with a separately acquired exclusive GPU lease and a no-foreign-KFD-process preflight; the CLI flag is not lease acquisition. Raw captured evidence is in `evidence/2026-11-09/rad50-s4-*.log`. All generation results include full GPU FP16 word verification, exact ticket matching, done/ACK, buffer reuse discipline, and post-cleanup completion. This is not a fused model consumer, a general memory-ordering proof, or an inference throughput result.

## Completed direct-child-publication runs

Values are run-level median and p95 elapsed microseconds. These are **not randomized interleaved A/B**; concurrent RAD49 stock Radiance build caused CPU/load variation, so the grids have not been statistically ranked for production.

| GPU | Rows | Blocks | Generations | p50 us | p95 us | CPU combine p50 us | Post-ACK p50 us |
|---|---:|---:|---:|---:|---:|---:|---:|
| XTX0 | 768 | 1 | 10 | 7499.35 | 7533.17 | 365.33 | 7016.43 |
| XTX0 | 768 | 32 | 30 | 679.44 | 813.33 | 364.37 | 282.16 |
| XTX0 | 768 | 64 | 30 | 618.09 | 1741.99 | 364.72 | 174.27 |
| XTX0 | 768 | 128 | 100 | 479.48 | 1698.41 | 365.48 | 82.81 |
| XTX0 | 768 | 256 | 100 | 464.23 | 520.50 | 363.00 | 69.71 |
| XTX1 | 768 | 64 | 30 | 530.38 | 833.31 | 363.80 | 117.07 |
| XTX1 | 768 | 128 | 100 | 492.02 | 608.55 | 366.51 | 85.63 |
| XTX1 | 768 | 256 | 100 | 526.39 | 708.70 | 406.27 | 73.40 |
| XTX0 | 1 | 1 | 30 | 137.05 | 145.68 | 0.73 | 95.87 |
| XTX0 | 1 | 32 | 30 | 123.57 | 129.37 | 0.61 | 83.47 |
| XTX0 | 1 | 128 | 30 | 121.17 | 129.20 | 0.62 | 82.27 |
| XTX0 | 1 | 256 | 30 | 125.83 | 132.92 | 0.63 | 83.83 |
| XTX1 | 1 | 128 | 30 | 127.07 | 133.20 | 0.61 | 85.42 |
| XTX1 | 1 | 256 | 30 | 127.47 | 177.75 | 0.62 | 85.22 |

Other successful 10/30-generation comparisons are retained as separate logs. The 256-block production-like baseline still has the fastest measured post-ACK median in the 100-generation 768-row comparisons on both XTXs. The XTX1 total with 256 blocks was worse than with 128 blocks because its measured CPU combine rose to 406.27 us versus 366.51 us: this is not evidence that 128 blocks win on the GPU itself. One-block verification scales disastrously at 768 rows but **not** because of the single-lane waiter; it scans the payload with just one block.

## Control, failures, and evidence integrity

- Coherent parent relay with 128 verifier blocks passed exact 768-row data/ticket checking for 10 generations on both cards: XTX0 p50 604.09 us, XTX1 p50 2045.72 us. On XTX1, combine itself measured 1778.34 us; treat the outlier as CPU-contention evidence, not a DMA-BUF or relay speed claim.
- **22 complete S4 logs passed the strict `slice_analyze.py` parser; one local capture was interrupted mid-summary**, preserved as `rad50-s4-xtx0-768-64blocks-10-PARTIAL.log`. Even though individual rows exist, the file must be treated as provisional and **not** counted as a completed run.
- Card-free `unittest`: 95 passed. Unified tester CLI `--self-check`: 75/75 passed. HIP fat binary `make` with `OPT=-O2` completed; `git diff --check` clean.
- All GPU tests were conducted under a separate exclusive RAD50 lease after no-KFD preflight; RAD50 released the lease. RAD49 reacquired it for its own work. Do not interfere with RAD49.

## Decision and next gates

**KEEP default 256 verifier blocks.** Do not set a shape-dependent switch on this evidence. If investigating 128 versus 256 further, acquire another exclusive lease, eliminate overlapping CPU builds, alternate randomized run order on the same card, repeat sessions, collect all stage distributions and CPU frequency/affinity. Avoid claiming p99 from 100 samples.

**S1 first:** isolate CPU BF16-to-FP16 combine arithmetic from destination BAR stores and PCIe/cache effects using real helper row masks/heat, multi-helper counts, batch layout, AVX2/F16C, CPU affinity/NUMA, and host-mapped versus direct DMA-BUF versus staged. Preserve byte-exact parity for masks, fallback sentinel, NaN, signed zeros, subnormals, overflow and helper order. The ~363-us 768-row combine cost dominates stable live timing.

**S0/S3 fault safety:** duplicate/stale/published-before-data tickets, child exit/restart, FD lifecycle, and bounded waiter without leaking mappings. **S4 production-equivalence:** compare a real Radiance fused gather/add consumer (not the parallel full-payload tester) and actual TP2 once-only behavior. Then controlled host-read contention, overlap modelling, real prefill/decode, KL and p95/p99 before RAD51 is eligible. Existing mapped-host GPU-pull remains the production fallback; no serving integration authorized.

For portability only the absolute tester path in each captured command header was normalized to ${BC_WORK_ROOT}; GPU samples, summaries, failures, exit-code fields and diagnostics are unchanged.
