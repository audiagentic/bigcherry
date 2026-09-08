# RD87 hipBLASLt offline-tuning oracle

Plan item: RD87
Status: answered (negative finding)
Owner: Claude Sonnet 5 (session)
Question state: answered

## Question

Does AMD's vendor hipBLASLt GEMM library -- run purely as an offline
oracle via `hipblaslt-bench`, with no llama.cpp dispatch-path
integration -- beat BigCherry's existing native/tuned MMVQ dispatch for
the real GEMM shapes a live decode workload actually produces?

## Inputs

- Real captured dispatch shapes + native/tuned timing, extracted from a
  live BigCherry tuning campaign: `~/.cache/bigcherry/tune-campaigns/
  hi171-27b-dual-reexport-20260908/tune.measurements.jsonl` on Brutus
  (dual-gfx1100, tierM/L Qwen-class decode workload -- 55 unique
  dispatch shapes after dedup, mostly small-N (1-5) MMVQ/GEMV decode
  ops plus a few BLAS shapes).
- `hipblaslt-bench` built from source at
  `vendor/rocm-libraries/projects/hipblaslt/hipblaslt-install/bin/`
  (see RD87.md history for the build path), run with
  `--precision f16_r` (hipBLASLt has no native quantized-type GEMM;
  f16 is the closest available dense proxy -- see Disposition).

## Outputs

- `rd87_shapes.sample.json` -- extracted real (K,M,N,type) shapes with
  their real native/tuned median_us from the source campaign.
- `rd87_comparison.csv` -- per-shape native_us vs hipblaslt heuristic_us
  vs hipblaslt best-of-all-solutions_us, with ratios.
- Raw `hipblaslt-bench` logs are not retained in-repo (regenerable via
  `run_bench.sh`); the CSV is the durable record.

## Runtime

GPU required: yes (real gfx1100 hardware for `hipblaslt-bench`)
Real compilation required: no (uses the already-built hipBLASLt bench
client from RD87's earlier tooling-blocker work)
Mutates canonical BigCherry state: no

## Safety

- Canonical-state mutation: none. Reads an existing campaign file;
  writes only to its own throwaway output directory.
- `run_bench.sh` and `extract_shapes.py` are diagnostic drivers, not
  libraries -- do not import from production/tests.

## Disposition

**Answered, 2026-09-09 (corrected same day)**: the first pass of this
analysis had a real bug -- the section-header match used
`header.endswith('all')`, but headers keep their trailing `' ==='`
(e.g. `"k5120_m5120_n1 all ==="`), so the match silently always failed
and every `all_ratio` (native vs hipBLASLt's best-of-every-solution
run) column came out empty. That produced a false blanket "no-go" that
hid every shape where hipBLASLt's exhaustive solution search actually
beat native. Fixed in `analyze_results.py` (strips the punctuation
before matching); `rd87_comparison.csv` now carries the real numbers.

**Corrected finding, split by native code path:**

- **17 of 55 shapes** show hipBLASLt's best-of-all-solutions run
  genuinely faster than native (`all_ratio < 1.0`).
- Every clear, sizeable win (`all_ratio` 0.37-0.83, i.e. hipBLASLt
  1.2x-2.7x FASTER) is on a shape whose native path is already
  `blas:native:v1` -- small-K (K=64/256), larger-N (N>=8) shapes that
  llama.cpp already routes through rocBLAS, not through BigCherry's own
  hand-written MMVQ/MMQ kernels. Best case: K=256,M=256,N=24 -- native
  18.02us vs hipBLASLt-tuned 6.58us (2.74x faster).
- The remaining `all_ratio < 1.0` shapes (K=5120, `mmvq:native:v1`
  path) are all within ~0.89-0.99x -- effectively noise-level parity,
  not real wins.
- Every shape where native already uses tiny-N (N<=5) MMVQ decode and
  hipBLASLt is clearly worse (`all_ratio` well above 1.0, up to 10x in
  heuristic mode) is exactly the same MMVQ-decode population as before
  -- that negative result stands unchanged.

Recommendation, corrected: **no-go on hipBLASLt for MMVQ decode
dispatch** (that negative result is real and survives the fix) --
**but a real, evidence-backed go-signal for BLAS-family solution
search on the shapes llama.cpp already routes through rocBLAS**
(`blas:native:v1`). That second finding lands squarely in HI173's
scope (BLAS-family candidate search over rocBLAS/Tensile solutions,
filed 2026-09-08) rather than RD87's own (hipBLASLt as an MMVQ-decode
replacement). See `rd87_comparison.csv` for the full per-shape table.
