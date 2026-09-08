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

**Answered (negative finding), 2026-09-09**: across all 55 real
captured decode-time dispatch shapes, `hipblaslt-bench` in heuristic
(default vendor pick) mode is **2x-10x SLOWER** than BigCherry's
existing native/tuned MMVQ dispatch. Even in `--algo_method all` mode
(library enumerates and reports every real kernel solution it has for
the shape, i.e. the best case hipBLASLt itself can offer) it reaches
only near-parity at best (~0.83x-0.99x on a handful of shapes) and is
still slower on the large majority -- never a clear win. See
`rd87_comparison.csv` for the full per-shape table.

Root cause (consistent with hipBLASLt's design target): these are all
memory-bound, tiny-N (N<=5) GEMV/skinny-GEMM decode shapes. hipBLASLt's
kernel library and heuristic selector are built for compute-bound,
large-N training/prefill GEMMs; the fixed per-call dispatch/heuristic
overhead of a generic GEMM library dominates at these sizes and is not
recovered even by exhaustive solution search. A dequantization step
(hipBLASLt has no native Q8_0 GEMM support -- this oracle used f16 as
the closest available proxy, which is itself an optimistic upper bound
for hipBLASLt) would only add further overhead on top of these numbers.

Recommendation: **no-go** on a real hipBLASLt dispatch-path integration
for decode-time GEMV shapes. RD87 should close as a real, evidence-backed
negative finding, not be reopened to chase a pass. If hipBLASLt is
worth revisiting at all, it would be for compute-bound, large-N
prefill/prompt-processing GEMMs specifically -- a different, separate
plan item, not a continuation of this one.
