# RHA08 reusable BC build-type matrix

This is the first complete execution of the configured 27B dual-XTX build-type
matrix through the declarative `runtime-matrix` adapter and the maintained
server-bench runner. It is a build/framework qualification artifact, not a
production parity claim.

- Matrix status: `completed`, 5/5 cells; see `status.json`, `events.jsonl`,
  `summary.json`, and `advisories.json`.
- Model: Qwen3.8-27B Q8_0, tensor split over gfx1100 devices 0 and 1.
- Every cell: return code 0, clean SIGINT shutdown, physical RCCL identity
  verified (`0000:03:00.0`, `0000:06:00.0`).
- Runner: `bench/run_bench.py --bench-type server-bench`; no llama-bench.
- Raw record/tune/replay-diagnostic sidecars are retained at the matrix root.

## Cell results (one sample, tokens/s)

| build type | pp512 | pp2048 | tg128 | tg512 | role |
|---|---:|---:|---:|---:|---|
| control | 924.42 | 1288.56 | 33.74 | 33.87 | diagnostic/framework control |
| record | 916.03 | 1288.85 | 33.92 | 34.07 | signature observation |
| tune | 96.58 | 894.58 | 33.88 | 33.92 | candidate measurement |
| replay | 870.96 | 1289.00 | 34.02 | 34.01 | cache replay observation |
| replay-diagnostic | 925.71 | 1286.69 | 34.03 | 34.33 | replay hit attribution |

The control/record/tune/replay-diagnostic builds contain instrumentation and
the replay/tune cells have framework/cache behavior that is intentionally not
comparable to the production stock/native/replay matrix in RHA04. In
particular, the tune and replay pp512 values must not be interpreted as a
production regression. RHA04's diagnostics-off, identity-bound parity result
remains the authoritative production comparison.

The tune cell emitted 58 measurement records plus a journal; record emitted 58
records; replay-diagnostic emitted 57 hit-log records. The full raw server,
preflight, bench, progress, and sidecar artifacts are retained here.
