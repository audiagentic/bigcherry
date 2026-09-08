# HI168 end-to-end native/replay matrix

Observed on the configured build server (`BC_HOST`) on 2026-09-07/08.  This
bundle records the completed owner-requested matrix using the maintained
`ab-benchmark` server endpoint runner and `server-bench` workload, never
`llama-bench`.

Raw receipts and server logs are retained under the ignored artifact area:
`artifacts/hi168-e2e-results/`.  Each run contains 18 cells: six balanced
orders, three arms (`stock`, `native`, `replay`).  All completed cells returned
zero and shut down through the HTTP endpoint without forced termination.

## Matrix

| run | model/topology | cells | result |
| --- | --- | ---: | --- |
| `9b-gpu0-three-arm-001` | Qwen3.5-9B Q6_K, GPU0 gfx1100 | 18 | complete |
| `9b-gpu1-three-arm-001` | Qwen3.5-9B Q6_K, GPU1 gfx1100 | 18 | complete |
| `9b-gpu2-three-arm-001` | Qwen3.5-9B Q6_K, GPU2 gfx1201 | 18 | complete |
| `9b-gpu3-three-arm-001` | Qwen3.5-9B Q6_K, GPU3 gfx1030 | 18 | complete |
| `27b-dual-three-arm-001` | Qwen3.8-27B Q8_0, dual GPU0+1 gfx1100 | 18 | complete |

Metrics are tokens/second; values below are the median of six cells per arm.

| run | arm | pp512 | pp2048 | tg128 | tg512 |
| --- | --- | ---: | ---: | ---: | ---: |
| 9B GPU0 | stock | 1263.465 | 2504.720 | 84.195 | 85.130 |
| 9B GPU0 | native | 1274.730 | 2514.735 | 84.345 | 85.120 |
| 9B GPU0 | replay | 1269.915 | 2517.235 | 84.370 | 85.085 |
| 9B GPU1 | stock | 1284.865 | 2508.855 | 84.525 | 85.250 |
| 9B GPU1 | native | 1286.170 | 2522.405 | 84.470 | 85.290 |
| 9B GPU1 | replay | 1285.170 | 2521.025 | 84.525 | 85.195 |
| 9B GPU2 | stock | 1239.950 | 2254.765 | 67.810 | 68.265 |
| 9B GPU2 | native | 1408.625 | 2799.170 | 67.850 | 68.280 |
| 9B GPU2 | replay | 1398.655 | 2800.940 | 67.865 | 68.360 |
| 9B GPU3 | stock | 906.915 | 1281.875 | 55.315 | 55.865 |
| 9B GPU3 | native | 894.990 | 1281.715 | 55.310 | 55.850 |
| 9B GPU3 | replay | 880.040 | 1281.850 | 55.250 | 55.800 |
| 27B dual XTX | stock | 929.710 | 1284.890 | 34.040 | 34.290 |
| 27B dual XTX | native | 929.055 | 1286.180 | 34.050 | 34.280 |
| 27B dual XTX | replay | 928.775 | 1286.225 | 34.160 | 34.390 |

## Interpretation

The dual-XTX 27B result is native-parity: BC native is within approximately
0.1% of stock on all four metrics, and replay is within approximately 0.3%
of native.  This does not demonstrate a replay win.

The two XTX single-GPU runs are also near parity.  GPU3 shows a consistent
replay disadvantage in prompt processing (about 1.7% at pp512), so replay is
not a universal promotion.  GPU2's stock prompt numbers are anomalously low
relative to native/replay and must be treated as a stock execution/build
anomaly, not as a tuning gain.

These are exploratory performance comparisons, not decision-grade physical
execution evidence: the runner recorded `performance_admitted=false` because
the server logs did not expose positive physical-device membership.  The
receipts do prove successful ROCm-shaped server startup, balanced ordering,
metric completion, and graceful shutdown.  They do not prove that every cell
was physically executed on the expected PCI device.  No lifecycle promotion
or claim of a universal replay benefit is made.

Exact build IDs, binary hashes, source slices, cache identities, schedules,
per-cell metrics, shutdown records, and logs are in the raw `run.json` files
under `artifacts/hi168-e2e-results/`.
