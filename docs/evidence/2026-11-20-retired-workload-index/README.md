# Retired workload index: what the 138 pre-record rows showed

Before run records (MRC01-MRC06) `bigcherry workload run` wrote one `result.json` per run under the work root's
`results/workloads/` and an index row. The manifests those rows named were purged on 2026-11-17, so the rows cannot
be turned into run records with content provenance, and the project rule is no legacy readers. This page keeps what
they showed; the folders and the listing code were then deleted.

Rows: 138, 2026-11-11 to 2026-11-12 (dates of the run folders), all `source=measured`, all on kernel
7.0.13-gfxoff, layouts flashnext-4gpu (the stock four-card Flash-Next) and flashnext-2xtx-rdna2-tail. Policy is the
live expert-placement policy state recorded in each row. Medians of each run's own median; n is the number of runs.

| workload | layout | policy | metric | n | min | median | max |
|---|---|---|---|---|---|---|---|
| cold-27k-ttft | flashnext-2xtx-rdna2-tail | off | ttft_ms | 1 | 13138.9 | 13138.9 | 13138.9 |
| cold-27k-ttft | flashnext-2xtx-rdna2-tail | on | ttft_ms | 1 | 10061.2 | 10061.2 | 10061.2 |
| cold-27k-ttft | flashnext-4gpu | off | ttft_ms | 1 | 10937.4 | 10937.4 | 10937.4 |
| cold-27k-ttft | flashnext-4gpu | on | ttft_ms | 1 | 7533.9 | 7533.9 | 7533.9 |
| decode-1stream | flashnext-2xtx-rdna2-tail | off | decode_aggregate_tps | 1 | 104.8 | 104.8 | 104.8 |
| decode-1stream | flashnext-2xtx-rdna2-tail | off | decode_engine_tps | 1 | 130.9 | 130.9 | 130.9 |
| decode-1stream | flashnext-2xtx-rdna2-tail | off | ttft_ms | 1 | 478.7 | 478.7 | 478.7 |
| decode-1stream | flashnext-2xtx-rdna2-tail | on | decode_aggregate_tps | 1 | 104.9 | 104.9 | 104.9 |
| decode-1stream | flashnext-2xtx-rdna2-tail | on | decode_engine_tps | 1 | 130.9 | 130.9 | 130.9 |
| decode-1stream | flashnext-2xtx-rdna2-tail | on | ttft_ms | 1 | 478.0 | 478.0 | 478.0 |
| decode-1stream | flashnext-4gpu | off | decode_aggregate_tps | 7 | 92.3 | 92.5 | 92.8 |
| decode-1stream | flashnext-4gpu | off | decode_engine_tps | 7 | 108.1 | 108.5 | 108.9 |
| decode-1stream | flashnext-4gpu | off | ttft_ms | 7 | 416.1 | 418.0 | 425.6 |
| decode-1stream | flashnext-4gpu | on | decode_aggregate_tps | 1 | 91.8 | 91.8 | 91.8 |
| decode-1stream | flashnext-4gpu | on | decode_engine_tps | 1 | 107.4 | 107.4 | 107.4 |
| decode-1stream | flashnext-4gpu | on | ttft_ms | 1 | 415.6 | 415.6 | 415.6 |
| decode-6stream | flashnext-2xtx-rdna2-tail | off | decode_aggregate_tps | 1 | 212.6 | 212.6 | 212.6 |
| decode-6stream | flashnext-2xtx-rdna2-tail | off | decode_engine_tps | 1 | 329.2 | 329.2 | 329.2 |
| decode-6stream | flashnext-2xtx-rdna2-tail | off | ttft_ms | 1 | 2534.9 | 2534.9 | 2534.9 |
| decode-6stream | flashnext-2xtx-rdna2-tail | on | decode_aggregate_tps | 1 | 227.2 | 227.2 | 227.2 |
| decode-6stream | flashnext-2xtx-rdna2-tail | on | decode_engine_tps | 1 | 365.0 | 365.0 | 365.0 |
| decode-6stream | flashnext-2xtx-rdna2-tail | on | ttft_ms | 1 | 2526.1 | 2526.1 | 2526.1 |
| decode-6stream | flashnext-4gpu | off | decode_aggregate_tps | 7 | 225.3 | 230.6 | 237.3 |
| decode-6stream | flashnext-4gpu | off | decode_engine_tps | 7 | 326.1 | 336.7 | 351.2 |
| decode-6stream | flashnext-4gpu | off | ttft_ms | 7 | 2070.5 | 2091.0 | 2093.7 |
| decode-6stream | flashnext-4gpu | on | decode_aggregate_tps | 35 | 234.9 | 243.8 | 254.7 |
| decode-6stream | flashnext-4gpu | on | decode_engine_tps | 35 | 320.1 | 342.1 | 358.1 |
| decode-6stream | flashnext-4gpu | on | ttft_ms | 35 | 1668.7 | 1754.4 | 1934.2 |
| prefill-doc-44k | flashnext-2xtx-rdna2-tail | off | prefill_tps | 1 | 3147.7 | 3147.7 | 3147.7 |
| prefill-doc-44k | flashnext-2xtx-rdna2-tail | on | prefill_tps | 1 | 3151.0 | 3151.0 | 3151.0 |
| prefill-doc-44k | flashnext-4gpu | off | prefill_tps | 7 | 3700.3 | 3719.2 | 3744.3 |
| prefill-doc-44k | flashnext-4gpu | on | prefill_tps | 35 | 3587.6 | 3701.9 | 3794.1 |
| pwr-phases | flashnext-4gpu | off | decode_tps | 1 | 87.3 | 87.3 | 87.3 |
| pwr-phases | flashnext-4gpu | off | prefill_tps | 1 | 2597.6 | 2597.6 | 2597.6 |
| pwr-phases | flashnext-4gpu | on | decode_tps | 35 | 76.0 | 88.8 | 92.5 |
| pwr-phases | flashnext-4gpu | on | prefill_tps | 35 | 2761.8 | 3117.2 | 3320.7 |

Units: ttft_ms in ms; every *_tps in tokens/s. `decode_aggregate_tps` is the clients' total, `decode_engine_tps` the
engine's own count (it includes drafts accepted per step).

## Notable points

- The cold 27K prefill record: TTFT 7.53 s (flashnext-4gpu, policy on; the best of the four cold rows) on kernel 7.0.13-gfxoff; the four rows range
  7.5-13.1 s depending on layout and policy.
- 44K document prefill on flashnext-4gpu: about 3.7 ktok/s with the policy on (n=36 rows), 3.1-3.8 ktok/s over all rows.
- Decode: 1 stream about 92.6 tok/s aggregate (108.5 engine) with TTFT near 418 ms; 6 streams about 242 tok/s
  aggregate (340 engine) with TTFT near 1.77 s.
- pwr-phases (the power-phase workload, 36 rows): decode 88.6 tok/s and prefill 3,114 tok/s median.

These are history for orientation only. They were taken under older builds, so they are not baselines; the
baselines are the run records and `docs/reference/engines/radiance/baselines/`.
