# HL02 2026-11-14 — GPU-live DMA-BUF visibility under root-port contention

**Scope:** bounded, exclusive-lease laboratory verification of RAD50's existing CPU-only-child → GPU-local DMA-BUF path. This is additional **partial HL02 Gate C** evidence and a higher-sample **serial synthetic Gate D** control, **not** a three-rank simultaneous transfer, 10 MiB proof or production VTP38 host reduce.

## Provenance and setup

- Parent managed measurements and test rationale: [2026-11-13 HL02 host-attribute matrix](../2026-11-13-hl02-transfer-matrix/README.md), [2026-11-13 R9700 30-generation check](../2026-11-13-hl02-vtp38-consolidation/R9700-LIVE-2026-11-13.md), [RAD50 mapping/visibility evidence](../2026-11-10-rad50-cpu-bar-push/PROGRESS.md).
- Experimental source: `tools/lab/radiance/hl02-host-attributes/run_live_repeats.py` and `run_live_under_read.py`, invoking unchanged archived `rad50-tester`; `run_push_control.py` for 30-iteration serial comparisons.
- RAD50 binary SHA-256: `e1e349b2686dfa1d7d2192da523290fc97502cfa1c7874bcfa784f0aa93af92c`. Contending-reader executable SHA-256: `1468ffaaee3de7d9a1c81a6f401fc2018071cb9e298cbf7586e4803daa4efffb`. Exact source/test semantics are in the lab README and original RAD50 test documentation.
- Topology *for this run*: GPU ordinal 0 = primary PEG-half XTX; 1 = other PEG-half XTX; 2 = R9700 on separate CPU x4. Ordinals are run metadata, **not** machine-independent implementation constants.
- Hardware: original 3,932,160-byte (768 × 5,120) FP16 payload, separate persistent CPU-only helper writing a GPU-local exported DMA-BUF mapping, `DMA_BUF_IOCTL_SYNC`, a pre-armed GPU waiter and full-word GPU verification, 256 verifier blocks, system-scope ready/generation/ACK and reuse after done. The corrected device-scaled `wall_clock64()` watchdog is used.
- Every GPU run acquired the exclusive `gpu` lease and released it; KFD preflight showed no foreign GPU processes, and final KFD state had no processes. The primary-reader contention run used only an owned child GPU reader, which was terminated and GPU-verified after both receivers completed.
- Normalized machine-independent receipts: [gpu-live-summary.csv](gpu-live-summary.csv) and [push30-controls.csv](push30-controls.csv). Full original stdout/stderr and JSON run receipts were retained under the run-scoped ignored `artifacts/HL02-20261114-*` paths during testing. The managed CSVs preserve all complete-run totals and per-arm session medians without physical BDF or host paths.

## A. 100-generation GPU-live full-word correctness

Every card passed exactly **100/100 generations** in a separate run with zero mismatches and explicit tester `RAD50_PARALLEL_COMPLETE ... status=0`. The two victim cards were additionally tested with the primary XTX continuously issuing host-memory reads (measured actual background read rate **13.431 GB/s**). No two receiver pushes ran concurrently in this phase.

| Destination | Background | Passed generations | Payload bytes | Diagnostic p50 | Diagnostic p95 | Diagnostic p99 |
|---|---|---:|---:|---:|---:|---:|
| XTX first PEG half | Idle | 100/100 | 3,932,160 | 464.10 µs | 547.10 µs | 814.14 µs |
| XTX second PEG half | Idle | 100/100 | 3,932,160 | 462.26 µs | 543.32 µs | 804.46 µs |
| R9700 x4 | Idle | 100/100 | 3,932,160 | 819.45 µs | 919.18 µs | 1210.54 µs |
| XTX second PEG half | Primary XTX reading | 100/100 | 3,932,160 | 463.01 µs | 553.49 µs | 792.85 µs |
| R9700 x4 | Primary XTX reading | 100/100 | 3,932,160 | 820.80 µs | 919.89 µs | 1165.87 µs |

**Interpretation:** repeated CPU-written GPU-local DMA-BUF data and GPU-ready publication remained correct under the primary-reader contention trigger. Measured median live test duration did not materially degrade in the contention arm for either receiving card. This provides a more representative safety/control signal than isolated serial proof, but is still **one destination at a time**. p99 values are diagnostic for only 100 observations, not production SLO claims. The full live sequence includes synthetic FP16 combine, CPU scheduling, ready/ACK and GPU verifier; it does not isolate BAR link time or show inference improvement.

## B. 30-iteration existing RAD50 serial transfer control

To increase sampling beyond the earlier 10-iteration sessions, the original `dmabuf-ab --mode direct --bytes 4194304` comparator was run at **30 iterations per arm** for each affected receiver, in ABBA-like idle/read/write/idle or idle/write/read/idle order. The primary reader remained at full rate while each serial comparison executed, and every arm completed with RAD50 GPU correctness verification and exit 0. One 30-iteration session was run per receiver, so this supplements rather than replaces the previous three-session results.

| Destination | Background | GPU-pull sequence p50 | CPU-push sequence p50 | Paired speedup |
|---|---|---:|---:|---:|
| Second XTX | Idle (first) | 1054.52 µs | 734.83 µs | 1.435× |
| Second XTX | Primary writes | 1084.33 µs | 783.76 µs | 1.383× |
| Second XTX | Primary reads | 1356.50 µs | 734.25 µs | 1.847× |
| Second XTX | Idle (last) | 1057.16 µs | 791.75 µs | 1.335× |
| R9700 | Idle (first) | 1260.33 µs | 816.68 µs | 1.543× |
| R9700 | Primary reads | 1885.32 µs | 814.50 µs | 2.315× |
| R9700 | Primary writes | 1258.43 µs | 815.74 µs | 1.543× |
| R9700 | Idle (last) | 1246.82 µs | 814.38 µs | 1.531× |

**Interpretation:** the contended synthetic 4 MiB CPU-push bypass advantage remains at approximately **1.85×** on the affected XTX and **2.32×** on the R9700 with triple the per-arm sample count used in the previous sessions. The previously observed primary-read-specific GPU-pull penalty persists, while CPU-push latency remains near idle. These are serial synthetic transfer *sequence* times, not simultaneous producer-to-three-consumer bandwidth or TP3 end-to-end latency.

## Acceptance impact and next tests

- **Additional partial Gate C PASS:** 500 GPU-live generations total (300 idle and 200 under primary read), 3.93 MiB per generation, zero failures on all tested GPU paths. The XTX and R9700 CPU-writable GPU-local VMM DMA-BUF mechanism can be used for **further experiments**. It is not yet qualified for 10 MiB, lifetime/crash/restart, three simultaneous destinations or VTP38 production.
- **Additional Gate D control PASS:** 30-iteration serial RAD50 comparisons support the previous 4 MiB speedup under active primary GPU reads and separately show the primary-write control.
- **Still OPEN:** true 320 KiB–10 MiB three-way CPU BAR push with concurrent GPU partial producers, P-core/memory pressure, posted-write/GPU-ready evidence at max size and bounded multi-rank abort/drain, accurate CPU ordered BF16 reduction versus GPU two-shot, VTP36 improved-GPU A/B, numerical/stress gates.
- No production code or default transport was changed. The original GPU collective remains the production fallback.
