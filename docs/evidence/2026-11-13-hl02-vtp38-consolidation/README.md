# 2026-11-13 host-link slot swap and VTP36 GPU-candidate receipts

Managed evidence summary for HL01 (completed), HL02 (laboratory transfer methods), VTP36 (TP3 performance) and VTP38 (production TP3 transport). **These are existing measurements; this consolidation did not rerun the GPUs.**

## Source runs and methods

- Physical swap source records supplied with consolidation: `before.txt` (2026-11-13 05:42:09 UTC), `after-partial.txt` (07:19:56 UTC), `after.txt` (10:21:18 UTC). Full measured lines are preserved in [before-normalized.txt](before-normalized.txt), [after-partial-normalized.txt](after-partial-normalized.txt), [after-normalized.txt](after-normalized.txt): physical GPU hashes and enumerated BDF strings have been replaced by role labels, with all numeric transfer results and timestamps retained. Two physical XTX cards were exchanged between the primary and secondary PEG x8 halves. Solo GPU host reads/writes measured 64 MiB buffers for ~5 seconds; the competing background reader used a 256 MiB buffer over ~12 seconds, with the victim sampled while active. The partial configuration had just the formerly non-contending XTX on the primary PEG half. Full card identity/PCIe configuration is in HL01.
- Exact preserved [VTP36 candidate ABBA extract](vtp36-candidate-abba.txt): 2026-11-13 two complete int8 and MXFP4 ABBA comparisons; n=8 per stream/TTFT arm and n=4 for mixed-prompt arms. Candidate is rad0032 fused pieces + rad0033 write-once + duplex weights with `all_reduce:nblocks=4,nthreads=256,ts_pieces=4,ts_fuse=1,ts_once=1,ts_duplex=1`; bit-identical logprobs. The corresponding managed patches and fuller rad0031 timelines are in the separate `perf/vtp36-tp3-performance` development branch at this checkpoint; stress and 5,5,6 split validation were not yet final.

## XTX physical-swap result — GPU host bandwidth, GB/s

| Transfer | Before swap | After swap |
|---|---:|---:|
| Solo XTX host read/write | 13.3–13.4 / 14.2 | 13.4 / 14.3 |
| Solo R9700 host read/write | 6.7 / 7.1 | 6.7 / 7.1 |
| Primary PEG-half GPU reads; other XTX reads/writes | 6.7 / 6.7 | 6.7 / 6.7 |
| Primary PEG-half GPU reads; R9700 reads/writes | 3.4 / 3.5 | 3.4 / 3.4 |
| Secondary XTX reads; primary XTX reads/writes | 13.4 / 14.2 | 13.4 / 14.2–14.3 |
| Secondary XTX reads; R9700 reads/writes | 6.7 / 7.1 | 6.7 / 7.1 |

The primary reader retained ~13.4 GB/s even if it joined after another GPU. The temporary single-XTX configuration reproduced a fall in R9700 read rate from ~6.7 to 3.4 GB/s with the other XTX removed. The halving therefore follows the **first PEG root port**, not physical card/VBIOS. Matching PCIe config and IOMMU passthrough support CPU-internal arbitration, but the exact coherent-snoop stage is not established.

Rad0031 int8 TP3 two-shot timeline (~10 MiB prefill collective): XTX reduce 889 µs at the first PEG half versus 1301 µs at the second, ~400 µs second-barrier skew per ~4 ms collective. This is an observed GPU-collective effect, not a host-reduced result.

## Two completed VTP36 GPU candidate ABBA comparisons (mean ± SD)

| Metric | int8 baseline → candidate | Change | MXFP4 baseline → candidate | Change |
|---|---|---:|---|---:|
| 1 stream tok/s | 88.4±0.5 → 89.2±0.1 | +1.0% | 116.8±0.3 → 122.4±0.3 | +4.8% |
| 4 streams tok/s | 203.4±5.0 → 218.9±5.7 | +7.6% | 222.4±4.1 → 252.7±6.4 | +13.6% |
| 6 streams tok/s | 194.7±3.0 → 216.0±3.2 | +10.9% | 216.6±2.8 → 245.9±1.5 | +13.5% |
| 8 streams tok/s | 233.2±1.4 → 265.6±1.8 | +13.9% | 237.8±1.7 → 277.4±2.4 | +16.7% |
| TTFT 512 ms | 282.7±4.8 → 246.0±16.0 | -13.0% | 285.7±3.4 → 225.0±5.7 | -21.2% |
| TTFT 2048 ms | 1054.3±3.2 → 797.5±4.8 | -24.4% | 1064.4±4.3 → 774.7±3.5 | -27.2% |
| TTFT 6144 ms | 2963.4±18.4 → 2322.7±17.8 | -21.6% | 2959.1±13.7 → 2216.5±4.9 | -25.1% |
| Four 4096 prompts slowest TTFT ms | 8609.7±37.8 → 7403.9±7.3 | -14.0% | 8799.6±15.7 → 6780.8±48.5 | -22.9% |
| 6144 prompt + 4 streams TTFT ms | 3415.4±1.0 → 2685.3±6.9 | -21.4% | 3433.3±6.8 → 2572.0±7.6 | -25.1% |

The above are **GPU fused-piece/write-once/duplex gains**, not CPU BAR-push or VTP38 host-reduce gains; stress and split checks were still outstanding.

## Earlier laboratory proof and current boundaries

- RAD40 2026-11-01: XTX GPU host reads ~13.3, R9700 ~6.7 GB/s; CPU BAR push XTX ~10.4, R9700 ~5.6 GB/s. SDMA is affected by the primary reader; CPU BAR push unaffected. Coherent/non-coherent/write-combined reads were equal **when idle**; no contended non-coherent result exists. XTX activation heads' makespan 853→683 µs with CPU push to one; chunked 10 MiB handoff 1.36x (128 rows), 1.27x (256). See completed RAD40 Notes/History.
- RAD50 2026-11-09: XTX DMA-BUF cross-process mapping and CPU-write/GPU-word verification passed; real FP16-combine 768-row serial push ~1.13x faster. Earlier 768-row live timeout was the gfx11 `clock64()` watchdog, corrected using `wall_clock64()` scaled by the HIP wall-clock rate; both XTXs passed 100 pre-armed live generations. See `docs/evidence/2026-11-10-rad50-cpu-bar-push/PROGRESS.md`, `SLICE_OPTIMIZATION.md` and `GATE4_LIVE_RESULTS.md`.
- RAD50 2026-11-10: Original R9700 HSA-pool serial canary inconclusive, **separate DMA-BUF API succeeded** for 4 KiB/4 MiB full-word checks, 30 live generations at 1/256/768 rows, and 100-iteration 768-row direct combine (GPU pull 959.0 µs vs CPU push 770.4 µs, 1.25x). One-row push 76.7 µs was slower than pull 74.2 µs. Same RAD50 PROGRESS journal.

## New bounded hardware check (2026-11-13)

A fresh R9700/GFX1201 pre-armed CPU DMA-BUF → GPU-full-word diagnostic ran under an exclusive GPU lease: **768 rows, 3,932,160 bytes, 30/30 verified, zero mismatches, successful cleanup**; p50 synthetic total 840.96 µs. The lease was released. This is **partial Gate C evidence**, not a 10 MiB/all-rank/production pass. Test method, executable digest, output summary and limits: [R9700-LIVE-2026-11-13.md](R9700-LIVE-2026-11-13.md).

## Still untested

HL02 Gate A contended non-coherent mailbox; Gate B simultaneous push to all three GPU VRAMs with GPU partial producers at 320 KiB–10 MiB, plus push-to-one; Gate C full-size GPU-live R9700/XTX fault and lifetime proof. VTP38's CPU ordered sum, exact BF16 comparison, all-rank abort/drain and end-to-end ABBA against the improved VTP36 GPU candidate are **not established by historical evidence**.
