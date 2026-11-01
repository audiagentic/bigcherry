---
id: RAD40
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-01T11:26:10.176424+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: S
priority: P1
---

# Host-link transfer options harness: GPU pull, SDMA copy, CPU BAR push, write-combined memory, chunked overlap

## Description

Quick, model-free answers on which transfer mechanism moves remote-MoE activations and result rows fastest on this host, alone and under the measured read contention (XTX 0 reading halves every other CPU-attached card's host reads; writes and CPU stores into a card's resizable BAR do not contend). Measured so far (tools/lab/radiance/host-link-bw.hip, 2026-11-01): kernel read/write of mapped host memory XTX 13.4/14.2 GB/s (Gen4 x8), R9700 6.7/7.1 (Gen4 x4), 6900 XT 3.6/3.6 (Gen3 x4 chipset); hipMemcpy H2D contends exactly like kernel reads; CPU push into R9700 VRAM through its BAR 5.6 GB/s and unaffected by XTX 0 reading (GPU pull drops to 3.4).

## Steps

1. One harness binary with modes: kernel pull (coherent / non-coherent / write-combined host memory), hipMemcpyAsync H2D/D2H on 1-2 streams, CPU BAR push (1-8 threads, NT stores), GPU write to host.
2. Synchronised multi-card runs (common wall-clock start) for every pair and all four, both directions.
3. Chunked overlap: producer writes 256-row chunks with a flag; consumer kernel starts on chunk k while k+1 arrives; measure end-to-end time for a 2048-row plane against whole-plane transfer.
4. Table of GB/s and latency per (card, mechanism, contention) into the Flash-Next model page.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Every mode reproduces the 2026-11-01 numbers above within 5%; results for all four cards and all pairs recorded; the chunked mode reports both total time and time-to-first-chunk.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

No P2P on this host (memory no-p2p-on-dev-host). Feeds RAD41-RAD43.

## Change Log

- 2026-11-01T11:26:10.176424+00:00 (created-by): Created by agent

## Harness results (2026-11-01, `/work/development/tmp/bigcherry/rad40/linkx.hip`, `runs/rad40-harness`)

Alone, 21 MiB a pass (GB/s): kernel pull from coherent, non-coherent and write-combined host memory are identical (XTX
13.3, R9700 6.7, 6900 XT 3.6); SDMA H2D on one or two streams the same; SDMA D2H and GPU writes 14.2 / 6.9-7.1 / 3.6; CPU
BAR push 10.4-10.5 (XTX, 4 threads), 5.6 (R9700), 2.8 (6900 XT), one thread 7.1-8.1 / 4.9 / 2.7.

While XTX 0 pulls at full rate: every GPU-initiated transfer of another CPU-attached card halves -- kernel pulls, SDMA H2D
on any number of streams, and GPU writes to host too (XTX 1 6.8, R9700 3.4-3.5); only CPU BAR pushes keep their rate
(XTX 1 10.3, R9700 5.6). XTX 0 wins whichever card starts first (XTX 1 streaming, XTX 0 joining a second later: XTX 0 13.4).
XTX 1 with the R9700 or the 6900 XT: both full.

Activation round (5.4 MB to each helper at once, p50 finish over 20 rounds): XTX 0 662 us, XTX 1 853, tail 1,601 (its own
Gen3 x4). Staggering XTX 0 by 100-800 us only moves the wait (head makespan 852-1,522 us). A CPU push of XTX 0's copy
instead: XTX 0 683, XTX 1 655 (its uncontended time 651): the heads' makespan 853 -> 683 us. Chunked hand-off of a 10 MiB
result plane, producer XTX 0 or 1, consumer R9700: 128 rows 1.36x, 256 rows 1.27x, 512 rows 1.17x, 1024 rows 1.04x faster
than write-then-read. A 6900 XT producer hung the device-flag variant (consumer never saw the flags): the host-mediated
protocol first, as the review says.

## Review (GPT, 2026-11-01) -- adopted
- One trace contract for every item: per layer-step publish, activation-ready per helper, compute start/end, result-write
  done, combine start/end, engine add start/end, bytes per endpoint; host monotonic clock across processes, HIP events
  for GPU durations. Go gate: >2% on the 42K probe or 8 sessions, no p99 regression >5%.
- Stop/go for a transfer change: >=15% off the transfer-path makespan or >=0.3 ms off a layer-step critical path.
- Contention attribution with root: `lspci -vv` (MPS, MRRS, relaxed ordering, extended tags) for the cards and their root
  ports; swapping XTX 0 and XTX 1 physically separates slot from card.
- Next: a complete per-layer critical-path trace of the engine and the three helpers before kernel or protocol work.
