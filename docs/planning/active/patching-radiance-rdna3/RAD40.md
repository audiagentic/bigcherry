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
