---
id: RAD42
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-01T11:26:21.039480+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Overlap the engine's result transfer: chunked rows and the combine pushing into engine VRAM

## Description

Today the engine starts reading result rows only after every helper (or the combine) is done, then pulls ~10.5 MB over its x4 link (~1.6 ms a 2048-row layer with the combine). Two overlaps: (a) helpers and helper 0's combine publish per-chunk done words (e.g. every 256 tokens) and the engine's add kernel consumes chunks as they land; (b) the CPU combine writes its fp16 rows straight into a buffer in the engine's VRAM through the resizable BAR (posted writes, no read contention), shared to helper 0 by IPC, so the engine reads locally. Estimate on the current topology 1-4%; larger after the R9700 moves to an x8 slot.

## Steps

1. Use the RAD40 harness to measure chunk overhead and BAR-push throughput under real concurrency first; stop if the modelled gain is under 1%.
2. Protocol: per-chunk generation words for the combined plane (and per-helper planes), release/acquire ordering across CPU and GPU, chunk size from the harness.
3. Engine allocates the combined plane in fine-grained VRAM and exports it; helper 0 maps it and the combine pool writes with non-temporal stores.
4. 42K probe with TRACE: engine exposed wait and add time before/after; 8-session run.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Engine-side transfer time a prefill layer reduced by the modelled amount; 8-session throughput up; output bit-identical to the non-overlapped path (same rows, same order).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

No P2P: only CPU stores into a BAR and host-memory chunks. Depends on RAD40.

## Change Log

- 2026-11-01T11:26:21.039480+00:00 (created-by): Created by agent

## Harness and review (2026-11-01)

Chunked hand-off measured 1.27x at 256-row chunks on a 10 MiB plane (RAD40). Review: design A first -- the CPU combine
publishes complete combined chunks (all contributing helpers' rows summed), the engine consumes them; a helper without
rows in a chunk is marked complete explicitly; 64-bit generations; no long-lived GPU polling (RDNA idle power, RAD20).
Go: >=0.3 ms off the p50 layer critical path and >=2% end to end. BAR push of the combined plane (design b) is slower
standalone (5.6 against 6.7 GB/s pull) and only pays through overlap; remeasure after the R9700 moves to x8. New lever from
the harness: CPU-pushing XTX 0's activations removes the read contention (heads' activation makespan 853 -> 683 us).

## Design A implemented (2026-11-01, protocol 22)

Chunk words: Header::chunks.seq[helper][chunk] (helper GPU, system-scope release after a chunk's rows) and
chunks.done[chunk] (helper 0's CPU after combining it). The engine skips its up-front wait on a combined step; a chunk's
first add block polls chunks.done and mirrors it into device memory (published[] + kCombineChunks), the other blocks wait on
the mirror; the full wait (combine.done, done words, status) runs after the add. Three faults on the way: the route check
asserted finished helpers; helper 0 read another helper's request before the engine's writes to it were visible on its CPU;
every add block polling host memory flooded the x4 link (15.5 s instead of 14.3). Result (runs/rad42-confirm): 8 sessions
2,513 tok/s against 2,369 (+6.1%), 42K probe 13.1-13.7 s against 14.3-14.5, KLD identical to the unchunked combine.
Design b (pushing the combined plane into engine VRAM through the BAR) is not done: re-assess after the R9700 moves to x8.

## Ledger-events

- chg_20261101_141636_flash-next-on-four-gpus-serves_8287
- 2026-11-01T14:16:40.613683+00:00 (updated-by): Updated: section:ledger-events
