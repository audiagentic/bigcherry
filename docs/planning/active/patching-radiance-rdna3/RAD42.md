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
