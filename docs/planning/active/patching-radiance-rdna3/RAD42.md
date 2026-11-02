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

DEFECT 2026-11-02 (open): the chunked combine hangs on the two-helper layout under big concurrent prefill. flashnext-2xtx-combine (no RX 6900 XT tail), W4H2 + 32-row path, 9 GiB host pool + 12 GiB KV tier, 8-session ~1M-token run: the engine watchdog reported 'a synchronisation has seen no queue advance for 5 s' with one queue at 'barrier-and dep[0]=1' and another stuck in mailbox_add_rows_kernel (written 384066 / read 382054), then 'the engine stopped on an error and is not serving (device error)', failing all eight requests. The same layout served 6 x short prompts and a 49K prefill cleanly (run tail-budget), and the four-helper flashnext-4gpu layout with the same combine runs the 8-session load at 3,043-3,045 tok/s. The engine log also shows 'runtime slot map changed across ranks' for layers 34/44/46/47 at start. Logs preserved: /work/development/tmp/bigcherry/notail-hang/ (engine.log, excerpt.txt, helper0/1 logs); run /work/bigcherry/runs/tail-budget2 (notail arm). Next: reproduce in the worker harness with two helpers at 2,048-row chunks (chunk words for a helper with no rows in a chunk, the leader-poll mirror, and helper 0 combining with one other helper are the suspects); no layout without the tail can become a default until fixed.



Hang fixed 2026-11-02 (PR #520). Cause: in a combined step, only a chunk's first block polled chunks.done[c] and mirrored it to device memory, and it did so only when its own token had a helper route. A chunk whose first token was entirely engine-local left the chunk's routed blocks spinning on a mirror nobody wrote. Fix: combine_chunk_wait() is shared by the kernel and r3_moe_shm_test; the first block always polls in a combined step. The test fails on the old guard. Hardware: flashnext-2xtx-combine (no tail), 8 sessions, 1,032,070 prompt tokens, completed in 422 s (2,452 tok/s prefill-dominated). Helper 0 ran 22,704 combine steps over 23,804 large-lane requests, about 4x past the 5,957 requests where it hung before. Run: /work/bigcherry/runs/rad42-fix.

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
- 2026-11-02T02:41:48.629382+00:00 (updated-by): Updated: section:notes
- chg_20261102_024204_planning-records-updated-with_6210
- 2026-11-02T02:42:29.034602+00:00 (updated-by): Updated: section:ledger-events
- chg_20261102_042612_fixed-a-hang-with-many-concurr_2245
- 2026-11-02T04:26:16.739750+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T04:46:35.372778+00:00 (updated-by): Updated: section:notes
