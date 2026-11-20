---
id: RAD42
order: 0
plan: patching-radiance-rdna3
state: superseded
created-at: '2026-11-01T11:26:21.039480+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P2
---

# Delivered chunked MoE result combine; BAR-push transferred to RAD51

## Description

Historical delivery record. Design A, 256-row chunked CPU combine/consumption, shipped in PR #497 with a reported +6.1% eight-session result in its original cohort. The no-route-first-token poller deadlock was fixed in PR #520 with regression coverage. Design B, CPU BAR/DMA-BUF writes into engine VRAM, was not shipped and belongs only to RAD51, gated by HL02 contention and RAD50 safety plus a live critical-path case. The prior ~0.1–0.2 ms exposed combine remains a negative gate; do not claim the synthetic BAR result reverses it. RAD42 does not own TP3, RAD41 payload format, or HL03 activations.

## Steps

1. Preserve the PR #497 delivery, cohort-limited measurement and PR #520 regression evidence as history; do not reimplement the shipped chunk path.
2. Transfer all unshipped CPU BAR-push serving acceptance to RAD51. HL02 owns shared-link measurements; RAD50 owns mapping/publication safety; VTP38 owns TP3.
3. Verify no unique chunk-combine acceptance remains in current source/ABI or linked reviews. If a future regression is found, file a narrow new item with current evidence rather than reopening the completed design under this broad item.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Historical source/PR references and original run evidence remain linked. Confirm PR #520 regression tests exist in current tree and RAD51 explicitly owns remaining serving, safety, quality and critical-path acceptance. No GPU retest is required for this disposition.

## Effort & Risk



## Standards



## Acceptance Criteria

Chunk combine delivery and its no-route-first-token repair are preserved with their original evidence and limits. All unshipped BAR-push acceptance is explicitly owned by RAD51 and HL02/RAD50. No unique RAD42 work remains; state may be superseded, not completed as if BAR-push shipped.

## Notes

**2026-11-13 reconciliation:** Stop referencing RAD40 as an *active* harness dependency: it is superseded historical evidence. HL02 owns new host-link/push tests; RAD50 owns the cross-process safety proof; HL04 remains closed; RAD51 remains pending and blocked on a separately justified live MoE critical path. A RAD50 synthetic 1.25x R9700 serial result does not overturn HL04's measured ~0.1–0.2 ms exposed path.

**History — earlier implementation/debug notes:**

No P2P: only CPU stores into a BAR and host-memory chunks. Depends on RAD40.

DEFECT 2026-11-02 (open): the chunked combine hangs on the two-helper layout under big concurrent prefill. flashnext-2xtx-combine (no RX 6900 XT tail), W4H2 + 32-row path, 9 GiB host pool + 12 GiB KV tier, 8-session ~1M-token run: the engine watchdog reported 'a synchronisation has seen no queue advance for 5 s' with one queue at 'barrier-and dep[0]=1' and another stuck in mailbox_add_rows_kernel (written 384066 / read 382054), then 'the engine stopped on an error and is not serving (device error)', failing all eight requests. The same layout served 6 x short prompts and a 49K prefill cleanly (run tail-budget), and the four-helper flashnext-4gpu layout with the same combine runs the 8-session load at 3,043-3,045 tok/s. The engine log also shows 'runtime slot map changed across ranks' for layers 34/44/46/47 at start. Logs preserved: the notail-hang scratch folder (engine.log, excerpt.txt, helper0/1 logs); run run tail-budget2 (notail arm). Next: reproduce in the worker harness with two helpers at 2,048-row chunks (chunk words for a helper with no rows in a chunk, the leader-poll mirror, and helper 0 combining with one other helper are the suspects); no layout without the tail can become a default until fixed.



Hang fixed 2026-11-02 (PR #520). Cause: in a combined step, only a chunk's first block polled chunks.done[c] and mirrored it to device memory, and it did so only when its own token had a helper route. A chunk whose first token was entirely engine-local left the chunk's routed blocks spinning on a mirror nobody wrote. Fix: combine_chunk_wait() is shared by the kernel and r3_moe_shm_test; the first block always polls in a combined step. The test fails on the old guard. Hardware: flashnext-2xtx-combine (no tail), 8 sessions, 1,032,070 prompt tokens, completed in 422 s (2,452 tok/s prefill-dominated). Helper 0 ran 22,704 combine steps over 23,804 large-lane requests, about 4x past the 5,957 requests where it hung before. Run: run rad42-fix.



BACKLOG TRIAGE 2026-11-04 (revalidated against main 6eb31a708): the proposed disposition "MERGE BAR/write path into RAD41, close duplicate" was NOT applied; this item stays open. Why: PR #497 implemented chunked combine, PR #520 fixed the chunk-poller hang, and moe_remote_shm.h:289-312 defines the 256-row generation scheme. RAD42.md:85-89 explicitly says design b, CPU BAR-push into engine VRAM, is not done and should be revisited after a link change; RAD41.md:14-31 does not include this write path or its publication/quality gate. Keep the still-unmeasured BAR-push design separately, while marking chunked design A delivered and its hang fixed.



2026-11-20 disposition: PR #497 chunk-combine and PR #520 deadlock repair are delivered; preserve their cohort and protocol history above. Unshipped BAR-push work is transferred to RAD51, with HL02/RAD50 gates. No TP3, RAD41 payload or activation scope is transferred. Superseded only after review RV4239 closes.

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
- 2026-11-04T04:24:27.433273+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4239
- 2026-11-20T02:28:17.055543+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:validation, section:acceptance_criteria
- 2026-11-20T02:28:29.510342+00:00 (updated-by): Updated: section:notes
- 2026-11-20T02:28:59.399164+00:00 (state-transition): State: pending → superseded
