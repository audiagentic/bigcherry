---
id: RAD12
order: 12
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-19T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Helper small lane that follows the rows: several pre-queued sizes, the smallest that fits

## Description

RAD10-02 (docs/evidence/RAD10/traced-180/, Flash-Next page "Where the decode step goes") showed that the helper small-lane row limit is the wrong
size for concurrency. A stream sends about 4 verify rows a step (MTP depth 3), so 12 and 16 rows at three and four streams fall onto the large lane
(small-lane share 25% and 7%) and cost about 100 us more exposed helper wait a layer-step. Raising R3_MOE_HELPER_SMALL_ROWS from 8 to 24 keeps them on the
small lane: +21% (3 streams), +12% (4), +6% (6) decode throughput, but -10% at one stream because every pre-queued small step is padded to the limit.
The lane should follow the rows: pre-queue small lanes at several sizes (for example 4, 8, 16, 24) and let the engine pick the smallest that fits the step.

## Steps

1. Read the lane layout (moe_remote_shm.h: Header::small_rows, two lanes a producer rank) and the engine-side lane choice in the rad0002 hand-off; design the smallest change that
   gives each rank N small lanes with their own pre-queued step ring, one large lane unchanged. Bump kVersion; the static_assert on the Request size stays.
2. Harness first: extend the replay harness to replay captures of 4, 8, 12, 16 and 24 rows on each lane size and compare bitwise (padding rows must not change the output) and time per request.
3. Implement behind a default-off setting (R3_MOE_HELPER_SMALL_LANES="4,8,16,24"); the single-size behaviour stays reachable as one entry.
4. Evidence: the helper-smallrows-ab.json pair with the new arm against 8 and 24 at 1, 2, 3, 4 and 6 streams, TRACE on, clean cells only; the lightweight tier for promotion.

## Validation

Bitwise-identical replay across lane sizes; at 1 stream no slower than the 8-row arm, at 3 and 4 streams within noise of the 24-row arm; per-stream
quality gates (KL, top-1) unchanged; no hang across repeated start/stop.

## Notes

Depends on RAD10-02 (done). The other half of the finding, that the helper wait per layer-step still grows 3 to 4 times with rows even on the small lane (152 -> 480 us), is the
kernel and dispatch question tracked in RAD10 (shadow aggregation, GEMV/grouped kernel for 12 to 24 rows). The helpers' median imbalance of 10-27% is a separate lever
(balanced ownership was neutral at one stream; retest under concurrency).
