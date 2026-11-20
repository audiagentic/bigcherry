# ENG05: v26 mailbox hand-off, one traced launch (flashnext-4gpu, 1 stream decode)

2026-11-20 (AEDT), under the gpu lease. Build: main + the ENG05 lab stamps (`ENG05_MAILBOX_TRACE.md`), complete engine,
helper and serving set built from this branch. `experiment.diagnostics.lanes.trace_level=1` (every request stamped),
placement policy configured, decode at 1 stream (900-token prompt, 256 tokens, 2 reps, seed 7).

## Numbers (9,809 decode requests, 45 layers, all stamps valid; engine clock only)

| field | mean us | p50 | p95 |
|---|---|---|---|
| publish (publish-kernel entry to just before the release store, includes the system fence) | 21.9 | 21.6 | 22.5 |
| wait bookkeeping (engine sees helper done, to the end of the wait bookkeeping) | 39.7 | 34.7 | 91.0 |
| finish kernel | 0.56 | 0.56 | 0.60 |
| combine engine wait (this layout has no combine on) | 0 | 0 | 0 |
| helper: local | 54.8 | 54.0 | 57.0 |
| helper: act / gemm / write | 15.9 / 85.5 / 10.3 | | |
| helper GPU total | 111.4 | 109.8 | 147.8 |
| engine wait on the helper (wait_us) | 67.5 | 65.5 | 104.4 |

Summed over the 45 layers of one step (mean of each layer's mean): publish 0.99 ms, wait bookkeeping 1.73 ms, finish
0.03 ms (protocol work, about 2.75 ms or 10.6% of the 26 ms step); helper GPU 5.10 ms and engine wait 3.10 ms (the
helper's own work and the overlap with it). `per-layer-stats.json` has every layer; `mailbox-trace.csv.gz` the raw rows.

No single stage dominates. The only heavy tail is the wait bookkeeping (p50 35 us, p95 91 us), which is the engine
noticing the helper's completion word, so it follows the helper's variance more than any v26 protocol work.

## Cost of the stamps

Trace off (this same build, no experiment): 25.78 ms/step, 1 stream, against 25.74-25.87 on main in the combine sweep
the same morning (`docs/evidence/2026-11-20-eng05-combine-threads`). Zero cost off, within the noise. Trace on:
26.26 ms/step (+0.48 ms), so the traced run is for diagnosis, never for headline numbers. Draft acceptance 0.516 in both.

## What this does and does not say

The v26 hand-off protocol work (publish + wait bookkeeping + finish) is about 2.75 ms a step. That is an absolute cost; it
does not by itself say how much of it is new since v1.17 (the ~0.9 ms step-time loss), because no pre-v26 build was
stamped the same way. The remaining question for ENG05 step 1 is the pre-v26 comparison (a build at the commit before
#1014, or a single-revert build); the engine-clock stamps here are the baseline for it.
