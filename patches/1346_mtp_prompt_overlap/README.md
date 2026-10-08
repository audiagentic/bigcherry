# 1346_mtp_prompt_overlap

QFP31 timing diagnostic. The historical package id is retained; the unqualified prompt-window optimization has been
removed. The package now changes no model work or ordering and emits output only when its timing flag is enabled.

## Flag

- `BIGCHERRY_MTP_PROMPT_TIMING=1`: default off. Splits prompt MTP time into explicit target synchronization,
  post-sync target fetch/copy, draft catch-up, and inter-submit host gap.

Output:

`BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=<...> target_sync_ms=<...> target_fetch_ms=<...> draft_process_ms=<...> draft_decode_ms=<...> host_gap_ms=<...> chunks=<...> tokens=<...>`

At resolved prompt start the diagnostic resets single-sequence attribution. Around each target `llama_process()` it
records the return-to-next-submit host gap. Inside native MTP `process()` it separates the blocking target join from
post-sync NextN fetch/copy and draft decode. Mixed-sequence batches are not attributed.

There is no `BIGCHERRY_MTP_PROMPT_WINDOW`, bounded replay, draft suppression, cache/cancel lifecycle state, or WINDOW
activation marker in this package.

## Promotion record

Promotion record (QFP18 lightweight tier, pin b11474 / b9acf138, 2026-10-08). Diagnostic promoted as a neutral enabler
so the production binary can be timed without a special build.

- Build and run: experiment build `b-metamem-mig-diag` compiled clean at b11474.
- Activation with `BIGCHERRY_MTP_PROMPT_TIMING=1` (38,673-token prompt, 77 chunks):
  `target_nextn_ms=25116.2 target_sync_ms=24979.7 target_fetch_ms=136.5 draft_process_ms=27319.0 draft_decode_ms=2201.9 host_gap_ms=27528.4`.
- Neutrality: greedy text with timing on equals the flag-off arm, md5 `fe307bdfb7e1`.
- The diagnostic identified the QFP31 critical path: the target join moves the wait immediately after each target
  chunk, then draft catch-up and host preparation serialize before the next target submit.

The removed `BIGCHERRY_MTP_PROMPT_WINDOW` mechanism was not qualified: production prompt cache/checkpoints prevented
it from arming, and its ABBA showed no activation marker. It is not part of the promoted package.

## Native llama.cpp comparison

Native llama.cpp has no counterpart. With `BIGCHERRY_MTP_PROMPT_TIMING` unset, this patch only adds dormant timing
branches/hooks and does not alter target or draft computation.
