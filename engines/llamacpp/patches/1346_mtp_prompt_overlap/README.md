# 1346_mtp_prompt_overlap

QFP31 implementation package. The package name is retained although WINDOW, not worker overlap, qualified as the first mechanism.

## Flags

- `BIGCHERRY_MTP_PROMPT_WINDOW=<tokens>`: default `0`. A positive value enables bounded final-window replay only for qualified fresh single-head MTP prompts.
- `BIGCHERRY_MTP_PROMPT_TIMING=1`: default off. Splits prompt MTP time into explicit target synchronization, post-sync target fetch/copy, draft catch-up, and inter-submit host gap.

Timing output:

`BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=<...> target_sync_ms=<...> target_fetch_ms=<...> draft_process_ms=<...> draft_decode_ms=<...> host_gap_ms=<...> chunks=<...> tokens=<...>`

Successful WINDOW replay output:

`BIGCHERRY_PATCH_HIT patch=1346_mtp_prompt_overlap mechanism=window window=<N> replay=<N> skipped=<P-N> end_pos=<P-1> host_mib=<...>`

## WINDOW contract

For prompt length `P > N`, the draft context is rebuilt at prompt end from exactly positions `[P-N, P)`. Replay row `p` uses token `x[p]` and the same previous target hidden row `h[p-1]` used by the native MTP catch-up. If `P-N` begins a new target chunk, the preceding chunk is fetched once to capture `h[P-N-1]`; that predecessor is not draft-decoded.

After replay, draft KV contains only the final prompt window at original absolute positions, `pending_h` is target hidden row `P-1`, and `draft()` starts generation at `P` from that carry. Replay is synchronized before the bounded host collector is freed.

WINDOW does not arm for cached prefixes, MTMD/shared-prefix prompts, shared MTP memory, chained/multiple MTP heads, prompt checkpoints, or `P <= N`. Cache/load/restore paths lacking the host carry suppress MTP proposals rather than using inconsistent draft state.

OVERLAP remains deferred pending WINDOW hardware measurements.
