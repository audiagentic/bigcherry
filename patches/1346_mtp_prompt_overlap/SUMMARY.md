# 1346_mtp_prompt_overlap

**Status:** untested
**Plan item:** QFP31

## What it does

The package keeps the original `1346_mtp_prompt_overlap` id, but qualification selected WINDOW before a worker-overlap mechanism.

`BIGCHERRY_MTP_PROMPT_WINDOW=N` is default-off (`0`). For a qualified fresh, text-only, non-shared, single-head MTP prompt with `P > N`, early prompt chunks skip the target NextN getter and draft-context catch-up. The driver retains exactly the final `N` token/position/previous-hidden rows, clears the draft sequence at prompt end, and replays those rows at their original absolute positions before generation. The target context and target sampling path are unchanged.

The first replay token needs target hidden row `h[P-N-1]`. Therefore the chunk containing that predecessor is fetched even when it is outside the replay window; it is not draft-decoded. All earlier qualified chunks avoid the MTP-induced target synchronization.

## Safety / lifecycle

WINDOW only arms when `n_cached == 0`, there is no MTMD/shared-prefix input, prompt checkpoints are disabled, MTP memory is not shared, and there is exactly one MTP head. `P <= N` stays on the native path.

Prompt-cache reuse/load, partial-prompt cancellation, and slot restore cannot reconstruct the host-only MTP carry; those paths fail closed by suppressing MTP proposals until a later fresh prompt. Full prompt clear resets the host carry. Existing context shift is retained because `common_memory` shifts target and draft memories together.

A successful replay emits one:

`BIGCHERRY_PATCH_HIT patch=1346_mtp_prompt_overlap mechanism=window window=<N> replay=<N> skipped=<P-N> end_pos=<P-1> host_mib=<bounded collector MiB>`

## Diagnostic

`BIGCHERRY_MTP_PROMPT_TIMING=1` remains default-off and reports:

`BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=... target_sync_ms=... target_fetch_ms=... draft_process_ms=... draft_decode_ms=... host_gap_ms=... chunks=... tokens=...`

`target_sync_ms` is the explicit target join; `target_fetch_ms` is post-sync NextN buffer/getter/copy time; `host_gap_ms` is accumulated target-process return to next target-process call time. Mixed-sequence batches are not attributed.

## Qualification status

Offline mechanics cover exact/idempotent application against the pinned tree when the vendor checkout is present, composition after 1255/1268, explicit edit contracts, WINDOW qualification, chunk-boundary row census, lifecycle fail-closed guards, default-off behavior, and package tags.

OVERLAP is intentionally not implemented in this package yet. Re-measure WINDOW first; add overlap only if the remaining in-window draft catch-up is material.
