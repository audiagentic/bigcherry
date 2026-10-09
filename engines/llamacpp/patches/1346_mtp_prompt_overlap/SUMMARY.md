# 1346_mtp_prompt_overlap

**Status:** validated
**Plan item:** QFP31

## What it does

Default-off MTP prompt timing diagnostic. The historical package id is retained, but the unqualified
`BIGCHERRY_MTP_PROMPT_WINDOW` optimization has been removed.

`BIGCHERRY_MTP_PROMPT_TIMING=1` reports:

`BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=... target_sync_ms=... target_fetch_ms=... draft_process_ms=... draft_decode_ms=... host_gap_ms=... chunks=... tokens=...`

`target_sync_ms` is the explicit target join; `target_fetch_ms` is post-sync NextN getter/copy work;
`draft_decode_ms` times the native draft-context decodes; `host_gap_ms` is target-process return to the next target
process call. The diagnostic uses a prompt-start reset and target-process entry/return hooks only for attribution.
Mixed-sequence batches are not attributed.

## Safety / neutrality

No target/draft computation, row selection, cache semantics, or ordering is changed. The explicit
`llama_synchronize(ctx_tgt)` is executed only while timing is active and makes visible the synchronization the
following NextN getter would otherwise perform. With the flag unset the native path is preserved.

The former WINDOW collector/replay, draft suppression, lifecycle invalidation hooks and
`BIGCHERRY_MTP_PROMPT_WINDOW` environment variable are absent.

## Qualification

QFP18 lightweight promotion evidence at b11474: clean hardware build, timing output present, and greedy text identical
to the flag-off arm (md5 `fe307bdfb7e1`). Full record is in README.md and the PR evidence file.
