# 1317_spec_round_timing

**Status:** validated
**Plan item:** FMTP01

## What it does

Diagnostic only. `BIGCHERRY_SPEC_TIMING=1` logs one `BIGCHERRY_SPEC_TIMING` line per speculative round with the
serial phase times: fresh draft, target submit (llama_process host time), target sync wait, draft-context catch-up
(common_speculative_process), target sample-and-accept, plus drafted/accepted counts. No synchronization is added or
moved. Feeds the FMTP01 Gate 0 decision between options F (reseed/sync elimination), C (device greedy) and B (early
hidden catch-up).

## Qualification

QFP18 lightweight promotion on b11474, Brutus 2026-10-08: build `deploy-v4-diag` compiled clean; with
`BIGCHERRY_DRAFT_TRACE=1 BIGCHERRY_SPEC_TIMING=1 BIGCHERRY_DRAFT_TIMING=1` the log contained 92 DRAFT_TRACE,
23 SPEC_TIMING and 23 DRAFT_TIMING lines, 0 error lines, and greedy text identical to production
(md5 `a34a34c1f48d`).
