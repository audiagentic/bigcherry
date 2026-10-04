# 1317_spec_round_timing

**Status:** untested
**Plan item:** FMTP01

## What it does

Diagnostic only. `BIGCHERRY_SPEC_TIMING=1` logs one `BIGCHERRY_SPEC_TIMING` line per speculative round with the
serial phase times: fresh draft, target submit (llama_process host time), target sync wait, draft-context catch-up
(common_speculative_process), target sample-and-accept, plus drafted/accepted counts. No synchronization is added or
moved. Feeds the FMTP01 Gate 0 decision between options F (reseed/sync elimination), C (device greedy) and B (early
hidden catch-up).
