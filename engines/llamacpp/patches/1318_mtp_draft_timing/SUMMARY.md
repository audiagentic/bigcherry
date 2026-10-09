# 1318_mtp_draft_timing

**Status:** validated
**Plan item:** FMTP01

## What it does

Diagnostic only. With `BIGCHERRY_SPEC_TIMING=1`, each single-head MTP draft call logs
`BIGCHERRY_DRAFT_TIMING steps submit_us sync_us rest_us total_us`: draft decode submit host time, the draft GPU's
remaining work (an explicit sync after each submit, which the sampler would wait for anyway), and the rest (sampling,
nextn hidden read, batch rebuild). Splits the 6.4-8.7 ms serial fresh draft into GPU work and host overhead.

## Qualification

QFP18 lightweight promotion on b11474, Brutus 2026-10-08: build `deploy-v4-diag` compiled clean; with
`BIGCHERRY_DRAFT_TRACE=1 BIGCHERRY_SPEC_TIMING=1 BIGCHERRY_DRAFT_TIMING=1` the log contained 92 DRAFT_TRACE,
23 SPEC_TIMING and 23 DRAFT_TIMING lines, 0 error lines, and greedy text identical to production
(md5 `a34a34c1f48d`).
