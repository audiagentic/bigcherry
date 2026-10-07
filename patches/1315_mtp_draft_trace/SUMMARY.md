# 1315_mtp_draft_trace

**Status:** validated
**Plan item:** QFP15

## What it does

Diagnostic only. With `BIGCHERRY_DRAFT_TRACE=1`, the single-head MTP draft path logs one
`BIGCHERRY_DRAFT_TRACE step` line per draft step (top token, probability as exact hex float, FNV-1a hash of the draft
hidden row) and one `BIGCHERRY_DRAFT_TRACE accept` line per round (accepted count, FNV-1a hash of the target hidden row
seeding the next round). Diffing two runs of the same build/prompt shows whether the target or the draft diverges
first. No behaviour change.

## Qualification

QFP18 lightweight promotion on b11474, Brutus 2026-10-08: build `deploy-v4-diag` compiled clean; with
`BIGCHERRY_DRAFT_TRACE=1 BIGCHERRY_SPEC_TIMING=1 BIGCHERRY_DRAFT_TIMING=1` the log contained 92 DRAFT_TRACE,
23 SPEC_TIMING and 23 DRAFT_TIMING lines, 0 error lines, and greedy text identical to production
(md5 `a34a34c1f48d`).
