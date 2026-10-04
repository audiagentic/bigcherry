# 1315_mtp_draft_trace

**Status:** untested
**Plan item:** QFP15

## What it does

Diagnostic only. With `BIGCHERRY_DRAFT_TRACE=1`, the single-head MTP draft path logs one
`BIGCHERRY_DRAFT_TRACE step` line per draft step (top token, probability as exact hex float, FNV-1a hash of the draft
hidden row) and one `BIGCHERRY_DRAFT_TRACE accept` line per round (accepted count, FNV-1a hash of the target hidden row
seeding the next round). Diffing two runs of the same build/prompt shows whether the target or the draft diverges
first. No behaviour change.
