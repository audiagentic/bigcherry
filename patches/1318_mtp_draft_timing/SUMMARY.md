# 1318_mtp_draft_timing

**Status:** untested
**Plan item:** FMTP01

## What it does

Diagnostic only. With `BIGCHERRY_SPEC_TIMING=1`, each single-head MTP draft call logs
`BIGCHERRY_DRAFT_TIMING steps submit_us sync_us rest_us total_us`: draft decode submit host time, the draft GPU's
remaining work (an explicit sync after each submit, which the sampler would wait for anyway), and the rest (sampling,
nextn hidden read, batch rebuild). Splits the 6.4-8.7 ms serial fresh draft into GPU work and host overhead.
