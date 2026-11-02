# 1314_ar_cpu_root_fused

## Summary

With `BIGCHERRY_AR_FUSED=1`, 1291's small-message cpu-root AllReduce enqueues one `bc_cpu_root_fused` kernel per rank instead of `bc_cpu_root_produce` + `bc_cpu_root_consume`: copy the slice to pinned host memory, publish the arrive epoch, spin on the CPU's done epoch, copy the result back, all in one launch.

## Outcome

Recorded state: `evaluated` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
