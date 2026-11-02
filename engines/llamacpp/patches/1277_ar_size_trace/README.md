# 1277_ar_size_trace

## Summary

Lab-only diagnostic for 0840's adaptive AllReduce dispatch. BIGCHERRY_AR_SIZE_TRACE=<n> prints the first n calls' bytes, ne0/ne1, type, **preferred** internal/RCCL route and switch threshold.

## Outcome

Recorded state: `untested` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
