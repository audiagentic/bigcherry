# 1293_sched_single_input_sync

## Summary

`ggml_backend_sched` synchronizes an event-less split backend once per split before copying user inputs, instead of before every input.

## Outcome

Recorded state: `superseded` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
