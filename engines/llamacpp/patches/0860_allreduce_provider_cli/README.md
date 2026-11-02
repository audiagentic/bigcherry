# 0860_allreduce_provider_cli

## Summary

Adds `--allreduce <auto|ccl|host|adaptive|p2p|root3|butterfly>` and `--allreduce-wire <native|q8>` so the multi-GPU AllReduce implementation is chosen by CLI argument instead of the GGML_CUDA_ALLREDUCE / GGML_CUDA_AR_WIRE environment variables.

## Outcome

Recorded state: `validated` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
