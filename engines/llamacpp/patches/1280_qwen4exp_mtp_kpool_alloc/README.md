# 1280_qwen4exp_mtp_kpool_alloc

## Summary

Forward-expands the qwen4exp k-pool `k_idxs` and `new_pool_*` inputs so the MTP draft graph (no QSA layer) still allocates them. Without it, `-sm tensor` + MTP aborts at server load with `GGML_ASSERT(buffer)` in `set_input_k_idxs` / `set_input_kpool`.

## Outcome

Recorded state: `rejected` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
