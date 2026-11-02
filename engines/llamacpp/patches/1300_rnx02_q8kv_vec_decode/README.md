# 1300_rnx02_q8kv_vec_decode

## Summary

Adds a default-off selector experiment in `ggml_cuda_get_best_fattn_kernel()`. With `BIGCHERRY_RNX02_Q8_VEC=1`, the selector chooses the existing native Q8_0 vector FlashAttention implementation only for D=256, single-token Q, Q8_0 K/V, an available D=256 vector case, and RDNA3/RDNA4.

## Outcome

Recorded state: `untested` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
