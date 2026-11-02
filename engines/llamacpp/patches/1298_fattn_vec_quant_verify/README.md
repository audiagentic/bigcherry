# 1298_fattn_vec_quant_verify

## Summary

On GPUs without tensor cores (HIP RDNA), quantized-KV flash attention uses the vector kernel for up to `BIGCHERRY_FA_VEC_QMAX` queries (default 4, upstream 2) instead of the tile kernel.

## Outcome

Recorded state: `rejected` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
