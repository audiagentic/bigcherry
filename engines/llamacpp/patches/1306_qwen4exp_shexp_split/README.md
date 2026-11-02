# 1306_qwen4exp_shexp_split

## Summary

With `BIGCHERRY_SHEXP_SPLIT=1`, the Qwen4Exp shared expert (`ffn_{up,gate,down}_shexp`) is tensor-split under `-sm tensor` using the DSV4 layout (up/gate axis 1, down axis 0, anchored to `ffn_down_shexp.weight`) instead of falling through to MIRRORED. Upstream mirrors it, so every GPU stores and computes the whole shared expert in every layer.

## Outcome

Recorded state: `evaluated` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
