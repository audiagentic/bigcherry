# 1305_ffn_tensor_split

## Summary

Under `-sm tensor`, `BIGCHERRY_FFN_TS="a,b,c"` gives the FFN/MoE family (routed-expert, shared-expert and dense FFN up/gate/down weights and biases) its own split vector, independent of `-ts` and of 1303's `BIGCHERRY_ATTN_TS`. Groups keep their upstream anchors and per-layer rotation.

## Outcome

Recorded state: `evaluated` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
