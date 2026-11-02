# 1201_rd20_attn_gate_tp_split

## Summary

Gave attn_gate.weight the same head-aligned split granularity as attn_q in the regular-attention branch of get_split_granularity, matching what the recursive-attention (pattern_qkv) path already did.

## Outcome

Recorded state: `superseded` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
