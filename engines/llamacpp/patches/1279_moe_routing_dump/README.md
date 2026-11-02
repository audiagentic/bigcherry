# 1279_moe_routing_dump

## Summary

Diagnostic for the eval-callback tool (common/debug.cpp): with `BIGCHERRY_DEBUG_FULL_TENSORS=1` matched tensors are printed in full instead of the first/last three values per dimension, so `llama-eval-callback --tensor-filter 'ffn_moe_topk.*'` yields every routed expert id per token and layer. Default behaviour is unchanged. Never part of a production recipe.

## Outcome

Recorded state: `untested` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
