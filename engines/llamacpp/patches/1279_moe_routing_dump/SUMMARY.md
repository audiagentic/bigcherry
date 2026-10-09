# 1279_moe_routing_dump

**Status:** untested
**Plan item:** QFN02

Diagnostic for the eval-callback tool (common/debug.cpp): with
`BIGCHERRY_DEBUG_FULL_TENSORS=1` matched tensors are printed in full instead of the first/last three
values per dimension, so `llama-eval-callback --tensor-filter 'ffn_moe_topk.*'` yields every routed
expert id per token and layer. Default behaviour is unchanged. Never part of a production recipe.
