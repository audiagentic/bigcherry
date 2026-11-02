# 1278_qwen4exp_sm_tensor

## Summary

Removes `LLM_ARCH_QWEN4EXP` from `llm_arch_supports_sm_tensor()`'s rejection list so Qwen3.8-Flash-Next loads with `-sm tensor`. The pinned llama.cpp already has the qwen4exp tensor-split segment rules; upstream only gated it pending `test-llama-archs`.

## Outcome

Recorded state: `superseded` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
