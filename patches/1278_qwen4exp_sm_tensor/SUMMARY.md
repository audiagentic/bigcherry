# 1278_qwen4exp_sm_tensor

**Status:** untested
**Plan item:** QFN01

Removes `LLM_ARCH_QWEN4EXP` from `llm_arch_supports_sm_tensor()`'s rejection list so
Qwen3.8-Flash-Next loads with `-sm tensor`. The pinned llama.cpp already has the qwen4exp
tensor-split segment rules; upstream only gated it pending `test-llama-archs`. Correctness must be
shown end to end against `-sm layer` before any use.
