# 1278_qwen4exp_sm_tensor

**Status:** superseded
**Plan item:** QFN01

Removes `LLM_ARCH_QWEN4EXP` from `llm_arch_supports_sm_tensor()`'s rejection list so
Qwen3.8-Flash-Next loads with `-sm tensor`. The pinned llama.cpp already has the qwen4exp
tensor-split segment rules; upstream only gated it pending `test-llama-archs`. Correctness must be
shown end to end against `-sm layer` before any use.

**Superseded 2026-10-02:** upstream llama.cpp at c061df198 (PR #29761 era) no longer lists
LLM_ARCH_QWEN4EXP in llm_arch_supports_sm_tensor(), so `-sm tensor` loads natively.
