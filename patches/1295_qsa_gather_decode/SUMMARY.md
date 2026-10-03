# 1295_qsa_gather_decode

**Status:** untested
**Plan item:** RNX02

## What it does

For batches of at most 8 tokens, Qwen4Exp QSA attention gathers each token's selected KV cells (padded to a
multiple of 256 with a masked sentinel) and runs flash attention over them, instead of masking the full
cache. Prefill keeps the masked path. `BIGCHERRY_QSA_GATHER=0` disables it.

## Why

On HIP the QSA mask does not reduce work: decode attention per token grew 7.6x from 10K to 80K context
(target 0.44 -> 3.36 ms/token, draft 0.18 -> 1.41 ms/token summed over GPUs). Upstream's sparse attention is
NVIDIA-only. See RNX02 review RV4213. Compared against 1296 (HIP port of the upstream sparse kernel path).

Validation: decode ms/step at 10K/80K/160K, greedy parity without MTP at 10K and 80K against the masked path.
