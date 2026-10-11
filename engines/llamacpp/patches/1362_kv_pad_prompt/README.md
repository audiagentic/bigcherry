# 1362_kv_pad_prompt

QFP53 experimental patch; state is `untested` and it is not in the production patch set.

Enable through the experiment composition and choose a KV-cell padding quantum:

```text
--experiment kv-pad-prompt
BIGCHERRY_KV_PAD=4096
```

Valid nonzero values are multiples of 256 no larger than the active KV cache. Unset/0 is exact upstream behavior.
The same value also stabilizes Qwen4Exp/GLM5-Next k-pool graph sizing with a pool quantum of
`max(64, GGML_PAD(BIGCHERRY_KV_PAD / indexer_kpool, 64))`. Padding-only pools remain masked and use the `n_kv`
sentinel. `BIGCHERRY_PATCH_TRACE=1` emits the one-time activation marker.

## Qualification

Mechanics:
- `python -m bigcherry patch-lint`
- `python tools/tests/patch/test_1362_kv_pad_prompt.py`
- `python -m bigcherry patch-rebase-check --source bigcherry`
- `python -m bigcherry patch-rebase-check --source bigcherry --experiment kv-pad-prompt`

Hardware acceptance: Flash-Next production profile at 8K/24K/98K with full 512-token prompt batches, recording
`BIGCHERRY_SUBMIT_TIMING` and requiring consecutive prompt batches to report `reused=1`, balanced prefill A/B, and
greedy identity. Decode/training and irregular/multi-sequence prompt reuse are not claimed qualified until measured.
