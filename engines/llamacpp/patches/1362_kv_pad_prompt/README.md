# 1362_kv_pad_prompt

QFP53 experimental patch; state is `untested` and it is not in the production patch set.

Enable through the experiment composition and choose a padding quantum:

```text
--experiment kv-pad-prompt
BIGCHERRY_KV_PAD=4096
```

Valid nonzero values are multiples of 256 no larger than the active KV cache. Unset/0 is upstream behavior.
`BIGCHERRY_PATCH_TRACE=1` emits the one-time activation marker.

## Qualification

Mechanics:
- `python -m bigcherry patch-lint`
- `python tools/tests/patch/test_1362_kv_pad_prompt.py`
- `python -m bigcherry patch-rebase-check --source bigcherry`
- `python -m bigcherry patch-rebase-check --experiment kv-pad-prompt`

Hardware: Flash-Next production profile at 8K/24K/98K, `BIGCHERRY_KV_PAD` in {2048,4096,8192}, recording
`BIGCHERRY_SUBMIT_TIMING` graph reuse/rebuild counts, balanced prefill A/B and greedy identity; then Qwen3.8-27B
dual-XTX. Decode and training are not claimed qualified by this package until measured.
