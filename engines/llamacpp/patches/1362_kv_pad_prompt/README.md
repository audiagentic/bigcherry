# 1362_kv_pad_prompt — rejected

**State: rejected.** QFP53 is superseded; see `docs/planning/completed/patching-qwen-flash-next/QFP53.md` and this package's `SUMMARY.md`. This patch is **not** in production or an active experiment. Do not use the old `--experiment kv-pad-prompt` invocation or queue further 1362 qualification without new independent evidence.

## Historical mechanism (retained for evidence)

`BIGCHERRY_KV_PAD=N` pads both `llama_kv_cache::get_n_kv` and Qwen4Exp/GLM5-Next `kpool_pad` to larger quanta (pool quantum derived from `N/indexer_kpool`). Zero/unset preserves upstream behavior; nonzero N must be a multiple of 256 and no larger than the active KV cache. The patch preserves dead/masked pools and `n_kv` sentinels. It was tested on the b11552 composition.

After the k-pool fix, 24/40 512-token prompt submits reused the graph and host graph-build time fell from 7.1 to 3.0 ms/batch. But shared input storage increased the input step from 3.3 to 17.0 ms/batch; Flash-Next prefill remained neutral at 8K/24K. Dense Qwen3.8-27B was slower at 10K/32K and had different greedy output at 32K. The graph-build work was already hidden by 1359 queue-ahead. **Reuse count is not a performance acceptance metric.**

## Archived mechanics checks (not promotion evidence)

- `python tools/tests/patch/test_1362_kv_pad_prompt.py` (requires pinned vendor source; may skip when absent).
- `python -m bigcherry patch-lint` and `python -m bigcherry patch-rebase-check --source bigcherry` for historical patch integrity only.

Do not re-enable this package to investigate the remaining prompt-boundary gaps. QFP48 owns the read-only HIP API/device timeline; QFP50/1359 owns queue-ahead and QFP27/1326 owns input-copy lifetime.