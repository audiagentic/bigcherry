# RD43: keep the concurrent-region join node out of op-fusion

## Scope

Temporarily lowers `cgraph->n_nodes` around `ggml_cuda_try_fuse` while a
concurrent region is active, so op-fusion cannot absorb the join `add` node
that rejoins an auxiliary stream. A `GGML_ASSERT` catches any future fusion
pattern that still crosses the join.

## Why

Without this guard, op-fusion could absorb RD42's (patch 1215's)
shared-expert join `add` as a preceding matmul's bias-add, so the join
handler never runs, the aux stream is never rejoined, and CUDA/HIP graph
capture aborts with "capturing stream has unjoined work".

## Upstream / provenance

Ported from AMD-Ecosystem/llama.cpp PR #71 (merge commit `0f0db6292`,
https://github.com/AMD-Ecosystem/llama.cpp). Fork-only work; applies after
patch 1215.

## Real hardware evidence (2026-09-11)

A dedicated correctness producer (`validation/rd43_correctness.py`, a thin
wrapper over the shared `tools/bigcherry/experiment/perplexity.py`
PPL-comparison primitive) was run for real on Brutus via
`validation_campaign.run_rd43_ppl_check()`, uniquely setting
`GGML_CUDA_GRAPH_OPT=1` (`GRAPH_OPT_ENV`) for both arms -- the real condition
under which RD42's join/graph-capture interaction actually reproduces:

- Model: `tierM-gptoss20b-q6k` (gpt-oss-20B, Q6_K)
- Corpus: real wikitext2 slice (`wikitext2-1024s-2048ctx.txt`)
- Result: **PASS** -- no graph-capture abort under `GGML_CUDA_GRAPH_OPT=1`,
  PPL = 561.6933 on both the subject (patch applied) and control (patch
  excluded) builds, delta = 0.0.

Artifact: `artifacts/rd43-ppl-check.json` (produced by the campaign run).

## Real performance evidence as part of the combined 1215+1216 unit (2026-09-12)

See `patches/1215_rd394041_amd_stream_moe_overlap/README.md` for the full
real-hardware performance campaign -- 1215 and 1216 must be qualified
together (1215 alone exposes the join-node fusion hazard this patch
fixes). Summary: real, reproducible, non-regressing +2.47% mean gain on
gfx1100 single-GPU under `GGML_CUDA_GRAPH_OPT=1`, direct profiler proof of
82.28% real temporal overlap between main and auxiliary CUDA/HIP streams,
GPT-reviewed. Not yet `validated` -- 1216's own Experiment Contract
requires a `backend_reference` correctness check, which the existing
PPL-equality evidence does not formally substitute for.

## Known limitations

- No `validation.toml` adapter exists for this patch. `patch-lint`'s package
  policy does not currently require one (RD43 is not in
  `config/external-sources.toml`'s tracked `PACKAGE_STATUSES` set and is not
  an RD-validation patch at `state = "validated"`), so this is a deliberate
  gap, not an oversight: adding one would require binding
  `config/experiment-contracts.toml`'s pre-authored
  `RD43-CONCURRENT-JOIN-FUSION-GUARD` contract, and binding a contract
  without wiring a real producer for every one of its required capabilities
  breaks `build_plan_for_patch()` (confirmed earlier this campaign by
  reverting an attempt across 9 patches). The correctness evidence above was
  produced by calling `run_rd43_ppl_check()` directly, bypassing
  `require_execution_package()`'s gate -- this is a standalone campaign
  function, not yet CLI-wired.
- No trace-marker activation check exists (RD43's guard has no
  `BIGCHERRY_PATCH_HIT`-style marker); the correctness evidence proves no
  PPL regression under the real reproduction condition, not that the guard
  code path specifically executed during that run.
- RD43 requires patch 1215 (RD39/40/41) as a hard prerequisite.
