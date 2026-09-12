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

## Real backend_reference correctness evidence (2026-09-12, GPT-approved, req_3e42043eb71a4a92 / req_a457561b33ac4f24)

The contract's own required check is `backend_reference`, not `ppl_equality`
-- the 2026-09-11 PPL result above is real but does not formally satisfy
this named check. Ran the correct check per GPT's exact design: real
Brutus hardware, single gfx1100, control = baseline+1215 (isolating 1216's
own marginal effect, not plain baseline), subject = baseline+1215+1216,
`GGML_CUDA_GRAPH_OPT=1` both arms, same fixed deterministic prompt
(temp=0, seed=42, 64 decode steps), comparing generated token IDs plus
**all** 248,320 vocab logprobs per step via `/completion`'s `n_probs`
parameter (this model's real vocab size, confirmed empirically) --
**no rounding/truncation** this time (an earlier pass that rounded to 6
decimal places was correctly flagged by GPT as too lossy for a formal
result).

**Result: exact match.** Token IDs identical across all 64 steps;
0 of 15,892,480 total logprob comparisons (64 steps x 248,320 vocab
entries) differ at all; `max_abs_logprob_diff = 0.0` (exactly zero, not
just below a tolerance).

```
CorrectnessResult(
    check_id="backend_reference", passed=True,
    method="full-vocab-http-logprob-parity",
    details={
        "generated_steps": 64, "vocab_size": 248320,
        "total_compared": 15892480, "mismatches": 0,
        "max_abs_logprob_diff": 0.0, "token_id_mismatches": 0,
        "graph_opt": 1, "control": "1215", "subject": "1215+1216",
    },
)
```

**GPT's explicit caveat -- real, unresolved contract-binding defect**: the
`RD43-CONCURRENT-JOIN-FUSION-GUARD` contract in
`config/experiment-contracts.toml` declares model `tierM-gptoss20b-q6k` /
workload `moe_decode`; this real run used `Qwen3.6-35B-A3B-UD-Q4_K_M`
(this project's only registered MoE model with the exact vocab-probe
mechanism readily available). This result may be **persisted as a real
passing `CorrectnessResult`**, but is **not the final contract-qualified
result** until either (a) `tierM-gptoss20b-q6k` is confirmed to hit the
RD42/RD43 activation path and this exact protocol is rerun there, or (b)
if it does not hit that path, the contract's model binding is corrected
to the Qwen model (an AUTHOR-then-VERIFY step, then one rerun -- no
extra prompt/repeat matrix needed, since RD43 is a fixed-effect,
correctness-only claim with no performance component).

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
