# RD13: mul_mat + RESHAPE + add fusion

## Scope

Extends the existing `mul_mat`+`add` fusion in `ggml_cuda_try_fuse` to accept one
`RESHAPE` node between the matmul and the add (via `ggml_can_fuse_subgraph`,
verifying the view's `src[0]` is the matmul), instead of only matching an `add`
node directly after the matmul.

## Why

SSM/MoE models (e.g. Qwen3-MoE-family) insert a reshape view between the output
projection and the residual add, so the existing fusion never fired for them and
every layer ran a separate `add` kernel instead of the fused epilogue.

## Upstream / provenance

Ported from `stew675-rdna-boosts` fork commit `0153d580d`
(https://github.com/stew675/llama.cpp). Not merged into `ggml-org/llama.cpp` master.

## Validation package

`validation.toml` wires three required checks: `apply`, `build`, and `activation`
(trace-marker regex `BIGCHERRY_PATCH_HIT patch=1206_rd13
path=mul_mat_add_view_fusion_(?:f|q)`). There is no `correctness` or `performance`
check bound in `validation.toml` -- this patch has no declared Experiment Contract
(see "Known limitations" below), so `require_execution_package()`'s
`is_framework_configuration_patch` path applies rather than a contract-driven
capability set.

## Real hardware evidence (2026-09-11)

A dedicated correctness producer (`validation/rd13_correctness.py`, a thin
wrapper over the shared `tools/bigcherry/experiment/perplexity.py` PPL-comparison
primitive) was run for real on Brutus via
`validation_campaign.run_rd13_ppl_check()`:

- Model: `tierM-gptoss20b-q6k` (gpt-oss-20B, Q6_K)
- Corpus: real wikitext2 slice (`wikitext2-1024s-2048ctx.txt`)
- Result: **PASS** -- PPL = 561.6933 on both the subject (patch applied) and
  control (patch excluded) builds, delta = 0.0, well within the 3-sigma
  combined-uncertainty tolerance.

Artifact: `artifacts/rd13-ppl-check.json` (produced by the campaign run).

## Known limitations

- The PPL-equality check confirms the ported fusion introduces no numerical
  regression versus the unfused baseline. It does **not** independently confirm
  that the RESHAPE-mediated fusion path actually activated during that specific
  run (that is what the `activation` trace-marker check in `validation.toml`
  is for, run separately) -- the PPL check and the activation check are
  complementary, not substitutes for each other.
- No Experiment Contract is bound for RD13 in `patch.toml`. One exists,
  unreferenced, in `config/experiment-contracts.toml`
  (`RD13-MUL-MAT-ADD-VIEW-FUSION`) from an earlier metadata migration, but
  binding it requires wiring every one of its required capabilities to a real
  producer in `validation.toml` first -- binding without that breaks
  `build_plan_for_patch()` (confirmed by reverting an earlier attempt across
  9 patches in this same campaign). Left unbound deliberately.
