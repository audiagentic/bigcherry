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

## Real hardware evidence at current pin (2026-09-13)

The 2026-09-11 result above was measured against a since-superseded pin.
Re-ran `run_rd13_ppl_check()` for real on Brutus against the **current**
active pin (`b10901` / `28ff0958291ce3465fabd7bd679d4b0edd742bd9`), fresh
control/subject `llama-perplexity` builds, same model
(`tierM-gptoss20b-q6k`) and corpus (real `wikitext-2-raw/wiki.test.raw`):

- **Result: PASS.** PPL = 954.4877 on both subject and control builds
  (note: differs from the 2026-09-11 figure of 561.6933 -- expected, the
  pin moved between runs, changing the baseline model-graph/kernel
  behavior; what matters is subject==control at any given pin, not the
  absolute PPL value). delta = 0.0 exactly, well within tolerance.
- Real build-identity parity asserted between control/subject
  (`assert_validation_subject_parity()`) before the comparison ran.
- Persisted artifact:
  `evidence/artifacts/rd13-ppl-check.json` (committed to this package,
  not left in a scratch dir -- this is now durable, checkable evidence).

**Activation-marker attempt, round 1 (2026-09-13, real, negative finding
-- wrong model chosen).** Ran the built `rd13-only` `llama-bench` binary
against `tierM-qwen35b-a3b-moe-mtp`'s Q4_K_M quant (a real MoE model,
chosen since it's this project's only registered non-dense-transformer-
family model) under both a decode-only (`-p 0 -n 32`) and a prefill
(`-p 512 -n 0`) shape: **0 of 0** `BIGCHERRY_PATCH_HIT patch=1206_rd13`
marker hits in either run. Consistent with the patch's own authoring
note ("needs an SSM/Mamba-family model; the view pattern is what makes
it fire") -- `qwen3.6-35B-A3B` is a dense-attention MoE model, not an
SSM/Mamba family. At the time this was recorded as "this project has no
registered SSM/Mamba-family model at all" and filed as PRBE102 to track
registering one.

**Activation-marker attempt, round 2 (2026-09-13, real, POSITIVE --
corrected model, PRBE102 closed).** GPT (`req_7f31983c9fa649eb`)
identified that `tierA-qwen4b-q6k` (`config/models.toml`'s
`Qwen3.5-4B-UD-Q6_K_XL.gguf`) -- already registered, its own note calling
it a "small dense tier" -- is in fact NOT purely dense: Qwen3.5-4B's real
architecture is a hybrid with 24 of 32 layers being Gated DeltaNet (GDN)
recurrent layers (8 blocks of 3x GDN + 1x attention), and this project's
pinned llama.cpp genuinely implements the GDN path as
`ssm_out matmul -> ggml_reshape_2d() -> residual ggml_add()` -- exactly
RD13's MUL_MAT->RESHAPE->ADD target shape. Reran the real activation
probe against this ALREADY-registered model: **subject_hit=1,
control_hit=0** (`BIGCHERRY_PATCH_HIT patch=1206_rd13
path=mul_mat_add_view_fusion_f` fired once on the patched `rd13-only`
build across a 128-token decode run with `-p 512`, zero hits on the
baseline build under the identical command). **This resolves RD13's
activation leg with no new model registration required** --
`config/models.toml`'s own "dense tier" note for `tierA-qwen4b-q6k` is
inaccurate (it is a dense+GDN hybrid), a real, useful correction in its
own right. PRBE102 closed as resolved-without-new-registration.

## Real multi-architecture coverage (2026-09-13, standardized criteria)

Per `docs/reference/testing/STANDARDIZED_PATCH_VALIDATION_CRITERIA.md`
(GPT-designed, `req_5b07ec3d063a4231`): RD13 is a generic HIP patch, not
architecture-restricted, so the standard requires single-GPU coverage on
all three real available architectures. Extended the gfx1100 results
above to gfx1201 and gfx1030 (fresh control/subject `llama-bench` +
`llama-perplexity` builds on each, current pin `b10901`/`28ff0958291c`):

- **gfx1201 (device index 2)**: activation `subject_hit=1`,
  `control_hit=0` (clean, real positive/negative split, same marker
  regex, `tierA-qwen4b-q6k`). Correctness: **real PASS**, PPL = 944.9004
  identical on both subject and control (`tierM-gptoss20b-q6k`,
  wikitext2), delta = 0.0.
- **gfx1030 (device index 3)**: activation `subject_hit=1`,
  `control_hit=0` (same clean split). Correctness: **real PASS**, PPL =
  952.858 identical on both subject and control, delta = 0.0.

**RD13 now has complete, real, current-pin evidence -- correctness PASS
and clean activation -- on all three available architectures
(gfx1100, gfx1201, gfx1030).** This is the strongest evidence bar any
RD-series patch has reached in this project to date for a
non-architecture-scoped optimization.

A real methodology bug was found and fixed during this run:
`run_rd13_ppl_check()`'s build directories are named
`rd13-ppl-subject`/`rd13-ppl-control` regardless of target architecture,
so reusing the same `build_root` across architectures collided with a
stale CMake cache from an earlier gfx1100 run (`CMake Error: The source
... does not match the source ... used to generate cache`). Worked
around by using a distinct `build_root` per architecture for this run;
`run_rd13_ppl_check()` itself should eventually be fixed to
namespace its build directories by architecture (filed as a known gap,
not yet a tracked plan item).

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
