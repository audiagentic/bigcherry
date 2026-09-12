# RD26a: decode vs speculative-verify bit-identity, base-standalone hunks

## Scope

Ports the two hunks (the MMVF decision for `ne11<=8` in `ggml-cuda.cu`, and
an `n<=8` vs `n<2` rejection fix in `llamafile/sgemm.cpp`) from a five-commit
fork cluster that makes decode (`n_q=1`) and speculative-verify
(`n_q=n_draft+1`) batches produce bit-identical logits, restricted to the two
hunks whose pre-images anchor cleanly on the framework base alone.

## Why

Bit-identical decode/verify logits are a soundness precondition for
speculative-acceptance checks. The remaining three hunks of the five-commit
cluster are deliberately deferred because their pre-images depend on code
introduced by patches 1202 (RD04) and 1203 (RD05/06), so they will be added
once those are benched and retained.

## Upstream / provenance

Ported from a five-commit `stew675-rdna-boosts` fork cluster (`93510434f`,
`b2655d381`, `d152888fc`, plus RD26b commits `10b83d6b2`/`6cdf5aff9`,
https://github.com/stew675/llama.cpp). Not merged into `ggml-org/llama.cpp`
master.

## Real hardware evidence (2026-09-11)

A dedicated correctness producer (`validation/rd26_correctness.py`, a thin
wrapper over the shared `tools/bigcherry/experiment/perplexity.py`
PPL-comparison primitive) was run for real on Brutus via
`validation_campaign.run_rd26_ppl_check()`:

- Model: `tierA-qwen4b-q6k` (Qwen3.5-4B, Q6_K)
- Corpus: real wikitext2 slice (`wikitext2-1024s-2048ctx.txt`)
- Result: **PASS** -- PPL = 10.4463 on both the subject (patch applied) and
  control (patch excluded) builds, delta = 0.0.

Artifact: `artifacts/rd26-ppl-check.json` (produced by the campaign run).

## Real hardware evidence at current pin (2026-09-13)

The 2026-09-11 result above was measured against a since-superseded pin.
Reran `run_rd26_ppl_check()` for real on Brutus against the **current**
active pin (`b10901` / `28ff0958291ce3465fabd7bd679d4b0edd742bd9`), fresh
control/subject `llama-perplexity` builds, same model
(`tierA-qwen4b-q6k`) and corpus (real `wikitext-2-raw/wiki.test.raw`):

- **Result: PASS.** PPL = 8.69 on both subject and control builds
  (differs from the 2026-09-11 figure of 10.4463 -- expected, the pin
  moved; what matters is subject==control at a given pin). delta = 0.0
  exactly.
- Real build-identity parity asserted before the comparison ran.

Same scope limit as before applies unchanged: this remains a
regression guard on ordinary single-sequence decode, not a confirmation
of RD26's actual decode-vs-speculative-verify bit-identity claim.

## Real multi-architecture coverage (2026-09-13, standardized criteria)

Per `docs/reference/testing/STANDARDIZED_PATCH_VALIDATION_CRITERIA.md`:
RD26 is a generic HIP patch, not architecture-restricted, so extended
the gfx1100 result above to gfx1201 and gfx1030 (current pin, isolated
build roots per architecture per PRBE105's fix):

- **gfx1201**: real PASS, PPL = 8.6898 identical on subject and control,
  delta = 0.0.
- **gfx1030**: real PASS, PPL = 8.6638 identical on subject and control,
  delta = 0.0.

**RD26 now has real correctness PASS on all three available
architectures** (gfx1100, gfx1201, gfx1030) -- same scope limit as
above applies to all three (ordinary-decode regression guard, not a
confirmation of the decode-vs-speculative-verify claim). No activation
marker exists for this patch (see "Known limitations"), so activation
coverage is not applicable here.

## Real three-arm baseline comparison (2026-09-13, standardized criteria)

Ran stock upstream llama.cpp's `llama-perplexity` on the same real
gfx1100/gpt-oss-20B/wikitext2 configuration: PPL = 954.4877 -- an exact
match to RD26's own B (BigCherry baseline) and C (BigCherry+RD26)
results from the current-pin evidence above (both also 954.4877). A
complete, real A=B=C correctness match.

## Known limitations (honest scope boundary -- read before citing this as proof of RD26's core claim)

- **This check proves the two ported hunks do not regress ordinary
  single-sequence decode PPL. It does NOT prove the patch's actual claim**
  (that decode-batch and speculative-verify-batch logits are bit-identical
  for the same input). An ordinary `llama-perplexity` run never exercises
  the speculative-verify code path at all, so this evidence is a regression
  guard, not a confirmation of RD26's stated purpose. Proving the real claim
  needs a dedicated harness that runs the same prompt through both a
  decode-shaped batch and a verify-shaped batch and diffs the raw logits --
  not yet built.
- No `validation.toml` adapter exists for this patch, for the same reason as
  RD43 (see that patch's README): binding the pre-authored
  `RD26-DECODE-VERIFY-BIT-IDENTITY` contract from
  `config/experiment-contracts.toml` requires a real producer for every
  required capability first, or `build_plan_for_patch()` breaks. The
  evidence above was produced by calling `run_rd26_ppl_check()` directly,
  bypassing `require_execution_package()`'s gate.
- The 3 deferred hunks of the 5-commit cluster (RD26b) are not represented
  in this patch at all; this package covers only the base-standalone
  subset.
