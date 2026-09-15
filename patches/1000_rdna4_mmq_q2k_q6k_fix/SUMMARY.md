# 1000_rdna4_mmq_q2k_q6k_fix: Upstream backport: RDNA4 MMQ codegen fixes for Q2_K and Q6_K

**Status:** rejected (2026-09-16, see DEMOTION section below)
**Plan item:** PA35

## What it does

Cherry-picks two narrowly-scoped fixes from unmerged upstream PR #25940 into the MFMA/WMMA MMQ vec_dot for Q2_K (forces a plain loop via #pragma unroll 1 to avoid a ROCm over-unroll/spill) and Q6_K (adds an explicit float cast before a scale multiply to change ROCm's codegen).

## Why

The PR's own numbers show large RDNA4 gains (Q6_K 1.90x, Q2_K 28.2x at n=512), and both quant types are in this project's own test corpus and hardware, so the fix is worth taking ahead of upstream merge rather than waiting.

## Upstream / provenance

Cherry-picked from open upstream PR https://github.com/ggml-org/llama.cpp/pull/25940. Deliberately excludes the PR's second change (a hand-written RDNA4 native-select heuristic), since this project's own tuner already measures candidates head-to-head per shape.

## DEMOTION (2026-09-16, PA35)

The "Why" section above states the original promotion justification verbatim
and is preserved, not rewritten -- it reflects the PR's own reported numbers
(Q6_K 1.90x, Q2_K 28.2x at n=512), which is what this patch's claim actually
was: a performance-optimization claim, not a correctness/behavior-bug claim,
despite `kind = "upstream-backport"` and prior membership in
`config/recipes.toml`'s `[patch-set.upstream-fixes]` (whose doctrine is for
correctness backports only -- this patch was mis-slotted there).

Real hardware evidence gathered 2026-09-16 (PA35 step 1, gfx1201/RDNA4,
exact-shape backend-ops paired A/B, control=serving-core without 1000 vs
subject=serving-core+1000, 5 rounds, correctness PASS both arms both
dtypes) -- full record in `evidence/validation.json`:

- **Q6_K: 1.365x [CI95 1.362x-1.367x]** -- a real, statistically significant
  gain, but materially below the claimed 1.90x.
- **Q2_K: 0.959x [CI95 0.954x-0.963x]** -- a real, statistically significant
  **REGRESSION**, directly contradicting the claimed 28.2x gain.

GPT design-review (dev-gpt-agent, `req_24ceb64100cb41dd`) confirmed the
governing axis is the patch's semantic claim/effect (performance
optimization), not its provenance (upstream backport), and recommended:
reject this combined package (a shipped package containing a real,
statistically-established regression cannot remain validated), extract the
independently-correct Q6_K fix into a new patch
(`1006_rdna4_mmq_q6k_codegen_fix`, untested pending its own fresh Q6-only
hardware A/B -- the combined-package Q6 measurement above is evidence this
mechanism works, but is not reused as sufficient promotion evidence for a
different composition identity), and leave no production patch for the
Q2_K change.

**Decision executed:** `patch.toml` `state`: `validated` -> `rejected`.
Removed from `config/recipes.toml`'s `[patch-set.upstream-fixes]` (see that
file's comment block for the full removal rationale, RD73-precedent
pattern). This `patch.py`/implementation is retained unchanged as the
historical rejected artifact -- it is not deleted, edited, or repurposed.
