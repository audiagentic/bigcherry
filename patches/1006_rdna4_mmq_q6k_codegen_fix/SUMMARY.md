# 1006_rdna4_mmq_q6k_codegen_fix: Upstream backport, RDNA4 MMQ codegen fix for Q6_K only

**Status:** untested (pending fresh Q6-only hardware A/B)
**Plan item:** PA35

## What it does

Cherry-picks the Q6_K half of unmerged upstream PR #25940 into the MFMA/WMMA
MMQ vec_dot: adds an explicit float cast before a scale multiply to change
ROCm's codegen. This is a split of the rejected patch
`1000_rdna4_mmq_q2k_q6k_fix`, keeping only the edit whose real-hardware
measurement was a genuine gain.

## Why

PA35 step-1 real hardware evidence (2026-09-16, gfx1201/RDNA4, exact-shape
backend-ops paired A/B, control=serving-core without 1000 vs
subject=serving-core+1000, 5 rounds, correctness PASS): Q6_K measured
1.365x [CI95 1.362x-1.367x], a real, statistically significant gain (below
the PR's own claimed 1.90x, but real). The same measurement found Q2_K
measured 0.959x [CI95 0.954x-0.963x], a real regression -- see
`patches/1000_rdna4_mmq_q2k_q6k_fix/SUMMARY.md`'s DEMOTION section for the
full record and the GPT design-review (dev-gpt-agent, req_24ceb64100cb41dd)
that recommended this split.

## Upstream / provenance

Cherry-picked from open upstream PR
https://github.com/ggml-org/llama.cpp/pull/25940. Excludes the PR's Q2_K
change (real regression on this project's hardware) and its second change
(a hand-written RDNA4 native-select heuristic), since this project's own
tuner already measures candidates head-to-head per shape.

## Lifecycle note

`state = "untested"`. The 1.365x figure above was measured against the
combined 1000 composition (Q2_K edit also present) and is preserved as
directional evidence only -- it is explicitly NOT treated as sufficient
promotion evidence for this patch, since the composition/subject digest
differs. A fresh Q6-only hardware A/B (same exact-shape backend-ops
methodology) is required before this patch can be promoted to `validated`
and added to `config/recipes.toml`'s `[patch-set.validated-enhancements]`.
This is tracked as a PA35 follow-up, not yet executed.
