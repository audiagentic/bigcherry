# 1000: RDNA4 MMQ codegen fixes for Q2_K and Q6_K

## Scope

Cherry-picks two narrowly-scoped fixes from unmerged upstream PR #25940
into the MFMA/WMMA MMQ `vec_dot` for Q2_K (forces a plain loop via
`#pragma unroll 1` to avoid a ROCm over-unroll/spill) and Q6_K (adds an
explicit float cast before a scale multiply to change ROCm's codegen).
Deliberately excludes the PR's second change (a hand-written RDNA4
native-select heuristic), since this project's own tuner already measures
candidates head-to-head per shape.

## Why

The PR's own numbers show large RDNA4 gains (Q6_K 1.90x, Q2_K 28.2x at
n=512); both quant types are in this project's own test corpus and
hardware, so the fix was taken ahead of upstream merge rather than waiting.

## Upstream / provenance

Cherry-picked from open upstream PR
https://github.com/ggml-org/llama.cpp/pull/25940.

## Known limitations

**Gap found 2026-09-11 (process audit)**: this patch is tagged
`optimization` and carries `state = "validated"`, but its performance
claim (Q6_K 1.90x, Q2_K 28.2x) is **entirely the upstream PR's own
reported numbers** -- no independent measurement of any kind (native
llama.cpp, BigCherry baseline, or BigCherry+patch) has been recorded on
this project's own hardware for this patch specifically. This is a real
gap, not just a missing-baseline-arm gap like 1001/1200's -- there is no
first-party performance evidence here at all. Per this project's new rule
(`docs/reference/patches/PATCH_AUTHORING.md`'s "`optimization` carries a
real validation obligation"), flagged for a real measurement (a genuine
3-arm sweep, since none exists) rather than continuing to rest on the
upstream PR's own claim.
