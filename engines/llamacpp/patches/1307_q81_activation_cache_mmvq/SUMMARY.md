# 1307_q81_activation_cache_mmvq

**Status:** validated
**Plan item:** PRBE05/PRBE06/QFP13

## What it does

Single package for the Q8_1 activation-cache mechanism: per-context stable cache foundation, MMVQ lookup/publish, RMSNorm+MUL Q8_1 producer, plain/gated activation producer, hyper-connection producer, and fused unary*MUL producer. The former package IDs 1235, 1309, 1310, 1311 and 1312 are retired into this package; runtime flags and activation markers are unchanged.

## PA44-E packaging proof

This is a packaging-only merge. The implementation retains the old edit sequence and historical env-doc source rows. `1313_scale_act_fuse` depends on this package directly because the former 1310 helper is now internal to 1307. The merged README preserves the former READMEs verbatim.

## PA44-B owned definitions

The four whole file-scope definitions formerly pasted by the merged 1309/1310/1311 parts now live in three
BigCherry-owned include files beside their CUDA sources: `bc-rms-norm-q81.cuh`, `bc-act-q81.cuh`, and
`bc-hc-pre-q81.cuh`. The MMVQ cache wiring, graph-compute/capture edits, dispatch hooks, state and env docs are
unchanged. The graph-compute edit remains upstream because it edits the beginning of an existing definition rather
than adding a whole definition.

## Identity proof

The moved source is unchanged: 187 physical lines / 180 non-blank whitespace-stripped lines before and after,
covering four definitions in three blocks. Replacing the three new include lines with their owned-file contents
reconstructs the pre-refactor composed source exactly under the non-blank whitespace-stripped multiset check; the
new form differs only by the three include lines (owned-definition multiset 180/180, include delta +3).
