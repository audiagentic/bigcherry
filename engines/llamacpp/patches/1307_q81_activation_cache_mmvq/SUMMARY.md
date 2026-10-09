# 1307_q81_activation_cache_mmvq

**Status:** validated
**Plan item:** PRBE05/PRBE06/QFP13

## What it does

Single package for the Q8_1 activation-cache mechanism: per-context stable cache foundation, MMVQ lookup/publish, RMSNorm+MUL Q8_1 producer, plain/gated activation producer, hyper-connection producer, and fused unary*MUL producer. The former package IDs 1235, 1309, 1310, 1311 and 1312 are retired into this package; runtime flags and activation markers are unchanged.

## PA44-E packaging proof

This is a packaging-only merge. The implementation retains the old edit sequence and historical env-doc source rows. `1313_scale_act_fuse` depends on this package directly because the former 1310 helper is now internal to 1307. The merged README preserves the former READMEs verbatim.
