# 1260 (PNRO09): Meta compute-container view headroom

## Scope

Raises `compute_headroom` in `ggml/src/ggml-backend-meta.cpp` from 16 to 80
so the Meta backend's compute-container object pool can hold the
between-eval views that recurrent + MTP graphs create
(`2*(n_rs_seq+1)` views per recurrent slot). Memory sizing only; no
numerical change.

## Validation

Contract `PNRO09-META-VIEW-HEADROOM` (correctness/capacity), producer
`nro09`:

- capacity (C): `validation/meta_boundary_test.cpp` probes the largest view
  count `N_max` that fits for two leaf sizes on both variants; the patched
  ratio must be ~80/16 = 5;
- real graph (D): tierA-qwen4b-q6k MTP decode on both variants completes
  without a Meta-backend object-pool abort.

The recorded (D) throughput difference (98.3 vs 76.5 gen t/s) is NOT a
performance claim: a headroom change cannot explain a 28% decode gap, so
that comparison was confounded and must not be cited.

## Status

untested.
