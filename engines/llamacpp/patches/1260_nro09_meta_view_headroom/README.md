# 1260 (PNRO09): Meta compute-container view headroom


## 2026-10-12 audit correction

**Untested; not qualified for promotion.** Meta's 16 is a per-static-tensor compute-container multiplier, not a global 16-view cap. The historical real MTP workload succeeded at both 16 and 80; its throughput comparison is confounded. The current validation producer cannot start as written: it confuses build targets with GPU architectures, lacks materialized source trees, uses a zero-length view fixture and copies historical hardware results. See PNRO09 and BCOP157 for a bounded repair/retirement gate. No new build or GPU run was performed.

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
