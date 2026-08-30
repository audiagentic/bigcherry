# HI141: mmvq:q8_0:w4:nw8:rpb1:sk0:v1 MTP acceptance regression, 2026-08-29

Real hardware investigation (dual RX 7900 XTX, Qwen3.8-27B-Q8_0, `--spec-type
draft-mtp`) into a genuine, reproducible regression: tuned dispatch caused
MTP draft acceptance to drop from 93.5% (native) to 62.1% (tuned) in the
decode segment immediately following a 4096-token prefill. Full narrative,
ablation chain, and GPT adversarial-review record are in
`docs/planning/active/hip-autotune/HI141.md` (or its completed/ location once
closed) -- this bundle holds the raw supporting data for the final two
decisive tests.

## Files

- `nw_rpb_isolation.txt` -- real hardware, real `test-backend-ops`
  correctness-evidence runs (10 seeds each) at the exact real production
  signature (`ne1=[5120,4,1,1]`) for all 4 combinations of
  `nwarps in {1,8}` x `rows_per_block in {1,2}`. Result: `nwarps=8` alone is
  the isolating variable (identical `e_c_nmse=6.269357e-14` for both rpb
  values); `rows_per_block` has zero measurable effect; `nwarps=1` matches
  native exactly at every seed for both rpb values.

- `elementwise_delta.py` -- the diagnostic script used to dump and compare
  raw per-element MUL_MAT output arrays (native vs `nw1:rpb1` clean vs
  `nw8:rpb1` guilty) at the same production signature. Depends on a
  throwaway, never-landed instrumentation patch to the vendored
  `tests/test-backend-ops.cpp` (see script docstring) that dumps `f1`/`f2`
  to `BIGCHERRY_DUMP_ELEMENTS_DIR` when set -- this was applied, used once,
  then reverted; it is not part of the `patches/` tree and must be
  reapplied by hand (see `elementwise_delta_output.log`'s trailing note) if
  this exact analysis needs to be repeated.

- `elementwise_delta_output.txt` -- the real captured output of that run.
  Decisive result: `nw1:rpb1` vs native is bit-identical (0 of 496640
  elements differ). `nw8:rpb1` (guilty) vs native differs in 379334 of
  496640 elements (76.38%), delta range 1.192e-07 to 1.907e-05 (all far
  below the 5e-4 correctness threshold), spread across all 5120 of 5120
  output rows with no spatial/lane clustering -- the fingerprint of
  legitimate floating-point reassociation from a different warp-reduction
  tree shape at `nwarps=8`, not a structured indexing/lane defect.

## Conclusion (adversarially reviewed by GPT, session `ses_330ae3c055084f38`)

The guilty candidate's kernel is numerically valid under HI67's existing
correctness contract -- `nwarps=8` produces a different but conformant
floating-point accumulation order. It is nevertheless behaviorally unsafe
under MTP's zero-tolerance greedy-argmax verification, which can be flipped
by a numerically legitimate perturbation as small as these.

Critically, this numeric error class is **not** itself predictive of
behavioral safety: the sibling candidate `mmvq:q8_0:w4:nw8:rpb2:sk0:v1`
carries the *identical* aggregate NMSE signature (confirmed in
`nw_rpb_isolation.log`) yet was independently ablated as behaviorally clean
(`tg128=100.91`) against the guilty `nw8:rpb1`'s reproduced regression
(`tg128=74.63`). This forecloses any candidate-exclusion policy keyed on
warp count or on aggregate numerical-correctness metrics (NMSE thresholds,
max_abs bounds) as a fix for this defect class.

**No kernel code change was made or is planned.** The durable fix is HI143's
mandatory real-generation behavioral gate for MTP-relevant promotions
(already implemented and wired into `tools/bigcherry/tuning/workflow.py`'s
promotion path) -- it has already been shown, on real hardware, to hard-fail
the actual guilty candidate while passing its numerically-identical but
behaviorally-clean sibling.

## Independent end-to-end proof (run-id `hi141-proof-20260829-2231`)

A completely fresh, unmanipulated tune-campaign was run afterward as a
direct real-hardware proof that the wired gate actually protects
production, not just that it can be made to pass a targeted falsification
test. New build, new record, new tune measurements -- no cache or
inventory reuse from any earlier run.

- `hi143_proof_campaign.txt` -- the campaign's own stderr tail. The
  campaign genuinely failed (exit 1) with:
  `tune-campaign: behavioral gate hard-fail: a promoted candidate's
  generated output diverged from native on a real regression vector --
  refusing to ship this cache`.
- `hi143_proof_behavioral_gate.json` -- the real gate verdict. Native MTP
  draft trace `[107, 100]`; candidate draft trace `[145, 90]`;
  `first_output_divergence: 1`; `verdict: hard_fail`.

The historically-guilty candidate `mmvq:q8_0:w4:nw8:rpb1:sk0:v1` was
re-selected in this fresh campaign as a provisional winner purely on its
own measured speed (one of 41 provisional non-native winners this run,
confirmed via the campaign's own `promoted.jsonl`) -- not forced, not
hand-picked. It passed synthetic correctness evidence, as it always does.
The behavioral gate caught the real divergence and the campaign correctly
refused to promote: only `dispatch.cache.provisional` exists in the
run's workdir; no final `dispatch.cache` was ever written.

This is the direct inverse of the earlier "candidate-mix masking" run
recorded in HI143's own notes, where the same guilty candidate, promoted
in a *different* fresh campaign's candidate mix, produced `exact_pass`.
Two independent real end-to-end campaigns, same guilty candidate, two
different real outcomes -- exactly the behavior HI141's closure predicts
for a candidate-mix-dependent defect, and exactly the job HI143 is
designed to do: not claim a candidate is universally safe or unsafe, but
catch a real regression in the specific cache about to ship.
