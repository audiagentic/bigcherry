# Dual-XTX real tune run, 2026-08-27

Real captured data from a manual (pre-`tune-campaign`, see HI130) run of the
full record -> tune -> correctness-evidence -> promote pipeline on Brutus,
using the production dual-XTX tensor-split profile against a real model.

**Hardware/model**: 2x RX 7900 XTX (gfx1100), tensor-split (`-sm tensor`),
Qwen3.8-27B-Q8_0 with MTP (`--spec-type draft-mtp`), matching the
`dual-xtx-27b` production bench profile (see the `project-dual-xtx-baseline`
memory note) -- `-c 64000`, `--flash-attn on`, `--ubatch-size 512`,
`--batch-size 2048`, `-ctkd q8_0 -ctvd q8_0`.

## Files

- `e2e_dualxtx_inventory.json` -- real record-mode inventory (66 signatures)
  captured from one driven completion request.
- `e2e_dualxtx_tune.measurements.jsonl` -- raw tune-mode measurements: 52
  results, 8296 measurements, 161 candidates. Real GPU timing data,
  `screen_samples=3`, `final_samples=15`.
- `e2e_dualxtx_tune.promoted2.jsonl` -- the SECOND `tune-promote` pass, run
  after real CPU-reference correctness evidence was generated for every
  non-native provisional winner (via `hi80_generate_correctness_evidence.py`
  against a `hi105-correctness`-patched `test-backend-ops` binary). Result:
  **20/39 candidates genuinely promoted** (`bh_accepted=true`, `q_value=0.0`,
  improvements up to +24.73%, e.g. `mmvq:q8_0:w1:nw1:rpb2:sk0:v1`). Each row
  also carries a full `ranking_decisions` array -- every candidate considered
  for that signature, with its verdict (`winner`/`outside_tie_band`/etc.) and
  `effective_us` -- not just the winner, so the full selection heuristic is
  auditable per-signature, not just the final pick.
- `e2e_dualxtx_cov.json` -- real replay-mode dispatch coverage from building
  and running a replay cache exported from the 20 promoted winners: 100%
  dispatched==executed (7150/7150), but `exact: 0, rerun_required: 51,
  misses: 14, stale: true` out of 52 cache entries. **Known issue, not yet
  re-verified**: this replay cache was exported using a hand-regenerated
  manifest instead of the tune build's own `manifest_ref` -- see HI130's
  point 7 for the planned re-verification once `tune-campaign` uses the
  correct manifest automatically.

## What this is for

This is the concrete regression scenario HI130's `bigcherry tune-campaign`
orchestrator should be able to reproduce with a single invocation. Also
useful as real input for deeper analysis of the promotion/ranking
heuristics themselves (which candidates were rejected and why, whether the
tie-band/threshold settings look right against real measured margins) --
see the `ranking_decisions` field in `e2e_dualxtx_tune.promoted2.jsonl` for
every candidate's full verdict, not only the ones that won.

The first `tune-promote` pass (before correctness evidence existed) is not
included here -- it differs from the file above only in every row's
`promotion_status` being `rejected_no_correctness_evidence` instead of a
real verdict; not worth a second multi-MB file.
