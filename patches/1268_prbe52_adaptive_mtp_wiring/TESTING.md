# Testing — PRBE52 adaptive MTP wiring

## Offline

Run:
- `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --experiment adaptive-mtp`
- `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --experiment adaptive-mtp-no1210`
- `python -m unittest tools.tests.patch.test_1268_prbe52_adaptive_mtp_wiring`

Require both compositions to apply, idempotence to pass, no 1210 dependency for 1268, and the no-1210 composition to include production 1321/1322/1295.

## Runtime matrix

Same-session A/B at 8K / 24K / 98K:
1. released build;
2. `adaptive-mtp`, adaptive off/on;
3. `adaptive-mtp-no1210`, adaptive off/on;
4. for adaptive-on, `BIGCHERRY_MTP_AHEAD=1` and `0` as a compatibility/benefit split.

Keep greedy, seed, prompt, decode length, device split and draft placement fixed. Report t/s, accepted/drafted, md5, depth-change markers and shutdown depth histogram.

## Expected activation

With `BIGCHERRY_PATCH_TRACE=1` and adaptive enabled:
`BIGCHERRY_PATCH_HIT patch=1268_prbe52_adaptive_mtp_wiring path=mtp_adaptive_depth ...`

At shutdown require at least one:
`event=depth_hist depth=<d> rounds=<n>`.

## Correctness

The controller is a pure function of accepted/drafted counts and resets in `begin()`. Repeated identical requests should therefore produce identical depth histories if the target accept stream is identical. If md5 or depth history diverges first at the same verify shape, investigate backend numerical determinism; if accept diverges first after a depth change, treat it as depth-dependent verify numerics.

A requested speculative configuration that fails initialization must fail server startup; serving with `spec == nullptr` is a test failure.

## Performance hypothesis

The new policy starts at depth 3. A sustained <=60% acceptance window moves toward floor 2; >=72% moves deeper; 60-72% holds. This is intended to keep the measured long-context benefit without paying the floor-1/2 short-context cold start. Hardware A/B, not source inspection, decides whether it succeeds.
