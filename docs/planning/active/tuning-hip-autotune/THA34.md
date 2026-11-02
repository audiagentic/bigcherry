---
id: THA34
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-11-02T20:46:46.291825+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# HI17 metadata-state migration test fails on main: blas-metadata-state edit finds no bigcherry_effective_call_api shape

## Description

tools/tests/tuning/test_hi17_blas_runtime_seam.py::test_metadata_state_migrations_handle_old_malformed_and_repeat_shapes fails on clean origin/main (925950d70), independent of any open PR. Its dry-run apply of the 0200_dispatch_hook patch's blas-metadata-state edit reports that the release does not have the shape the edit handles ('bigcherry_effective_call_api'), so either the pinned release's text moved, or the test's synthetic view no longer matches the edit's anchors. main must be green on tools/tests apart from hardware-dependent tests.

## Steps

1. Run the test on origin/main and read which view text the edit is applied to.
2. Decide whether the patch's anchor or the test's synthetic input is stale, and fix that one (a patch anchor change follows the patch-lifecycle rules: re-run apply and idempotence).
3. Confirm the rest of tools/tests/tuning stays green.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/llamacpp/patches/0200_dispatch_hook/patch.py; tools/tests/tuning/test_hi17_blas_runtime_seam.py

## Validation

PYTHONPATH=tools python3 -m pytest tools/tests/tuning/test_hi17_blas_runtime_seam.py passes; python3 -m bigcherry patch-lint reports no new warnings; the full tools/tests run is green apart from hardware-dependent tests.

## Effort & Risk



## Standards



## Acceptance Criteria

The test passes on main and the full tools/tests run is green apart from hardware-dependent tests.

## Notes

## Change Log

- 2026-11-02T20:46:46.291825+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_204658_burst-trace-activation-evidenc_4902
- 2026-11-02T20:47:03.346658+00:00 (updated-by): Updated: section:ledger-events
