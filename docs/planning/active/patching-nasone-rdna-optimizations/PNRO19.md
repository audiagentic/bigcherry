---
id: PNRO19
order: 0
plan: patching-nasone-rdna-optimizations
state: done
created-at: '2026-09-27T12:55:18.984027+00:00'
breadth: ''
skill: ''
created-by: agent
priority: P2
work: S
---

# 1270 (PNRO14) patch mechanics test fails: anchor matched 3x / 0x against vendor source

## Description

**Resolved by independent commit `f4228540b774afee0efb2f8e31e760b0a3c65f67` (2026-09-27).** The original two patch-1270 mechanics failures were patch-authoring errors, not confirmed upstream drift: `pnro14-includes` used `max_span_lines=2` for a three-line match; `pnro14-device-select` matched comments even though the patcher strips comments before matching. Current `patch.py` uses `max_span_lines=3` and `_csource.strip_noise(_DEVICE_OLD, "c")`. No additional patch is needed for this issue. PNRO14 remains authoritative for the independent gfx1151 performance/architecture qualification decision.

## Steps

1. Terminal: inspect `f4228540b7` and current `patch.py` (completed).
2. No separate test queue or implementation owner. PNRO14 retains future patch-lint/rebase, route attribution and hardware gates; do not revive PNRO19 without a new reproducible regression.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

2026-10-09 source review confirmed both corrections in current patch 1270 and verified unique include/host/device selector anchors against b11474. 11/11 host-only assertions passed as part of PNRO14 audit; the actual package pytest and patch-rebase-check were **not run** in this invocation. Independent fix commit states the original two failures were repaired.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Terminal reconciliation: BCOP74 records the closure; PNRO14 owns the remaining hardware-gated experiment. Do not duplicate the work.

test_missing_host_selector_fails_closed (the other test in the same file) still passes -- only test_apply_and_idempotent is affected.

## Change Log

- 2026-09-27T12:55:18.984027+00:00 (created-by): Created by agent

## Ledger-events
