---
id: PVPS05
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-25T12:52:33.805104+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Production dual-GPU MTP no-regression lane for every patch

## Description

RD73 passed single-GPU and regressed -2% on the production dual RX 7900 XTX -sm tensor MTP server. --production-lane runs validated BC vs validated BC + patch on that shape (fresh llama-server per request, paired/interleaved, mtp_wall_tps), records draft acceptance per arm, verdict pass when ci95_low >= -1%.

## Steps

1. [done] tools/bigcherry/patch/campaign/production_lane.py + --production-lane flag + producer hook + summarize line + tests.
2. Set BIGCHERRY_PRODUCTION_MODEL / BIGCHERRY_PRODUCTION_DEVICES on the build server.
3. Add --production-lane to every queued job; first hardware run on a known-neutral patch (1216) to measure lane noise and duration.
4. Add a pp4096 -sm tensor paired lane (RD73 also regressed prefill).
5. Make the lane verdict a promotion prerequisite in the lifecycle skill/gate once its noise floor is measured.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline tests (tools/tests/patch/test_production_lane.py); hardware: 1216 with --production-lane shows PASS with acceptance equal across arms.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Pending GPT review of metric/round counts (gateway failing 2026-09-25).

## Change Log

- 2026-09-25T12:52:33.805104+00:00 (created-by): Created by agent
