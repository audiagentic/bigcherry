---
id: RSC17
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-27T03:42:00.914197+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Shared KV startup allocation: measured per-rank capacity and native contract

## Description

RSC16 stage A/B owner. Diagnose and deliver exact per-rank shared KV allocation, independent L/S/C intent and versioned admission. Follow docs/design/radiance-shared-kv-capacity-code-guide.md A-D. No scheduler redesign or model-specific exceptions.

## Steps

1. Reproduce archived failure and matched long-profile 256K/512K variants; record actual calibration, per-rank ledger, rad0029 counterfactual. 2. Refactor existing own-split into pure candidate before admission; only extend beyond current algorithm where needed. 3. Typed native capacity, sealed rank/group facts and policy/version, legacy unchanged. 4. Validate final rank budgets before allocation and post-carve geometry; typed drift/refusal. 5. Patch chain and existing VTP33 regression plus synthetic boundary/shortfall; hand off to RSC18/RSC16.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Existing config/fleet/mem tests and serving S5, measurement, receipt and cache tests; exact rounded boundaries, COW-independent capacity, unequal ranks, windows/state/streamed/fleet/unified or explicit unsupported refusal. Managed 256K/512K A/B facts; no GPU writes.

## Effort & Risk



## Standards



## Acceptance Criteria

Exact limiting rank/bytes identified; one full request or actionable refusal; maximal feasible shared geometry with measured guards; no S*L floor in shared mode; same logical rank tables, actual allocation matches sealed plan; legacy unchanged. Report unsupported combinations. Hand off to RSC18 and RSC16.

## Notes

Second review RSC16 second design review, code guide H: reuse validated rad0029/VTP33 own-split; common rad0011 refit refuses before that path. Instrument exact archived input and matched long profile, selected calibration (seed entries FlashNext only) and complete rank facts. Refactor existing solver before final admission; verify every rank before and after carve, no duplicate writer. Separate managed patch responsibilities, dependency/anchor/composition and legacy tests.

## Change Log

- 2026-11-27T03:42:00.914197+00:00 (created-by): Created by agent
- 2026-11-27T03:47:49.501797+00:00 (updated-by): Updated: section:notes, section:steps
- 2026-11-27T03:48:39.685143+00:00 (updated-by): Updated: section:notes
