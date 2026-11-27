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

Instrument VramFacts, KV group geometry, refit inputs and rank receipts; reproduce refusal. Implement typed native option and checked per-rank feasibility, preserve legacy. Seal measured intent and capacity, update calibration/cache/replay, verify actual allocation. Qualify with card-free and installed measured plan.

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

## Change Log

- 2026-11-27T03:42:00.914197+00:00 (created-by): Created by agent
