---
id: RSC16
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-27T03:33:31.874138+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Generic shared KV capacity independent of request context and concurrency

## Description

Deliver model-neutral shared paged cache planning and admission for independent request length, session slots and total capacity. Client harness owns individual limits. Design: docs/design/radiance-shared-kv-capacity.md. Do not assume S*L resident. Preserve legacy behavior and physical safety.

## Steps

A. Instrument per-rank ledger and reproduce 512K IQ8R BF16_I8 refusal. B. Native capacity intent and exact per-rank common-token solver; version calibration and replay. C. Runtime allocation, fairness and existing metrics. D. Mixed-length GPU proof with actual token counts. E. Cross-family/tier/fleet regressions and release acceptance; use ENG04/ENG03/MEN17.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Card-free arithmetic, schema, replay, tiny-model concurrency, rollback and fairness gates; managed GPU captures under lease. Compare existing baselines with explicit thresholds. Separate stock/efficient. See design.

## Effort & Risk



## Standards



## Acceptance Criteria

No S*L backing demand in shared mode; independently reported L/S/C; measured per-rank and tier capacity; at least one L request or actionable refusal; mixed-length requests bounded by actual pages and slots, no deadlocks or lost live state; legacy parity and supported-family matrix before rollout.

## Notes

Motivation: 2020.23.1 IQ8R TP3, yarn:2, bf16_i8, 524288, 16/8/4 slots and batch512 refused RAD_E_NOMEM at refit; no inference. RSC12 owns variants, ENG04 owns format/quality, MEN17 release acceptance. No GPU or code changes in planning stage.

## Change Log

- 2026-11-27T03:33:31.874138+00:00 (created-by): Created by agent
