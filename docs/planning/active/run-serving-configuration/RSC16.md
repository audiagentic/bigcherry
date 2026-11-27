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

1. RSC17 owns A/B: instrument failure, typed native intent, exact allocation and sealed policy. 2. RSC18 owns C: live accounting, scheduler pressure and metrics; depends on RSC17. 3. RSC16 integrates D/E: overlapping mixed-length GPU and cross-family/tier/fleet tests, release decision. Reuse ENG04/ENG03/MEN17 and RSC12; no duplicate implementation.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

RSC17 and RSC18 card-free gates required. RSC16 owns managed installed GPU proof, actual per-rank/group capacity versus observed pages and allocations, mixed resident lengths, accuracy and performance, legacy and unsupported-path regressions. Owner-approved thresholds before promotion.

## Effort & Risk



## Standards



## Acceptance Criteria

No S*L backing demand in shared mode; independently reported L/S/C; measured per-rank and tier capacity; at least one L request or actionable refusal; mixed-length requests bounded by actual pages and slots, no deadlocks or lost live state; legacy parity and supported-family matrix before rollout.

## Notes

Motivation: 2020.23.1 IQ8R TP3, yarn:2, bf16_i8, 524288, 16/8/4 slots and batch512 refused RAD_E_NOMEM at refit; no inference. RSC12 owns variants, ENG04 owns format/quality, MEN17 release acceptance. No GPU or code changes in planning stage.

Code-level review RV4697: reuse existing fresh-prefill/recompute guards and KVManager; measure rectangular BatchBuilder overhead before changing ABI; replace fixed-inclusive kv_one/L assumptions in shared mode; rank feasibility must use same-rank costs. See docs/design/radiance-shared-kv-capacity-code-guide.md. No implementation yet.

RV4698 second review: rad0029 already solves a subset of per-rank capacity; rad0011 common gate runs earlier. Revised design reuses it, avoids new scheduler where not needed, corrects first-test profile mismatch and demands actual calibration and physical ledger. Scope first shared-mode qualification to supported dense full-paged discrete paths, extend capability-driven across models/tiers only with gates. See guide H. No code or GPU changes.

## Change Log

- 2026-11-27T03:33:31.874138+00:00 (created-by): Created by agent

## Reviews

- RV4697
- 2026-11-27T03:42:47.951296+00:00 (updated-by): Updated: section:steps, section:notes, section:validation
- RV4698
- 2026-11-27T03:48:04.634120+00:00 (updated-by): Updated: section:notes
