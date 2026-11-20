---
id: PVPS07
order: 0
plan: patching-validation-package-standard
state: superseded
created-at: '2026-09-25T23:16:36.887890+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Production lane: adaptive rounds, pp4096 sentinel, acceptance bound, scoped gating

## Description

GPT review req_4d131e8b7c1c452d on PVPS05: 8 fixed rounds is reasonable only if the CI is decisive; add rounds when inconclusive (pre-declared max, no optional stopping beyond it). Add a pp4096 -sm tensor paired sentinel for prefill-affecting patches. Give MTP draft acceptance an explicit equivalence bound. Gate promotion on the lane only for patches touching TP/all-reduce/device placement/speculative decode (declared in patch.toml subsystems); otherwise record as secondary.

## Steps

1. Record CI width; extend to a declared max_rounds when ci95 straddles -1%.
2. pp4096 paired llama-bench -sm tensor lane.
3. acceptance bound (e.g. |delta| <= 0.02 absolute) as a lane check.
4. Lifecycle gate hook keyed on subsystems.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Unit tests for the round extension rule and acceptance bound; hardware run on 1216 standalone.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

2026-11-20 consolidation: requirements transferred to active PVPS01 (patch qualification matrix), including this item's specific validation and acceptance gates. Source history is retained. The target remains pending; no implementation, hardware validation, or promotion acceptance is implied. See PVPS01 inherited qualification gates.

## Change Log

- 2026-09-25T23:16:36.887890+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260925_232233_correctness-checks-for-approxi_1217
- 2026-09-25T23:22:39.457794+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-20T04:11:07.178971+00:00 (updated-by): Updated: section:notes
- 2026-11-20T04:11:19.952002+00:00 (state-transition): State: pending → superseded
