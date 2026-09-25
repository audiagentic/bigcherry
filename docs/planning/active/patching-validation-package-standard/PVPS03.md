---
id: PVPS03
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-25T12:03:55.053290+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Four-arm patch validation: stock / base BC / validated BC / validated BC + patch, with ongoing reference ladder

## Description

User direction 2026-09-25: every patch validation must build the promoted (validated-enhancements) set together with the candidate, and keep an ongoing stat of base llama.cpp vs base BC vs validated BC vs validated BC + patch. Today control = bigcherry-tuning, subject = control + focal; patches are never measured on top of what ships, so accumulation/clash is invisible (RD73 precedent: passed alone, regressed in production).

## Steps

1. Scaffold: control composition = baseline source + validated-enhancements (+common), excluding the focal if it is itself in the set; subject = control + focal.
2. Build a base-BC arm (baseline + common) only when its composition digest differs from control; stock arm already built.
3. Reference ladder: after the producer, paired interleaved llama-bench (tg128, pp512) across stock/base/validated/subject on the producer's model, order rotated per round; write ladder.json artifact (reference, non-gating) and append one row per session to a per-arch history.
4. summarize.py prints the ladder.
5. Tests: composition resolution (validated set included, focal excluded, dedupe), ladder artifact shape.
6. Follow-up (separate item): mandatory dual-XTX -sm tensor production no-regression lane.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline unit tests for composition + ladder; one hardware session (e.g. 1216 gfx1201) shows 4 arms (3 distinct builds while validated-enhancements is empty), ladder.json present, verdict unchanged.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

validated-enhancements is empty at 2026-09-25 (RD73 demoted), so arm 3 == arm 2 until the first promotion; serial-2 verdicts remain valid under the new definition.

## Change Log

- 2026-09-25T12:03:55.053290+00:00 (created-by): Created by agent
