---
id: VTP42
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-15T13:01:55.739115+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Swappable prefill governor: token budget with reservation, backfill within a delay bound, card-free simulator

## Description

Owner proposal (2026-11-15/16). Replace fixed prefill admission numbers (VTP39 count cap, VTP41 threshold) with policies behind one interface. (a) Token budget: prefill tokens in flight (remaining, not whole prompts) limited to a multiple of the largest request (owner: 1.5x max request), capped by what free KV can hold, so a max-size request always fits and fp8/bf16 caches differ automatically. (b) Reservation: a request waiting past its max wait stops further admissions ahead of it until it fits. (c) Backfill: smaller later requests may go ahead of a waiting large one while the delay they add to it (tokens / measured prefill rate; steps are shared, so it is never zero) stays within a bound. Policies selected by a manifest field with parameters; engine stays generic.

## Steps

1. Governor interface in the scheduler (state in: running prefills with remaining tokens, waiting queue with sizes/ages/priorities, free budget and KV; out: admission decisions and walk order). Move arrival and VTP41 remaining logic behind it; randomised test that step plans are unchanged.
2. Card-free simulator: drive the scheduler with recorded or synthetic request mixes and a measured cost model (TTFT(N) = aN + bN^2, per-step times); calibrate against the VTP41 mixed A/B.
3. `budget` governor (multiple of max request, KV cap, reservation on age); simulate.
4. `backfill` governor (delay bound); simulate; tune.
5. Hardware A/B of the finalists against the VTP41 -long defaults on the mixed workload; owner picks defaults.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

sched_test (identity of plans for existing policies, governor unit cases), simulator calibration error against hardware, mixed-workload A/B per request size, preemptions, decode beside the mix.

## Effort & Risk

L. Interface refactor of admission/walk order is the risky part (identity test guards it); simulator fidelity limits how far tuning offline can go.

## Standards



## Acceptance Criteria

- [ ] Existing policies reproduce their step plans exactly behind the interface.
- [ ] Simulator predicts per-size TTFT of the VTP41 A/B within 15%.
- [ ] A budget or backfill governor beats the VTP41 defaults on mid-size waits with no worse 100k mean, or the owner keeps VTP41 defaults with the comparison recorded.
- [ ] No request waits past its reservation bound (test).

## Notes

## Change Log

- 2026-11-15T13:01:55.739115+00:00 (created-by): Created by agent
