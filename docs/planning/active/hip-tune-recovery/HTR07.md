---
id: HTR07
order: 0
plan: hip-tune-recovery
state: pending
created-at: '2026-09-06T00:37:59.206243+00:00'
breadth: ''
skill: advanced
created-by: claude-opus-5
work: M
priority: P0
---

# Adaptive recovery: bisect with the shortest vector that actually reproduces the failure

## Description

Extending the behavioural corpus to 128/512/2048 makes recovery ~21x more expensive per evaluation, and recovery ran 15 evaluations during bisection in campaign 27e45ae32ec4. The obvious economy -- bisect using only the cheap 128-token vector -- is WRONG, and dev-gpt-agent (req_bdb8d2b1) rejected it in one line: 'Do not bisect using 128 if 128 cannot observe the failure. That removes the causal signal.'

That is exactly the trap the corpus extension exists to fix. The regression that escaped was invisible at 128; bisecting against a vector with zero sensitivity to the failure would isolate nothing while appearing to work.

## Steps

1. Run the full corpus ONCE to identify which vectors fail.
2. Select the shortest vector that reproduces the failure, and use only that for bisection and alternative search.
3. Optionally minimise further: find the shortest n_predict that reliably reproduces the same FIRST acceptance divergence, and use that as the recovery oracle.
4. Validate the final candidate cache against the full immutable corpus before publishing -- non-negotiable, and it is what keeps qualification evidence unchanged by this optimisation.
5. Cache the deterministic native reference trace per vector: recovery regenerated the native side repeatedly where the trace is deterministic under temp=0 + fixed seed and could have been computed once.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

On a known failure, confirm the selected oracle vector reproduces the same first divergence as the full corpus, that bisection reaches the same isolated signature as a full-corpus bisection would, and that the published cache passed full-corpus validation.

## Effort & Risk

The risk is choosing an oracle that reproduces A failure but not THE failure, isolating the wrong signature. Anchoring selection on the same first acceptance divergence -- not merely 'this vector also fails' -- is what guards against that.

## Standards



## Acceptance Criteria



## Notes

Net effect: neither the 21x cost of running every vector at every bisection step, nor the blind cheap oracle. Final validation stays on the full immutable corpus, so evidence quality is unchanged and only the search is cheaper.

## Change Log

- 2026-09-06T00:37:59.206243+00:00 (created-by): Created by claude-opus-5
