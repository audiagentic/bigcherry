---
id: BCOP313
order: 313
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-19'
created-by: agent
priority: P2
work: S
---

# RNX02: vector-FA KQ WAR fence is non-HIP; retire redundant 1300 selector

## Description



## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

2026-11-22 consolidation review RV4337: secondary decision note now superseded by RNX02 (patching-r9700-cross-rdna). Existing RNX02 includes detailed BCOP313: NVIDIA upstream KQ WAR syncwarp is a HIP no-op, source proof does not establish HIP race. No patch/queue absent AMD reproduction; RNX02 retains independent existing Q8/D256 path. RNX02 remains active; no new runtime implementation or acceptance asserted.

## Reviews

- RV4337

## Change Log

- 2026-11-22T05:03:07.564396+00:00 (updated-by): Updated: section:notes
- 2026-11-22T05:03:17.696825+00:00 (state-transition): State: pending → superseded
