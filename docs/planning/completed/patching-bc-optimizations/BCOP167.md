---
id: BCOP167
order: 167
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-12T17:04:14+00:00'
created-by: agent
priority: P1
work: S
---

# QFP23: make 0910 runtime profile parity and auto failure deterministic

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

2026-11-22 consolidation review RV4338: secondary decision note now superseded by QFP23 (patching-qwen-flash-next). Existing QFP23 already contains the BCOP167 deterministic 0910 runtime auto-profile admission analysis: unsorted profile order, topology-attested auto selection and flag-read timing remain open; no duplicate profile loader. QFP23 remains active; no new runtime implementation or acceptance asserted.

## Reviews

- RV4338

## Change Log

- 2026-11-22T05:03:27.136042+00:00 (updated-by): Updated: section:notes
- 2026-11-22T05:03:37.436283+00:00 (state-transition): State: pending → superseded
