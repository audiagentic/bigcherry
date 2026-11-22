---
id: BCOP238
order: 238
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-15T18:11:29Z'
created-by: agent
priority: P2
work: S
---

# PGC12: retire unproven phase-aware AllReduce; transfer receipt ownership to PGC09

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

2026-11-22 consolidation review RV4335: secondary decision note now superseded by PGC09 (patching-gpu-collectives). Existing PGC09 contains BCOP238 section: PGC12 phase-aware dispatcher is terminal given historical exact-F32 96KiB vs BF16 mismatch. PGC09 owns sole actual provider/threshold/conditional completion receipt; no second runtime policy. PGC09 remains active; no new runtime implementation or acceptance asserted.

## Reviews

- RV4335

## Change Log

- 2026-11-22T05:02:28.422075+00:00 (updated-by): Updated: section:notes
- 2026-11-22T05:02:38.659086+00:00 (state-transition): State: pending → superseded
