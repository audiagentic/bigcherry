---
id: BCOP221
order: 221
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-15T01:08:55Z'
created-by: agent
priority: P1
work: S
---

# RNX11: fence default-off CPU-root large graph replay; retire duplicate fixed-grid patch

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

2026-11-22 plan consolidation: BCOP221 is a secondary decision gate already recorded with higher specificity in RNX11 (docs/planning/active/patching-r9700-cross-rdna/RNX11.md). RNX11 explicitly incorporates BCOP221's 1291 CPU-root large-path capture/replay and stale-generation risk, default-off status, and provider-specific refusal tests. No second 1311 fixed-grid implementation is warranted. No independent code task remains here. Review RV4317 records ownership; this is supersession, not an assertion that RNX11 is completed.

2026-11-22 consolidation: Secondary decision note superseded by active primary RNX11 in patching-r9700-cross-rdna. RNX11 already records BCOP221's 1291 large CPU-root generation/replay correctness concern, default-off state and provider-specific refusal proof. No separate fixed-grid patch 1311. Review RV4317 records the evidence. No completed runtime work is claimed; RNX11's pending acceptance remains unchanged.

## Reviews

- RV4317

## Change Log

- 2026-11-22T03:29:59.009897+00:00 (updated-by): Updated: section:notes
- 2026-11-22T03:30:47.171091+00:00 (updated-by): Updated: section:notes
- 2026-11-22T03:31:02.243992+00:00 (state-transition): State: pending → superseded
