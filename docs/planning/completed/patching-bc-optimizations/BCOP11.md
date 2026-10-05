---
id: BCOP11
order: 11
plan: patching-bc-optimizations
state: superseded
created-at: '2026-10-05T04:40:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Consolidate async host-input lifetime onto scheduler copy generations

## Description

Follow-through for QFP27/QFP24/1326. Upstream #29963 provides copy-indexed scheduler input events that overlap strongly with QFP24's proposed private pinned ring/lifetime protocol. Prove one scheduler-owned lifetime invariant and reduce QFP24 to a transport-only pinning experiment.

## Steps

1. Port/qualify only #29963 scheduler input-event semantics against 1326.
2. Stress 10,000 immediate-overwrite generations at copy depth 2/3/4 and delayed 3-child Meta fanout; zero stale-generation reads allowed.
3. Determine whether one Meta completion event is ordered after all child DMAs; if not, implement an aggregate Meta completion handle attached to the existing scheduler generation.
4. Compare pageable vs pinned backing with identical lifetime semantics.
5. Keep the smallest copy depth hiding >=95% H2D latency.
6. Add event-query API only if copy-reuse waits consume >=1% prefill wall.
7. Retire QFP24's duplicate generation/event/ring protocol if scheduler semantics pass.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

- One `copy_id + generation` lifetime protocol owns all source reuse.
- Zero stale-generation reads across stress matrix.
- Pinned backing is promoted only for >=3% PP/TTFT improvement.
- Duplicate QFP24 lifetime machinery is removed/retired.

## Notes

2026-10-05 review: superseded by owner item QFP27, which is the same consolidation (10,000-generation stress, copy depth, aggregate Meta completion, 95% gate). Not implemented; work stays open under QFP27. #29963 is not in pin 050439614.

## Related

QFP16/1326, QFP24, QFP27, MET06; llama.cpp #29963.

## Change Log

- 2026-10-05T04:37:16.193171+00:00 (updated-by): Updated: section:notes
- 2026-10-05T04:38:01.685352+00:00 (state-transition): State: pending → superseded
