---
id: PEF04
order: 0
plan: patching-external-fixes
state: pending
created-at: '2026-09-28T21:40:59.924359+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Re-snapshot tracked forks before/after llama.cpp bump to b11233 (61 sources-check findings)

## Description

`bigcherry sources check` (run 2026-09-29, ahead of a llama.cpp bump from b11126 to upstream b11233, 107 builds) surfaced 61 findings across all 3 tracked forks. None are the high-priority 'CHANGED content on an already-ported patch' signal -- all are 'not found by title' (commit squashed/renamed/reverted, ambiguous, mostly the huge stew675-rdna-boosts fork after 607 new commits), 'moved' (source snapshot staleness), or 'mainline-check-incomplete' (git cherry timed out at 600s). Not bump-blocking, but the snapshots are stale and each tracked commit's status needs a fresh manual review pass before the next round of patch planning against these forks.

## Steps

1. Re-run `bigcherry sources check --timeout <larger>` per source to get past the git-cherry timeout on stew675-rdna-boosts and joursbleu-llama-cpp.
2. For each 'not found by title' finding, manually check whether the commit was squashed/renamed (harmless) or genuinely dropped/reverted (needs a real decision) -- prioritize commits already `ported-*` in a live patch (RD04, RD08, RD12, RD13, RD17, RD19, RD20, RD21/PRBE16, RD22, RD54, PRBE20/1210) over `planned`/`excluded` ones.
3. Record a fresh snapshot (v4) for stew675-rdna-boosts once reviewed, per config/external-sources.toml's existing snapshot-versioning convention.
4. File any real content-drift finding (a ported patch whose upstream source changed) as its own separate plan item -- do not fix inline here.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes



## Change Log

- 2026-09-28T21:40:59.924359+00:00 (created-by): Created by agent
- 2026-09-28T21:41:10.539648+00:00 (updated-by): Updated: section:description, section:steps
