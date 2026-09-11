---
id: BRBC03
order: 0
plan: build-reusable-build-campaign
state: pending
created-at: '2026-09-11T04:38:44.621798+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P3
---

# Toolchest publication integration (approach TBD)

## Description

BigCherry needs to publish verified llama.cpp builds to Toolchest (the external audiagentic/bigcherry-toolchest service, which is currently live and implements POST /api/builds?external=1 with automatic BigCherry handoff per its own docs). How and what to integrate is not yet determined -- this item exists to track the need, not to prescribe an approach.

A prior attempt existed on a now-deleted branch (`toolchest-publish`, commits 44ec6810/70f6ea2a/71ff1633) adding tools/bigcherry/integrations/toolchest.py, changes to campaign/planner.py and cli/build.py, a GitHub Actions workflow, and docs/reference/build/TOOLCHEST.md. Per a GPT deep-review (2026-09-11) it was not mergeable as-is: 569 commits behind main, and both production files it touched (campaign/planner.py, cli/build.py) have materially evolved since (experiment-aware/contract-aware lane identity, pin-state preflight, tree_activity.Lease, build advisories) in ways a naive merge/rebase would silently break. The branch was discarded per explicit user direction rather than forward-ported, since the real integration approach is still undecided.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria

Not yet defined -- pending a decision on the actual integration approach. This item should be scoped for real once that decision is made, rather than inheriting the discarded branch's design wholesale.

## Notes

Do not resurrect or rebase the deleted toolchest-publish branch -- design the integration fresh once the approach is actually decided. If useful as a reference for what a prior attempt looked like, the old branch's diff is recoverable from git history/reflog for a while, but it is not a starting point to build on.

GPT's specific technical notes from reviewing the old attempt, worth keeping in mind whenever this is designed for real: publication should consume the authoritative runtime_bundle_ref rather than treating binary_ref alone as sufficient runtime identity; a CI workflow for this should not push to a feature branch (the old one incorrectly pushed to toolchest-publish itself instead of main/canonical CI).

## Change Log

- 2026-09-11T04:38:44.621798+00:00 (created-by): Created by agent
- 2026-09-11T04:38:51.393239+00:00 (updated-by): Updated: priority='P3', section:description, section:acceptance_criteria, section:notes
