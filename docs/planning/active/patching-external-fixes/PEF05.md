---
id: PEF05
order: 0
plan: patching-external-fixes
state: pending
created-at: '2026-10-02T01:11:33.440268+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: M
---

# Triage sources check findings at pin c061df198 (63 findings: fork drift, mainline-check timeouts)

## Description

bigcherry sources check run for the b11233 -> c061df198 bump (2026-10-02) reported 63 findings; full output in artifacts/pin-bump/sources-check-c061df198.txt. Classes: one tracked fork rebased (active head no longer an ancestor of tip -> new snapshot + re-audit), four forks moved (49/112/147/607 new commits since review), ~20+ tracked commits 'not found by title' (RD04, RD30, RD49, RD69, PRBE10, PRBE20, ...), and 3 mainline-check-incomplete (git cherry timed out at 600 s, merged-upstream status unknown). None of the 16 production (bigcherry source) patches failed rebase at c061df198, so nothing blocks the pin.

## Steps

1. Rerun the three mainline checks with a larger --timeout to settle merged-upstream status.
2. For the rebased fork: take a new snapshot, compare patch-ids of tracked commits, re-audit.
3. For each 'not found by title' commit: find what happened by hand (squash/rename/revert/drop); update external-sources.toml tracked entries.
4. For moved forks: scan new commits for anything relevant to ported patches (RD30 is production patch 1237).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

sources check re-run with no unexplained findings; external-sources.toml snapshots refreshed.

## Effort & Risk



## Standards



## Acceptance Criteria

**Transferred PEF04 acceptance (independent gate):** Preserve PEF04's 2026-09-29 b11126->b11233 evidence cohort of 61 findings alongside PEF05's later b11233->c061df198 63-finding cohort. Re-run the one maintained bigcherry sources check per currently tracked fork with a bounded, explicitly larger per-source mainline git-cherry timeout and record the exact current source/bump refs and independent result receipts. For each not-found-by-title / moved / rebased / mainline-check-incomplete case, match tracked commits using immutable SHAs/patch ids and manual squash/rename/drop/revert review, prioritising ported production owners RD04/RD08/RD12/RD13/RD17/RD19/RD20/RD21/RD30/RD49/RD54/PRBE16/PRBE20 and distinguishing genuine content drift from harmless rebase or upstream inclusion. Refresh external-sources.toml tracked entries, versioned snapshots (including stew675 v4 historical gap), reviewer disposition and per-fork current state; no blanket success or taking a larger timeout as proof. If a currently ported patch's effective source content changed, file/refer it to the real owning patch plan and stop no-op acceptance; do not fix patch code or invent a second source snapshot workflow here. Acceptance requires no unexplained findings on the current pin, with historical 61/63 inventories and their residual unknown statuses preserved and linked, rather than falsely treating b11233 or c061df198 as the latest source.

## Notes

CONSOLIDATION 2026-11-22 LAB-AND-SOURCES: PEF04 has one canonical owner: PEF05. Both tickets own the same bigcherry sources check manual fork-snapshot refresh and changed/upstream-status triage for consecutive llama.cpp bump checkpoints; neither introduces a separate source-check subsystem. This is planning consolidation only, not a completed acceptance or release. Historical evidence and current-pin/real-run stop rules remain binding.

## Change Log

- 2026-10-02T01:11:33.440268+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261002_011157_llamacpp-updated-to-include-q_5054
- 2026-10-02T01:12:03.118991+00:00 (updated-by): Updated: section:ledger-events

## Reviews

- RV4541
- 2026-11-22T11:26:32.000953+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
