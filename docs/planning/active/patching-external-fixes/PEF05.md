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

**Transferred PA50 independent acceptance — b11552 and inherited PEF03:** The one canonical `bigcherry sources check` and `engines/llamacpp/external-sources.toml` snapshot/audit owner is PEF05. Preserve PA50's b11474→b11552 exact *63-finding* result from `artifacts/pin-bump/sources-check-b11552.log` (host-local/ignored, not an immutable published pass), distinct from PEF05's earlier c061df198 63-finding and PEF04's b11233 61-finding cohorts. Preserve the inherited PEF03 b11126 evidence: mrlordcat RD69/71/72/73/76, nasone NRO06, AMD-ecosystem RD39-44/RD49/RD50/RD30/RD97/98 and untracked commits, stew675 approximately 33 apparently drifted tracked ports with a history-rewrite/network-reset possibility, joursbleu rebased/RD54, davetha incomplete, and the unchanged upstream/charlie control. For b11552, mrlordcat has 147 new commits and timed-out `git cherry`; nasone 49; AMD-ecosystem 607 with timed-out cherry; stew675 tip lacks ancestry; joursbleu rebased; davetha cherry timed out; upstream 319 expected. These are historical *unresolved statuses*, not evidence that the corresponding production patches are removed or merged upstream. Re-run each still-relevant fork on an exact current pinned checkout/snapshot SHA with bounded higher timeouts or an equivalent checked patch-id/content comparison, and classify every tracked commit as identical-through-rebase, squashed/renamed, upstream-absorbed, genuinely changed/reverted, or UNKNOWN (including network/timeout), recording immutable source and comparison receipts per cohort. Prioritize production recipe-selected RD04/RD05–08/RD13–22/RD27/RD30/RD39–44/RD49/RD50/RD54/RD69–76/RD97/98 and PRBE05/06/10/11/13/16/18/19/20 only where still present. Separately skim recent new commits for candidate ports and hand them to their real patch owners, never install a patch or make another sources-check system in this plan. Refresh `external-sources.toml` snapshots only against demonstrated content/provenance; no 'not found by title' or longer timeout alone establishes absorption. Require no unexplained *current-pin* findings, with every historical b11126 / b11233 / c061df198 / b11552 result either evidence-backed disposition or explicitly UNKNOWN and linked, and keep PA49's unrelated pin-bump automation, timeout/streaming phase and diagnostic build workflow separate. No validation or current-source adoption is implied by this transfer.

## Notes

CONSOLIDATION 2026-11-22 LAB-AND-SOURCES: PEF04 has one canonical owner: PEF05. Both tickets own the same bigcherry sources check manual fork-snapshot refresh and changed/upstream-status triage for consecutive llama.cpp bump checkpoints; neither introduces a separate source-check subsystem. This is planning consolidation only, not a completed acceptance or release. Historical evidence and current-pin/real-run stop rules remain binding.

CONSOLIDATION 2026-11-22 SOURCE-TRIAGE: PEF05 now owns PA50 b11552 manual fork-snapshot triage and inherited PEF03 b11126 findings, alongside the original PEF04 b11233 and PEF05 c061df198 cohorts. PA49 remains the sole pin-bump automation owner; none of the current-pin provenance, content or timeout outcomes have been requalified by consolidation.

## Change Log

- 2026-10-02T01:11:33.440268+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261002_011157_llamacpp-updated-to-include-q_5054
- 2026-10-02T01:12:03.118991+00:00 (updated-by): Updated: section:ledger-events

## Reviews

- RV4541
- 2026-11-22T11:26:32.000953+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- RV4552
- 2026-11-22T11:58:53.750987+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
