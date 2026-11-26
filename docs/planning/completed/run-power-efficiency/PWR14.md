---
id: PWR14
order: 0
plan: run-power-efficiency
state: superseded
created-at: '2026-11-26T05:40:49.412413+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# Progressive GPU qualification: slim recipes and bounded execution

## Description

Extend PWR13 existing platform planner with sanity/slim tiers; preserve strict evidence and distinguish diagnostic screen from native and production. Do not edit active owner recovery plans.

## Steps

1. Resolve short capability-aware recipes reusing existing case groups, with explicit workload budget. 2. Execute strict recipe under one lease, harvest each group, fail fast and aggregate missing groups. 3. Measure wall time and verified work on all cards, compare with screen/extended budgets. 4. Investigate persistent HIP context and allocation reuse only with isolation, reset and oracle proof; integrate matched candidate references and escalation with PWR12/owner recovery and RCS27/28 native plans.

## Detailed Solution & Technical Design

Use existing TOML recipe groups with validated duration, pause and memory-fraction overrides; no hot overrides. Managed plan binds source digests, target snapshot and options. Single-lease strict --all rechecks each group, harvests terminal captures, stops on first nonpass, aggregates existing report with recipe and nominal duration. Future: profile repeated HIP startup vs process isolation; reuse allocations only with complete initialization, per-case oracle/guards, independent deadlines, explicit teardown and identical receipt coverage. Candidate automation depends on owner recovery, per-device exact matched references and RCS27/28 native route evidence.

## Code Samples & Guidance



## Files

config/gpu-ai-qualification.toml, tools/bigcherry/hardware/ai_qualification.py, tools/bigcherry/cli/ai_qualification_cmds.py, tools/tests/hardware/test_ai_qualification.py, existing documentation.

## Validation

Offline resolver/receipt/fail-fast tests; leased observe-only physical cohort, exact coverage and unchanged controls; measure end-to-end runtime.

## Effort & Risk



## Standards



## Acceptance Criteria

Strict slim pass is recipe-bound only; unsupported/missing cases explicit; no weakened hot/precision/restore gates; no unattended writes or promotion. Persistent probe and candidate escalation remain open until separately proven.

## Notes

2026-11-26: first slice implemented, 261 hardware tests pass/1 skipped. One sanity and four slim runs passed 56 cases (449792 GEMMs across five runs); 16 group captures harvested, unchanged controls and identities, no transaction. Sanity 3.554 s wall; slim gfx1030 15.270 s, gfx1100 19.448/19.259 s, gfx1201 21.847 s including planning/harvest/report excluding lease waits. Managed report IDs and detailed counts in AI_POWER_SCREEN.md. Driver health incomplete; no native execution or profile mutation. Current target achieved without persistent HIP; keep that optimization evidence-led. Matched candidate sweep and escalation remain blocked on power-owner recovery and native/production integration; PWR14 remains in progress.

2026-11-27 consolidation (superseded): Existing progressive slim implementation retained; transfer bounded recipe acceptance to PWR12. Persistent probe and candidate escalation only for measured coverage/cost gap; no weakened gates. All earlier results remain historical; see power-session-closeout inventory.

## Acceptance

Strict slim pass is recipe-bound only; unsupported/missing cases explicit; no weakened hot/precision/restore gates; no unattended writes or promotion. Persistent probe and candidate escalation remain open until separately proven.

## Change Log

- 2026-11-26T05:40:49.412413+00:00 (created-by): Created by agent
- 2026-11-26T05:40:54.168002+00:00 (state-transition): State: pending → in_progress
- 2026-11-26T05:44:38.034429+00:00 (updated-by): Updated: section:acceptance_criteria, section:detailed_solution
- 2026-11-26T05:44:48.978429+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4688
- 2026-11-26T23:16:27.741520+00:00 (updated-by): Updated: section:notes
- 2026-11-26T23:18:38.062485+00:00 (state-transition): State: in_progress → superseded
