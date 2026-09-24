---
id: PRBE23
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:00.176506+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-MOE-001: MoE-aware MMQ tile sizing from average expert occupancy

## Description

IMPLEMENTED-AS-PATCH (umbrella). Patch 1237_rd30_moe_mmq_compact_grid (state=untested) already implements the compact actual-work grid concept (block_start[]/block_expert[] scheme, gated gfx1100, fail-closed to legacy rectangular grid). Real informal evidence: hostile-routing correctness passes (uniform/all-one/skew/Zipf/tiny/n_expert=256), interleaved dual-gfx1100 A/B on Qwen3.6-35B-A3B shows +0.73%..+0.90% pp512. STATE stays untested pending the project's own HI83-governed tools/bigcherry/patch_validation_campaign.py run, never yet executed. PRBE23 is the umbrella qualification item; PRBE24 and PRBE25 (below) are its two sub-scoped validation items (map-construction correctness/overhead, and launch-grid downstream benefit respectively) since patch 1237 implements both together.

## Steps

1. Confirm patch 1237's current STATE is still untested and no new hardware evidence has landed since 2026-09-12 (check patch.toml + SUMMARY.md).
2. Run the full tools/bigcherry/patch_validation_campaign.py record->tune->promote->export->replay->bench->report pipeline for patch 1237 (not run yet, per its own SUMMARY.md) -- this is the umbrella action that formally supersedes the informal evidence already gathered.
3. Treat mean/expected-occupancy selection as an EXPLANATORY control only inside this campaign, never a standalone promoted candidate -- the compact block-map (PRBE24/PRBE25 scope) is the preferred treatment arm.
4. Preserve exact expert/tile enumeration and legacy fallback identity throughout; do not let campaign tooling silently drop the fallback arm.
5. Re-run EC13/RD94 hostile routing (uniform, Zipf/skew, concentrated, single-hot, captured natural routing at n_expert=256) as part of the formal campaign, even though informal versions already passed, since campaign-recorded evidence is what gates promotion, not prose.
6. Any prior candidate-tuning evidence for MMQ MoE that predates patch 1237's grid change must be remeasured (grid-compaction changes real per-launch cost, per patch.py's own validation-consequence note).

## Detailed Solution & Technical Design

PRBE23 does not need new source-code design -- patch 1237 already exists and is code-complete per its SUMMARY.md. The only remaining work is running the project's own formal validation campaign (never yet executed) and folding in PRBE24/PRBE25's narrower sub-scope evidence (see those items) as part of the same promotion decision. No GPT design consultation was needed for this item since there is no new code to design; effort here is qualification/process, not implementation.

## Code Samples & Guidance

N/A -- no new source edits. Reference: patches/1237_rd30_moe_mmq_compact_grid/patch.py (existing, already implements mmq_build_moe_block_map + compact launch); patches/1237_rd30_moe_mmq_compact_grid/patch.toml (state=untested, plan-item=RD30).

## Files

patches/1237_rd30_moe_mmq_compact_grid/ (existing, unchanged); tools/bigcherry/patch_validation_campaign.py (the campaign runner to invoke); patches/1237.../evidence/ (new formal campaign output).

## Validation

Full tools/bigcherry/patch_validation_campaign.py run on Brutus (not run here): record->tune->promote->export->replay->bench->report, dual-gfx1100, Qwen3.6-35B-A3B, hostile routing matrix, native/tune correctness, non-target architecture controls. Only after this formally-recorded run may patch 1237's STATE move from untested toward validated.

## Effort & Risk

S (qualification/process only, no new code) but gated on real hardware time on Brutus which is outside this batch's scope.

## Standards

Redesign-first; exact enumeration; fail-closed overflow; causal identity; preserve negative/noise evidence.

## Acceptance Criteria

Compact-grid candidate has durable identity and passes correctness/hostile routing; real E2E gain is reproduced with controlled A/B; no broad default or mean-based selector is promoted without its own evidence.

## Notes

Supersedes: RD30
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd30

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH (1237, untested). No GPT design request needed (no new code -- qualification-only item); PRBE24/PRBE25 sub-scope GPT design was likewise not needed for the same reason. Umbrella item depends on PRBE24 and PRBE25's narrower validation passing first.

## Change Log

- 2026-09-09T10:55:00.176506+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:15.436526+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.231064+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.956316+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:53:23.331205+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025409_moe-mmq-successors-prbe2325-n_6205
- 2026-09-10T02:54:09.740115+00:00 (updated-by): Updated: section:ledger-events
- chg_20260911_220609_documented-the-most-thoroughly_1414
- 2026-09-11T22:06:09.771118+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:31:29.818080+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
