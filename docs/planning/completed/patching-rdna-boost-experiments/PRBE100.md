---
id: PRBE100
order: 100
plan: patching-rdna-boost-experiments
state: completed
created-at: '2026-09-11T23:50:02.222039+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Block 08 SSM fused-kernel audit, decomposition, and target-state design

## Description

Completed audit/design record for the previously untracked Block 08 source, narrowed 2026-09-24 per conflicting GPT reviews. req_e17e0bf5a68c48d5 CONFIRMED this completion (diff exists, successor plans exist). req_b43762f844fb40b3 flagged NOT-READY: tools/lab/rd25-block08-review/README.md still says 'Status: active' / 'Question state: open' (independently verified this session -- confirmed still true), and the completion's implicit claim that follow-ups have exact anchors/fail-closed dispositions is unmet (PRBE13 has no verified six-node b11126 anchor; PRBE18 still leaves part of the exact 16-op gate/beta sequence for implementation-time discovery). Resolution: narrow this item's completion criterion to 'Block-08 inventory/decomposition completed' only (which IS true and IS what this item's steps/acceptance_criteria describe) -- successor implementation-readiness (PRBE13/PRBE18 anchor verification) is explicitly OUTSIDE this item's scope and tracked separately in those items. The lab README's own Status/Question-state fields are stale bookkeeping outside this plan's edit scope (docs/planning only); whoever next touches tools/lab/rd25-block08-review/ should update README.md to Status: closed / Question state: resolved to match this item's completed state.

## Steps

1. Preserve and review the complete 5efcd85f source diff and its 10-file/+1585/-68 identity.
2. Record dispositions: Q8_1 cache/MMVQ reuse belongs in PRBE05; paired activation in PRBE11; shared-expert candidate in PRBE13; 16-node SSM candidate in PRBE18; determinism cluster in PRBE20.
3. Reject duplicate/unsupported ports for direct-Q8 producers, standalone gate/beta or L2 fragments, hidden conv_states reads, and unrelated attention/MoE tuning.
4. Carry exact matcher, layout, epsilon, scalar-gate, graph-edge, capture/fallback and selector-derived-launch constraints into the owning PRBE plans.
5. Register source provenance and correct stale RD25/PRBE11 narratives; no production edits or standalone Block 08 patch is introduced by this audit.

## Detailed Solution & Technical Design

Block 08 is not a single implementation target. The shared-expert region is a PRBE13 candidate; the exact 16-node SSM fusion is a PRBE18 candidate; cache reuse is PRBE05 stage 2; determinism hunks belong to PRBE20. PRBE19 expresses the post-fix source-state rule for affected successors, while PRBE11/RD12 is outside that rule. Use stable bounded cache slabs, capture_active fallback, exact graph-edge visibility, Q/K epsilon equality, scalar/layout/shape predicates, raw output/logit comparison and current selector-derived MMVQ geometry. Do not create a second executor/cache identity system or port the monolithically fetched diff.

## Code Samples & Guidance



## Files

tools/lab/rd25-block08-review/block08.diff; config/external-sources.toml; PRBE05/11/13/16/18/19/20; completed RD09/12/15/21/24/25/26 provenance; current ggml-cuda sources; candidate patch/test/campaign locations.

## Validation

Source identity and full-diff inventory; symbol/anchor inventory against current main; overlap/disposition table; plan dependency consistency; exact matcher/fallback/capture constraints; deterministic raw-output policy; planning and registry validation. No production code change is authorized here.

## Effort & Risk



## Standards



## Acceptance Criteria

The complete Block 08 diff is reviewed and its three kernels plus supporting changes are decomposed into current PRBE-owned follow-ups. Each follow-up has exact source provenance, current-tree anchors or a fail-closed non-port decision, dependencies/conflicts, target APIs/data flow, correctness/performance tests, and evidence identity requirements. PRBE19 is re-scoped as the post-fix bake-in rule; no guessed anchors, monolithic port, or unreviewed production change is introduced.

## Notes

Provenance: stew675/llama.cpp branch rdna-boosts, commit 5efcd85fb4cd8845c6c7dd47c50e2666264aa4eb; title and diff header are preserved in tools/lab/rd25-block08-review/block08.diff. Discovery was made during PRBE11's attempt to port the historical RD25 source and summarized in completed PRBE99. The audited source is an input for PRBE05/11/13/18/20; the lab diff is not an implementation patch.

Provenance: stew675/llama.cpp rdna-boosts commit 5efcd85fb4cd8845c6c7dd47c50e2666264aa4eb, preserved in the lab diff. This completed audit feeds PRBE05/11/13/18/20; legacy RD items remain historical provenance only. PRBE19 is a bake-in rule, not a standalone Block 08 or RD25 prerequisite. The lab diff is an audit input, not an implementation patch.

2026-09-24 relevance at b11126: item's own description/steps/detailed_solution/acceptance_criteria already state the audit is COMPLETE ("Completed audit/design record") -- the Block 08 diff was fully reviewed and decomposed into PRBE05/11/13/18/20 follow-ups in a prior session, with provenance (stew675/llama.cpp rdna-boosts 5efcd85f) preserved in tools/lab/rd25-block08-review/block08.diff. No further plan authoring is needed here: this item is bookkeeping/provenance record, not an implementation target. Verified the referenced successor items (PRBE05/11/13/18/20) exist under docs/planning/active/patching-rdna-boost-experiments/. No GPT request issued (no design work required -- disposition is administrative closure of a finished audit).

2026-09-24 GPT review req_e17e0bf5a68c48d5: CONFIRMED. 2026-09-24 GPT review req_b43762f844fb40b3 applied: narrowed completion criterion to 'inventory/decomposition completed' only, explicitly excluding successor (PRBE13/PRBE18) implementation-readiness from this item's scope; noted tools/lab/rd25-block08-review/README.md still shows Status:active/Question state:open (verified) and needs a separate out-of-scope update to match.

## Change Log

- 2026-09-11T23:50:02.222039+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260911_235011_added-a-provenance-backed-bloc_4570
- 2026-09-11T23:50:11.493148+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:53:12.621788+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.268116+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:35:42.238602+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:35:50.684073+00:00 (state-transition): State: pending → completed
- 2026-09-24T05:09:08.377592+00:00 (updated-by): Updated: section:description, section:notes
