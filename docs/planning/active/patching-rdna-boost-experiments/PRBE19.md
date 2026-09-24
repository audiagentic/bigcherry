---
id: PRBE19
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:40.683647+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Apply post-fix source-state bake-in across affected PRBE ports

## Description

TODO (process rule, not code). PRBE19 is a cross-cutting source-state constraint, not an implementable patch: whenever PRBE13/16/18/20 are materialized, their SSM/MMVQ source regions must be taken from the reviewed post-fix state (provenance commit 9e46e1fd...) rather than the raw historical RD25 diff, and checked for semantic/content equivalence against the current v3 snapshot (c8af5361...). Relevance: still meaningful -- PRBE18/PRBE20 (this batch) are live dependents. No GPT design request needed (no source code to design); the deliverable is a documented verification step embedded in each dependent item's plan.

## Steps

1. When authoring/updating PRBE13, PRBE16, PRBE18, or PRBE20's implementation plan, add an explicit validation step: 'confirm sourced SSM/MMVQ region matches reviewed post-fix provenance 9e46e1fd... and is semantically equivalent to current v3 snapshot c8af5361..., not the raw RD25 diff'.
2. For PRBE18 (this batch), that step was added via its own plan update (see PRBE18 validation section) referencing this rule.
3. For PRBE13/PRBE16/PRBE20, whoever materializes those items must add the equivalent check at that time -- PRBE19 itself performs no source edit and defines no test file.
4. Do not treat 'branch tip' as if it were the reviewed v2 image; always diff against the specific provenance commit.
5. PRBE11/RD12 is explicitly outside the affected set -- do not add this check there.

## Detailed Solution & Technical Design

PRBE19 has no data flow or dispatch of its own. It is enforced by cross-referencing: each dependent item's `validation` section must name the post-fix provenance commit and require a semantic-equivalence diff against it before the dependent's own correctness gate runs. This plan item's own completion criterion is that the cross-reference exists in every live dependent (PRBE18 done this batch; PRBE13/PRBE16/PRBE20 when they are next materialized).

## Code Samples & Guidance

N/A -- no source file is touched by PRBE19 itself. See PRBE18's code_samples for the one dependent materialized this batch.

## Files

No source files. Cross-referenced from: docs/planning/active/patching-rdna-boost-experiments/PRBE13.md, PRBE16.md, PRBE18.md, PRBE20.md (validation sections).

## Validation

Audit: for each of PRBE13/16/18/20 that is 'materialized' (has a patch package), confirm its validation section names the post-fix provenance commit and a semantic-equivalence check step. PRBE18 satisfied this batch (see PRBE18.md validation). PRBE20 Wave-2 MMVQ/SSM regions also require this check when authored (see PRBE20.md, added this batch).

## Effort & Risk

Low effort (documentation/cross-reference only); risk is entirely in dependents applying it correctly, not in PRBE19 itself.

## Standards

Immutable reviewed post-fix source state; no standalone patch; PRBE-owner-aware sequencing; dependency-aware promotion.

## Acceptance Criteria

Every affected PRBE13/16/18/20 port uses semantically equivalent reviewed post-fix source state and passes its own correctness gate; no broken pre-fix region is materialized; PRBE19 remains a sequencing/source-state rule and RD25 remains provenance only.

## Notes

Supersedes: RD25
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd25

Supersedes: RD25 (closed historical predecessor). RD25 remains historical evidence explaining why PRBE19 exists. It must never again appear as a prerequisite task; PRBE11/RD12 is outside the affected set.

2026-09-24 relevance at b11126: TODO (process rule). Not upstream-absorbed (RD25's fix was fork-local, not upstream), not obsolete (PRBE18/20 in this batch are live dependents), not a standalone patch. No GPT design request needed -- cross-reference added directly to PRBE18 and PRBE20 plans instead (GPT request ids req_9d3d9188405f49f3 for PRBE18, req_bc106a2613754b6b for PRBE20).

## Change Log

- 2026-09-09T10:54:40.683647+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:57.480362+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.214040+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.929217+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:50:36.902436+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025049_rdna-successors-prbe1719-now_5726
- 2026-09-10T02:50:49.233411+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:52:35.243207+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.248056+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T10:09:43.707400+00:00 (updated-by): Updated: section:title, section:validation, section:standards, section:acceptance_criteria
- chg_20260912_101015_fixed-the-remaining-plan-taxon_2574
- 2026-09-12T10:10:15.552804+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:23.238612+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
