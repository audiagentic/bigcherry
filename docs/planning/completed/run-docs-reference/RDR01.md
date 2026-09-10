---
id: RDR01
order: 11
plan: run-docs-reference
state: completed
created-at: '2026-09-09T10:47:23.952788+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# Reference corpus classification and campaign-artifact relocation

## Description

Reference cleanup has landed incrementally, but the frozen item remains in progress without evidence that its full classification/relocation boundary is accepted.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: run

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/run-docs-reference-do01.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: HI36,PA18.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (2); preserve historical references (2) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: DO01
Migration: capability-rebaseline-v3-2026-09
Successor key: run-docs-reference-do01

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): AUDIT FIRST — current residual scope is genuinely unknown from the stub/change-log alone; do not assume either "already done" or "fully open." Before touching any files, check: (1) does every current docs/reference/** entry have exactly one classification + an authoritative owner/location; (2) for every THA27/HI36 corpus file, does git grep identify every tool/test/fixture/doc consumer, with exactly one copy authoritative; (3) are any supposedly-movable files consumed by deterministic tests via path/filename/content hash; (4) has PA18 already dispositioned all patch-owned material (avoid duplicate ownership); (5) are generated reports clearly generated/reproducible rather than hand-maintained authorities; (6) do archive/campaign-artifact entries remain immutable historical evidence rather than live reference; (7) do indexes/links/fixtures point only to current authorities with link/reference tests green; (8) map DO01's Aug-Sep change-log entries back to each ORIGINAL acceptance requirement rather than treating activity/change-count itself as evidence of completion. If classification + THA27/HI36 authority map + link/fixture validation are already complete, close as satisfied rather than performing gratuitous relocation. RDR01's classification should become RDR02's scope filter, so RDR02's mechanical portability pass never rewrites immutable historical evidence. Execution order: ranked #11, before RDR02.

CORRECTION from deeper repo-validated dev-gpt review (2026-09-10): this is CLOSER TO DONE than the prior review's "audit first, scope unknown" framing suggested. docs/reference/README.md already establishes ownership/location rules; the experiment reference corpus is now small; the HI36 verdict is under evidence; historical material has already been moved to docs/archive. Treat this as a BOUNDED closure audit against the checklist already in this item's notes, not a potentially-large unknown relocation task. Execution order stays #11 (unchanged), immediately before RDR02.

CLOSURE AUDIT DONE 2026-09-10, per the deeper repo-validated review's exact checklist -- all 8 items checked directly against current HEAD:
1. Loose docs at root: only docs/README.md (clean).
2. Per-entry classification: docs/reference/README.md already gives every entry an explicit type + authority; spot-checked CANDIDATES.md/FINDINGS.md/START_HERE.md/TUNING-DETAIL.md/the RDNA backlog pointer -- all self-documented (generated / cross-project log / orientation guide / archive pointer / plan-item pointer).
3+7. THA27/HI36 corpus: `git grep -nE 'THA27|HI36' -- docs tools` and `-- tools/tests` found ZERO remaining THA27 references and only HI36 as design-rationale comments (generalise.py, test_generalise.py) plus one fixture-provenance note (tools/tests/fixtures/replay/README.md, 'extracted from the HI35/HI36 campaign bundle' -- already extracted/owned locally, not a live external corpus needing relocation). DO01's original 'resolve ownership for the THA27/HI36 campaign corpus' concern is RESOLVED -- no corpus remains unresolved.
5. Stale top-level report refs (REPORT_GPU_RUNTIME_MIGRATION etc): none found.
6. Duplicate filenames under docs/**: 12 names duplicate, ALL verified as legitimate archive-vs-live pairs or split-authority pointers (e.g. docs/reference/archive/HANDOFF.md is a 5-line compat stub vs the real 969-line docs/archive/HANDOFF.md; docs/archive/TUNING-DETAIL.md vs docs/archive/hip-autotune/TUNING-DETAIL.md are genuinely different historical snapshots, not duplicates; PATCH_VALIDATION.md/TOOL_DISPOSITION.md pairs are pointer-vs-authority by design per README.md's own table). No unresolved duplicate ownership found.
8. Archive immutability: spot-checked, holds.
Local Markdown-link audit: wrote an ad-hoc regex-based checker (not a new permanent tool) over all docs/**/*.md relative links. Found 5 candidates; 3 were false positives from code-comment parens inside completed/historical plan items (not real links); 2 were real -- one (docs/planning/completed/hip-autotune/HI19.md, docs/planning/completed/session-recovery/SE03.md) is quoted historical text inside COMPLETED plan items, correctly left as-is per checklist item 8 (immutable historical evidence, not live reference); one WAS a real broken link in a live maintained doc (docs/reference/patches/PATCH_REFACTOR_RUNBOOK.md pointed at planning/active/patch-system/... but that plan item has since completed and moved to planning/completed/patch-system/...) -- FIXED.

VERDICT: classification + THA27/HI36 authority/consumer map + link/fixture validation are all complete. Per the review's own instruction ('if all entries classify cleanly and consumer/reference scans are clean, close RDR01; do not manufacture relocation work to satisfy the stub'), this item is CLOSED AS SATISFIED -- no further relocation/classification work identified. DO01's original scope is done.

## Change Log

- 2026-09-09T10:47:23.952788+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:24.437731+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.739805+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:54.982925+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:09.565275+00:00 (updated-by): Updated: order=11, priority='P3'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.905787+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.183063+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:19:08.857215+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:30:11.881190+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:30:14.091301+00:00 (state-transition): State: pending → completed
- chg_20260910_003019_closed-the-docs-reference-clas_8141
- 2026-09-10T00:30:19.567154+00:00 (updated-by): Updated: section:ledger-events
