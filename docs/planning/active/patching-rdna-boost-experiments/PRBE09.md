---
id: PRBE09
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:07.484798+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# UP-VK-005: Vulkan tensor-parallel AllReduce -- upstream PR/issue tracking and BigCherry adoption path

## Description

NON-PATCH-IMPLEMENTATION ITEM (confirmed by GPT review): explicitly upstream-tracking only -- no Vulkan tensor-parallel AllReduce implementation exists or is pinned yet, so no patch package can be authored from this plan today. Concrete next action: search upstream llama.cpp issues/PRs for a Vulkan tensor-parallel AllReduce proposal, record PR/issue number + status + license, and grep b11126 to confirm nothing has landed yet. If a concrete upstream commit is later identified, open a NEW, separate plan item with exact b11126 insertion/dispatch/fallback anchors and tests -- do not fold implementation into PRBE09. Item stays pending until then.

## Steps

1. Search upstream llama.cpp issues/PRs for a Vulkan tensor-parallel AllReduce proposal (search terms: "vulkan", "allreduce", "tensor parallel", "tensor-split collective"); if found, record its PR/issue number, current status (draft/open/merged/closed), API/ABI shape and license via `gh api`/`gh pr view`.
2. `git -C work/upstream/llama.cpp.git.git grep -n <candidate symbol>` against b11126 to check whether any part of it has already landed at this pin -- if the whole thing is already merged and present, this item becomes UPSTREAM-ABSORBED instead.
3. If nothing is merged yet: read this repo's own RD68 and RD57 items (`docs/planning -- grep RD68/RD57`) to record their tensor-parallel topology and Vulkan-provider-boundary constraints this candidate must respect; do not silently duplicate RD57's scope.
4. Only if an upstream implementation is available and buildable, define a reproducible local control/treatment plan: single-GPU control, dual-GPU tensor-split treatment, correctness of the collective op, fallback to host-mediated reduce on failure, end-to-end throughput.
5. Record adopt / defer / reject with explicit evidence (or absence of an implementation to test) and keep the upstream reference (PR/issue URL + commit) in this item's notes.

## Detailed Solution & Technical Design

This item owns tracking and an adoption decision only. If no upstream Vulkan tensor-parallel AllReduce implementation exists yet at b11126 (to be confirmed in step 1-2), the correct disposition is TODO/deferred with an explicit "nothing to adopt yet" recorded, re-checked at the next llama.cpp pin bump -- not a fabricated implementation plan. If one exists and is merged, re-classify as UPSTREAM-ABSORBED with the exact file:line evidence. If one exists as an unmerged PR, this item's job is the adoption analysis, not authoring a BigCherry patch port until that analysis recommends adoption.

## Code Samples & Guidance



## Files

No repo files change from this item alone. Deliverable: this item's own notes recording the upstream PR/issue status, RD68/RD57 cross-reference, and the adopt/defer/reject decision. A future adoption becomes its own new plan item + patches/<id>/ package.

## Validation

Upstream status/license check (cite PR/issue URL and commit); `git -C work/upstream/llama.cpp.git.git grep` evidence for any already-merged pieces at b11126; RD68/RD57 cross-reference confirmed by grep in docs/planning/; no hardware validation applies unless step 4's conditions are met, and any such run happens on Brutus separately, not as part of this triage.

## Effort & Risk

S effort for the tracking pass; work escalates to L only if an implementation is actually available to test. Risk: claiming adoption-readiness without a real upstream implementation to point at.

## Standards

Upstream provenance; provider/topology isolation; no implementation claim without source and evidence.

## Acceptance Criteria

An auditable upstream/adoption decision is recorded; any adopted code has exact source identity and passes correctness/fallback gates; otherwise the candidate remains deferred without production claims.

## Notes

Supersedes: RD104
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd104

2026-09-24 relevance at b11126: TODO. No GPT design request used -- this is upstream-tracking/adoption triage with no local kernel code to design against yet; the concrete next action (search upstream PRs, grep b11126 for any already-landed pieces) is procedural, not a design problem.

2026-09-24 GPT review req_215c89d0b13a4bb7 applied: confirmed PRBE09 is explicitly upstream-tracking only, no implementation exists to pin; description now states the concrete next action (search upstream, grep b11126) and that adoption spawns a separate implementation item. Item remains pending.

2026-09-24 GPT review req_7f4dea253b7247f0 applied: confirmed PRBE09 is explicitly upstream-tracking only, no implementation exists to pin; description now states the concrete next action (search upstream, grep b11126) and that adoption spawns a separate implementation item. Item remains pending.

## Change Log

- 2026-09-09T10:54:07.484798+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:13.948243+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.169761+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.860761+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:35:11.802321+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023529_three-more-rdna-successors-now_3176
- 2026-09-10T02:35:29.332235+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:28:13.834516+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:37:18.873788+00:00 (updated-by): Updated: section:description, section:notes
- 2026-09-24T04:37:30.168702+00:00 (updated-by): Updated: section:notes
