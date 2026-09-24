---
id: PRBE58
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:31.760072+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# FORK-MTP-002: Remove four-copy Vulkan MTP handoff

## Description

TODO. Same GPT/fork-availability situation as PRBE57 (coupled item, same source area). Relevance: no existing patch for RD72 (grep = no hits); no functional overlap found in existing patches (same keyword search as PRBE57). Source commit 1fcc05da (MrLordCat fork) not locally available. Disposition: TODO, architecture-level plan pending the real fork diff, explicitly coupled to PRBE57/RD71's placement decision per the item's own text.

## Steps

1. Fetch/inspect MrLordCat commit 1fcc05da (and diff against 41a8ca78/RD71 per the item's explicit instruction 'diff with RD71 commit 41a8ca78 before implementation') -- same hard prerequisite as PRBE57 step 1, shared fetch. 2. Enumerate today's four-copy Vulkan MTP handoff path (host<->device or device<->device copies moving hidden state/draft tokens between the target model's last layer and the NextN/draft head, and back for acceptance) -- this requires reading the current Vulkan MTP/speculative-decode dispatch code (likely in src/llama-context.cpp's speculative decode loop and the Vulkan backend's buffer-transfer helpers) to identify which of the 4 copies are structural (unavoidable device boundary crossings) vs redundant (e.g. an intermediate host staging buffer that could be skipped with a direct device-to-device transfer, if the Vulkan backend supports peer transfer on the target hardware). 3. Design path variants: (a) unchanged 4-copy baseline, (b) direct device-to-device transfer skipping host staging where peer access exists, (c) fewer logical copies via combining hidden-state and draft-token transfers into one dispatch. 4. Make the reduced-copy path CONTINGENT on PRBE57/RD71's placement choice (if NextN is placed on the same device as the copy source, some copies vanish entirely rather than being 'reduced' -- coordinate designs, do not finalize independently). 5. Add hidden-state/output identity tests (bit-exact vs the 4-copy baseline) as the correctness bar -- this is a transfer-path change, not a numerical change, so ANY difference is a bug. 6. Measure copy count/bytes/time and effective TG across path variants, draft depths, and contexts on dual RDNA4 Vulkan hardware, with single-GPU controls.

## Detailed Solution & Technical Design

This item is explicitly downstream of / coupled with PRBE57 (RD71): a copy-reduction design chosen independent of NextN's placement decision could contradict it (e.g. optimizing away a copy that RD71's chosen placement would have made moot anyway, or vice versa). This plan therefore treats PRBE57 as a co-requisite to finalize alongside, not a separate independent task -- whoever implements should read both plans together and likely land them as one coordinated change (or two patches with an explicit `requires` relationship in patch.toml, per this project's dependency-declaration convention) rather than two fully independent packages.

## Code Samples & Guidance

No real anchors available (fork not locally mirrored, same as PRBE57). Once fetched, follow the same provenance-dict/PATCHES-list convention as patches/1203_.../patch.py or patches/1215_.../patch.py.

## Files

To be determined after fetching the fork commit; likely Vulkan backend buffer-transfer code and src/llama-context.cpp's speculative-decode loop. New package patches/<order>_rd72_mtp_copy_reduction/ (only after fetching source; consider making it depend on / be sequenced with PRBE57's package via patch.toml `requires`).

## Validation

Fetch prerequisite (step 1) blocks further validation design. Once designed: hidden-state/output bit-identity vs the 4-copy baseline (mandatory, zero tolerance), copy count/bytes/time + effective TG + PP + VRAM across path variants/draft depths/contexts on dual RDNA4 Vulkan (Brutus, not run here), single-GPU controls.

## Effort & Risk

Unscored by item; set to M -- correctness bar is strict (bit-identity) but the change is confined to transfer/copy mechanics, not compute; real effort risk is coordination with PRBE57's coupled decision, not code complexity.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance; coupled RD71/RD72 analysis required (explicit item requirement); keep host-staging/four-copy fallback if correctness or VRAM/PP cost changes for the worse.

## Acceptance Criteria

Promote only if copies fall without hidden-state/output changes, PP/VRAM cost increase, or acceptance loss; retain fallback otherwise.

## Notes

Supersedes: RD72
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd72

2026-09-24 relevance at b11126: no existing patch for RD72 (grep = no hits); no functional overlap found. External fork source (commit 1fcc05da) not locally available -- same blocker as PRBE57, documented as mandatory step 1. GPT design request req_83fbfa0995034d2d (covering this + PRBE57/59/101) was in progress when this plan was authored; check for its response and merge/reconcile if it landed with fork-specific detail.

## Change Log

- 2026-09-09T10:57:31.760072+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:43.820256+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.389761+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.197952+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:13:26.102767+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031346_repaired-four-more-migrated-pa_4345
- 2026-09-10T03:13:46.510818+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:52:14.738266+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
