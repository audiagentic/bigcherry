---
id: PRBE07
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:00.396867+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Evaluate BridgeSpec (kdheeraj-p/bridgespec) -- HIP MTP/DFlash speculative decoding sidecars + gfx1100 MMVQ tuning

## Description

NON-PATCH-IMPLEMENTATION ITEM (confirmed by GPT review): this is explicitly a research/triage item -- it defines no patch package, files, anchors, dispatch, gate, marker, or correctness test, and none should be fabricated for it. Concrete next action: pin BridgeSpec's exact source commit + license via `gh api`, read it in full, and produce the written adopt/adapt/inspiration-only/not-applicable recommendation described in steps/detailed_solution. If BridgeSpec is adopted, open a NEW, separate plan item with its own exact b11126 files/anchors/patch-package design -- do not fold implementation into PRBE07. Item stays pending until the recommendation is written.

## Steps

1. `gh api repos/kdheeraj-p/bridgespec` (or its actual host) to pin the exact source commit and license; read README, benchmark README/CSV, integration patches and MMVQ tuning source in full at that pinned commit -- do not summarize from memory of the project name.
2. Grep this repo's docs/planning for THA05 and any RD-series speculative-decoding item to identify the current dispatch/verification-width design to compare against (`grep -rl THA05 docs/planning/`).
3. Build a comparison table: verification-width 2-8 handling, sidecar ABI (how the sidecar process/thread talks to llama-server), host-mediated KV state transfer, vocabulary slicing/remap scheme, VRAM co-residency (sidecar model + main model on the same GPU set).
4. Explicitly assess dual-XTX tensor-split applicability -- BridgeSpec's published benchmarks are single-GPU; do not generalize them to this project's dual-XTX production topology without saying so.
5. Treat every published BridgeSpec number as directional only. If (and only if) a reproducible local probe is authorized later, define it here as a follow-up: a small 9B or 27B model, fixed prompt set, acceptance-rate and throughput measurement, with an explicit fallback/rejection gate -- but do not run it as part of this item.
6. Write the adopt/adapt/inspiration-only/not-applicable recommendation into this item's notes with explicit evidence citations (source commit, license, comparison table) and, if anything is adopted, open a NEW plan item + patch package for it rather than folding it into PRBE07.

## Detailed Solution & Technical Design

This is research triage, not kernel design -- there is no upstream b11126 source to anchor against because BridgeSpec is an external, non-integrated fork. The deliverable is a provenance-backed written comparison and a recommendation with an explicit evidence trail (never representing BridgeSpec's own benchmark numbers as BigCherry evidence, per this item's own standards). Any adopted technique becomes its own new plan item and patch package with its own real hardware qualification -- PRBE07 itself only ever reaches a recommendation, never a promotion.

## Code Samples & Guidance



## Files

No repo files change. Deliverable is this plan item's own notes/description (the written recommendation) plus, only if something is adopted later, a new plan item under docs/planning/active/patching-rdna-boost-experiments/ and a new patches/<id>/ package (out of scope for PRBE07 itself).

## Validation

Source/license verification (pinned commit hash recorded); comparison table cross-checked against this repo's real THA05/RD-series dispatch code (cite file:line); no BigCherry-hardware validation applies to PRBE07 itself since no code changes -- any local reproduction probe described in step 5 is future, separately-authorized work, not part of this item's closure.

## Effort & Risk

S/M effort (reading + writeup, no code). Risk: representing third-party numbers as validated BigCherry evidence -- explicitly disallowed by this item's own standards; mitigate by citing provenance for every number used.

## Standards

Third-party research-preview provenance; never fabricate promotion evidence; separate controlled A/B from high-water claims; preserve memory-safety and VRAM-budget gates.

## Acceptance Criteria

A written recommendation identifies concrete transferable technique or explicitly rejects applicability; no third-party benchmark is represented as BigCherry evidence; any adopted change receives its own patch and qualification item.

## Notes

Supersedes: RD101
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd101

2026-09-24 relevance at b11126: TODO, no patch exists, pure research/comparison item (no kernel-level design applies). No GPT design request used -- this item has no code to design against; it is a literature/provenance triage whose output is a written recommendation, consistent with its own Acceptance Criteria ("A written recommendation... no third-party benchmark is represented as BigCherry evidence").

2026-09-24 GPT review req_7f4dea253b7247f0 applied: confirmed PRBE07 is explicitly not a patch-implementation item (research/triage only); description now says so explicitly with the concrete next action (pin commit/license, write recommendation), and reaffirms any adoption spawns a separate implementation item. Item remains pending.

## Change Log

- 2026-09-09T10:54:00.396867+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:06.504082+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.160426+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.848976+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:32:20.227395+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_023232_the-next-three-rdna-successors_5807
- 2026-09-10T02:32:33.005493+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:27:56.635263+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:37:06.308167+00:00 (updated-by): Updated: section:description, section:notes
