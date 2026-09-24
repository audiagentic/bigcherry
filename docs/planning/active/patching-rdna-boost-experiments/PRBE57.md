---
id: PRBE57
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:26.943294+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# FORK-MTP-001: Independent NextN/MTP tensor placement

## Description

TODO. GPT design requested this batch (req_83fbfa0995034d2d, covering PRBE57/58/59/101 together) was still running past a reasonable wait; this plan was authored directly, to be merged with GPT's output when it lands (see notes). Relevance: no existing patch for RD71 specifically (grep patches/*/patch.toml for RD71 = no hits); functional-keyword grep (nextn/mtp/speculative/topology/placement across patches/*/SUMMARY.md) found related-but-distinct MTP patches (1254 GDN MTP prefix tail, 1255 adaptive MTP depth, 1260 meta-view headroom for recurrent+MTP graphs, 1261 spec_ctx backend-device-list dedup for draft/target contexts) -- none of them address NextN/MTP tensor PLACEMENT by inter-device transfer cost, so this item is genuinely uncovered. Source (per item notes) is the external MrLordCat fork, commits 41a8ca78 (this item) and 1fcc05da (PRBE58) -- NOT locally available (no mirror for this fork in work/upstream/), so exact anchors cannot be verified in this pass; this plan is architecture-level, with fetching/inspecting the real fork diff as the mandatory first implementation step.

## Steps

1. Fetch/inspect the MrLordCat fork commits 41a8ca78 and 1fcc05da (both, since the item requires coupled RD71/RD72 analysis before choosing patch shape) -- this cannot proceed further without the real diff; register the source in config/external-sources.toml per this project's external-source provenance convention (see patches/1203_.../patch.py's PROVENANCE dict for the required shape) once fetched. 2. Identify where NextN/MTP draft-model output tensor placement is currently decided -- likely in llama.cpp's backend-scheduling layer (ggml_backend_sched_alloc_graph's tensor->backend assignment, or explicit backend hints set at model-graph-build time for the draft/target split, related to patch 1261_nro10_spec_ctx_other_devices which already touches draft/target backend-device-list plumbing -- read that patch first as the closest existing analog). 3. Design a cost model: for each candidate 'owner' GPU (first target-model GPU, last target-model GPU, or a dedicated device), estimate copies/token needed to move NextN's input (hidden state from the target model's last layer) and output (draft logits/tokens back to the acceptance-check code) -- this is a small, enumerable set of device-order choices (target has ~2 GPUs in the dual XTX/R9700 case), not a general optimizer. 4. Implement the placement as an explicit, topology-aware choice at model-load/graph-build time (not a runtime heuristic per-token) -- set the NextN tensor's preferred backend/device based on the cost model's cheapest owner. 5. Add temp-0 identity tests (deterministic greedy decode must produce identical tokens regardless of NextN placement) and an MTP-acceptance-rate regression test (placement must not change accept/reject statistics, only transfer cost). 6. Measure copies/token, TG, PP cost for owner alternatives (first-GPU/last-GPU/dedicated) x device orders, against single-GPU and MTP-off controls.

## Detailed Solution & Technical Design

Per the item's own text, this is explicitly a COUPLED investigation with PRBE58 (RD72, same source area, commit 1fcc05da) -- do not finalize RD71's placement choice independent of RD72's copy-count reduction, since the cheapest placement depends on which copy-path RD72 ends up choosing (fewer copies via host-staging changes the cost model RD71 optimizes over). This plan's step 1 (fetch both commits, diff/map overlap) is the prerequisite the item itself demands ('map overlap before choosing one patch, ordered pair, or supersession') and this planning pass could not perform it (no network access to the external fork, no local mirror). Everything past step 1 is a design SKELETON pending that real diff.

## Code Samples & Guidance

No real anchors available (external fork not locally mirrored). Once fetched, follow the provenance-dict and PATCHES-list pattern from patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/patch.py or patches/1215_rd394041_amd_stream_moe_overlap/patch.py (both document a fork commit, snapshot head/base, and adaptations list) as the required shape for the new package's patch.py header. patches/1261_nro10_spec_ctx_other_devices/ (existing, read its patch.py for the real current draft/target backend-device-list mechanism) is the closest current-tree analog to study before designing new placement logic.

## Files

To be determined after fetching the fork commits; likely src/llama-context.cpp or src/llama-model.cpp (draft/target device/backend assignment, near patches/1261's edits), ggml_backend_sched-related scheduling code. New package patches/<order>_rd71_nextn_mtp_placement/ (only after step 1).

## Validation

Step 1 (fetch fork source) is a hard prerequisite -- cannot be validated further without it. Once designed: temp-0 identity, MTP acceptance-rate parity, copies/token + TG + PP cost across owner alternatives and device orders vs single-GPU/MTP-off controls, on dual XTX/R9700 hardware (Brutus, not run here).

## Effort & Risk

Unscored by the item itself; set to M-L -- real risk is that this item cannot be fully scoped without the source fork diff, which was not obtainable in this planning pass; effort could shift significantly once the real diff is seen.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance; coupled RD71/RD72 analysis required before choosing patch shape (explicit item requirement, not yet satisfied).

## Acceptance Criteria

Select placement only after coupled RD71/RD72 overlap analysis and repeatable lowest total transfer cost with no acceptance regression; otherwise preserve existing placement.

## Notes

Supersedes: RD71
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd71

2026-09-24 relevance at b11126: no existing patch for RD71 (grep = no hits); functional-keyword search found related-but-non-overlapping MTP patches (1254/1255/1260/1261), closest analog is 1261 (draft/target backend-device-list plumbing). External fork source (MrLordCat, commits 41a8ca78/1fcc05da) not locally available -- this is a genuine blocker for anchor-level design, documented as this plan's mandatory step 1. GPT design request req_83fbfa0995034d2d (covering this + PRBE58/59/101) was still in progress when this plan was authored; check for its response and merge/reconcile with this plan if it landed with fork-specific detail this session could not obtain.

## Change Log

- 2026-09-09T10:57:26.943294+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:40.158372+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.384830+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.190479+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:13:19.105683+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031346_repaired-four-more-migrated-pa_4345
- 2026-09-10T03:13:46.500792+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:51:54.199727+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
