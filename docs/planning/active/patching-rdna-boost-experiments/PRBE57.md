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

TODO, corrected per GPT review (req_b43762f844fb40b3). Provenance/mechanism was misassigned: commit 41a8ca78 is backend-resident NextN hidden-state HANDOFF (that's PRBE58's/RD72's territory); 1fcc05da is the actual Vulkan NextN PLACEMENT change (this item's, RD71's, territory) -- the two commits were swapped between PRBE57/PRBE58 in the prior draft. This item's 'home-backend cost model' framing is new design, not a port of the cited fork commit as originally implied; exact package scope/anchors were unspecified. Relevance unchanged: no existing patch for RD71 (grep = no hits).

## Steps

1. Base PRBE57's design on 1fcc05da (not 41a8ca78 -- that belongs to PRBE58). 2. Anchor at src/llama-model.cpp::llama_model_base::create_tensor, at the verified text `const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;` -- confirm this anchor against live b11126 source before use (not yet re-verified this session; treat as tentative same as prior GPT design). 3. Port llama_model_tensor_is_nextn() plus a first-Vulkan-device buft_list_layer override, following 1fcc05da's actual mechanism once fetched/inspected (still not locally mirrored -- fetch/inspect remains the mandatory first implementation step). 4. Explicitly EXCLUDE from this item's scope: 1fcc05da's unrelated Vulkan init tracing, its Windows staging script, its loader template instantiation, and its 8192->512 server prefill-window change -- none belong to NextN tensor placement. 5. Treat 1fcc05da's llama-context.cpp pipeline-parallel-disable change as a SEPARATE copy-count subchange, not part of placement -- do not fold it into this item's package.

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

2026-09-24 GPT req_83fbfa0995034d2d COMPLETED. Its design is more concrete than this plan's skeleton: treats the handoff as choosing a HOME BACKEND for t_h_nextn (not just tensor placement), with a cost(owner)=copy-in+copy-out+sync+host-staging-penalty model over candidate owner GPUs, hook points at src/llama-model.cpp (NEXTN_PROJ_PRE/POST placement override), src/llama-context.cpp (t_h_nextn backend selection pre-graph-alloc), and a new BIGCHERRY_MTP_TOPOLOGY=off|auto|device:N env gate restricted to layer-split + >=2 accelerator devices initially. Test bar: temp=0 identity, byte-identical t_h_nextn rows, identical accept/draft counts, copies/token reporting. This is a genuinely better starting point than this plan's own skeleton -- prefer GPT's design when implementing, using this plan's step-1 fetch requirement (real fork commits 41a8ca78/1fcc05da) to fill in GPT's still-tentative anchors (GPT also had no access to the real fork diff, so its file/hook references are structurally plausible but unverified, same caveat as this plan).

2026-09-24 GPT review req_b43762f844fb40b3 applied: corrected commit attribution (this item is 1fcc05da/Vulkan placement, not 41a8ca78/backend handoff which belongs to PRBE58); added the verified llama-model.cpp::create_tensor anchor and an explicit exclusion list (Vulkan init tracing, Windows staging script, loader template instantiation, 8192->512 prefill window, pipeline-parallel-disable) for scope discipline.

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
- 2026-09-24T04:53:30.568089+00:00 (updated-by): Updated: section:notes
- 2026-09-24T05:09:23.096378+00:00 (updated-by): Updated: section:description, section:steps, section:notes
