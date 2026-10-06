---
id: MSM03
order: 3
plan: patching-meta-split-memory
state: pending
created-at: '2026-10-06T11:48:21.824495+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Attention-side state only on the cards that run attention (subset-mirrored indexer cache and masks)

## Description

For qwen4exp the lightning-indexer key cache (cache_idx_k_l*) and the context-shaped attention inputs (KQ mask [n_kv, n_tokens], kpool cells/idxs/mask, QSA mask_all / indexer_sel [n_kv+n_sel, n_tokens]) are MIRRORED on every device of the tensor split, so they scale with -c on a card whose attention share is zero (BIGCHERRY_ATTN_TS=1,1,0). Goal: such a card holds none of them.

## Steps

FIRST STEP (new patch, flag off by default): add active_mask to the split state and its propagation rules with the mask always 0 (no behaviour change) plus the unit of the mechanism that is testable offline; second step seeds the mask from the attention split for cache_idx_* only; third step the compute-side masks. Each step: offline test, build, identical output gate, per-device MiB from MSM01.

## Detailed Solution & Technical Design

From GPT (req_cc4f6bc692c2431f). The existing split state cannot express 'full replica on GPU0/1, absent on GPU2': llama_meta_device_get_split_state() returns MIRRORED for pattern_idx_cache (src/llama-model.cpp ~L506); handle_lightning_indexer() in ggml-backend-meta.cpp (~L808) requires all four inputs MIRRORED; build_attn_inp_kq_mask() (src/llama-graph.cpp L29-45) and build_inp_kpool() (src/models/qwen4exp.cpp L699+) create context-sized compute GGML_OP_NONE inputs which the meta backend makes MIRRORED. Changing pattern_idx_cache to an AXIS split is INCORRECT: each participating attention GPU needs the complete cache for its local indexer/top-k. Correct semantics are subset-mirrored: extend the meta split state with an active_mask (0 = legacy/all devices), propagate it through MIRRORED operations, make handle_lightning_indexer() require equal masks and return that mask, make init_tensor_impl() create zero-sized simple tensors on inactive devices (alloc_buffer_n already supports contexts with only zero-sized tensors via a dummy buffer), and seed the qwen4exp indexer / KQ / kpool / QSA branches from 1303's BIGCHERRY_ATTN_TS[j] != 0. No operation was found that requires these tensors on an attention-share-zero GPU.

## Code Samples & Guidance



## Files

new patch package patches/1341_meta_subset_mirrored/ (patch.py, patch.toml, SUMMARY.md), tools/tests/patch/test_1341_meta_subset_mirrored.py; vendor files: ggml/src/ggml-backend-meta.cpp, ggml/include/ggml-backend.h (split state struct), src/llama-model.cpp

## Validation

Flag off: identical. Flag on with BIGCHERRY_ATTN_TS=1,1,0: identical greedy text and probes; indexer cache and mask MiB on the zero-share device drop to zero in MSM01's report; no change on the attention devices. Must compose with 1303, 1283, 1334 (sparse FA), 1335 (tiled indexer), 1327 / 1332 (QSA).

## Effort & Risk

High: changes the split-state struct seen by every rule; a mask that is dropped by one rule silently produces a tensor that is missing on a device. Mitigation: mask 0 = legacy everywhere, assert on mask mismatch, step-wise seeding.

## Standards

Package-only patch, fail-closed anchors, offline test; requires 1303.

## Acceptance Criteria



## Notes

Depends on MSM01 for measurement; independent of MSM02 but the saving on the compute side only materialises with MSM02 (the masks are compute tensors in the common arena).

2026-10-06 FIRST STEP ON HARDWARE (1341_meta_subset_mirrored by GPT, commit 6b897b6d; build b-metamem-b11402b2 = production set + 1339 + 1340 + 1341; BIGCHERRY_META_SUBSET_MIRROR=1 with BIGCHERRY_ATTN_TS=1,1,0; run metamem-b11402b2, depth 2048, one request per arm). Indexer cache MiB on ROCm0 / ROCm1 / ROCm2: flag off 288 / 288 / 288 (ctx 49152) and 1440 / 1440 / 1440 (ctx 245760); flag on 288 / 288 / 0 and 1440 / 1440 / 0 - in the production row split and in the owner's expert-split layout. Greedy text identical to the flag-off arm in both layouts and both contexts (md5 8668d7dd production, 2571b60b owner's layout); no error lines. So the R9700 gives back 1440 MiB at the full context with unchanged output; the production layout benefits too, because it already runs attention on the XTXs only. Offline tests pass. Pending for promotion (base framework change: identical output + no regression): ABBA on production at ctx 245760, depths 8K / 98K / 202K (queued, msm03-b2), then the compute-side masks (KQ / kpool / QSA), which need MSM02's per-device arena to show up as memory.

2026-10-07 FIRST STEP VALIDATED AND PROMOTED (1341 at head c0f18e33 + later; build b-metamem-b11402f2). Production, ctx 245760, ABBA (msm03-f2): 8K prefill 1070.2/1070.3 vs 1060.9/1048.3, decode 84.2/84.7 vs 84.5/84.7; 98K prefill 991.1/990.1 vs 972.6/989.1, decode 58.3/58.2 vs 58.1/58.3; greedy text identical at both depths; probes flag on vs off top-1 24/24, TV 0.0000. Memory: indexer cache on the R9700 288 -> 0 MiB (49K) and 1440 -> 0 MiB (245K), production and owner's layout, re-measured on three builds. 1341 is in validated-enhancements and BIGCHERRY_META_SUBSET_MIRROR=1 is set in src/profile/flashnext.ini (the profile already sets BIGCHERRY_ATTN_TS=1,1,0, which is what makes the R9700 a zero-share device). README with the evidence in the patch package. REMAINING for this item: second step = compute-side context-shaped inputs (KQ mask, kpool tables, QSA mask_all / indexer_sel) subset-mirrored; their memory only returns once MSM02's per-device arena is correct, so the order is MSM02 first.

## Change Log

- 2026-10-06T11:48:21.824495+00:00 (created-by): Created by agent
- 2026-10-06T12:36:47.707067+00:00 (updated-by): Updated: section:notes

## Ledger-events

- chg_20261006_144829_flash-next-uses-about-14-gb-l_2832
- 2026-10-06T14:48:32.643443+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-06T14:48:46.318179+00:00 (updated-by): Updated: section:notes
