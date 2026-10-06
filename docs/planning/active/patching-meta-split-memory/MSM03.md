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

## Change Log

- 2026-10-06T11:48:21.824495+00:00 (created-by): Created by agent
