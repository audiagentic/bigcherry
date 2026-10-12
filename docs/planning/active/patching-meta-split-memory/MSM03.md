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

## 2026-10-12 implementation audit: instance-owned KQ-mask subset admission

**Scope/eligibility.** Only 1341's graph-input subset-mask lifetime and the already-owned MSM01-03 documentation. No independent 1340/1341/MSM01-03 commits, material PRs, benchmark/queue activity in the preceding 12 hours on main or the selected branch. Recent BCOP145-154 audited different mechanisms. Active Radiance RR08/MXFP4/GDN/paged prefill/wire, PA44/PA49 release, QFP15 re-screen, QFP53/1362 and QFP49/1361 are protected and untouched. Do not add a scheduler, allocator, mask cache, dispatch table or telemetry system.

### Current source and measured facts

- `engines/llamacpp/patches/1341_meta_subset_mirrored/patch.py::_N_INPUT_RULE` already assigns `active_mask` to `GGML_OP_NONE` inputs whose name contains `kq_mask` (including scheduler-prefixed copies). `_N_MODEL_SEED` sets the mask while classifying Qwen4Exp `cache_idx_(k|v)_l*`. `BIGCHERRY_META_SUBSET_MIRROR_INPUTS` defaults to **1** when subset mirroring is enabled. The patch header, SUMMARY and historical README claiming "indexer only; KQ inputs not seeded" are stale. Independent kpool and computed QSA mask coverage is **not** established by this rule.
- In pinned llama.cpp **b11552**, `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_get_split_state` handles graph `GGML_OP_NONE` locally. Its `ggml_backend_meta_buffer_context::split_state_cache` is keyed by tensor pointer, `assume_sync` and a tensor-struct snapshot, **not** external mutable policy. `ggml_backend_meta_device_context` already stores `get_split_state` and `get_split_state_ud` per device, currently used for static model tensors.
- **Source-proven conditional correctness risk, not reproduced on GPU:** 1341 stores `static uint32_t bc_meta_attn_input_mask = 0` in process scope, writes it via a non-atomic setter and reads it during graph split inference. No per-Meta-instance identity, reset, synchronization or cache generation exists. 1303 explicitly acknowledges concurrent split-state callbacks. Parallel plain accesses can be a C++ data race; sequential different-mask instances can read the last writer's mask, and an unseeded instance can inherit a stale mask. Current production `BIGCHERRY_ATTN_TS` is process-configured and Qwen4Exp-only, so differing concurrent policies are **not proven to be supported or observed**. Do not claim a production crash.
- Existing first-party b11402/b11474 qualification: 1341 saved **1,440 MiB** indexer-cache memory on the zero-attention-share R9700 at ctx 245760, with identical greedy/probes. With 1340 and mask-input rules, common 1020.9 MiB arena became 880.8/880.8/378.8 MiB across XTX/XTX/R9700; fusion-on text differed because address-sensitive fusion chose other kernels (FKE01 owner). These are historical receipts, **not new gains**.

### Cheapest discriminator completed

Compiled a standalone C++17 ownership fixture using g++ 14.2: **169 assertions passed across 42 distinct-mask pairs**. Setting instance A `0b011`, then B `0b101`, changes A's process-global read; an unseeded instance sees B's mask. Instance-owned fields preserve both. Replacing the global with `std::atomic<uint32_t>` alone does **not** isolate contexts. This is a host *model* of the source's state choice, not a repository test or GPU failure. Existing `test_1341_meta_subset_mirrored.py` asserts the global literal and patch composition, not two-instance behavior.

### Bounded implementation-ready gate (MSM03 owns it)

1. **CPU first; no new mask optimisation.** Reuse existing `ggml_backend_meta_device_context::{get_split_state,get_split_state_ud}` to classify narrowly eligible `GGML_OP_NONE` KQ-mask graph inputs through the existing model callback instead of the process-global setter. In `ggml/src/ggml-backend-meta.cpp::ggml_backend_meta_get_split_state`, call the callback only for verified graph-input mask names; otherwise preserve native MIRRORED. In `src/llama-model.cpp::llama_meta_device_get_split_state`, handle graph inputs **before** `get_tensor_config()` (which parses weight/cache names), derive the mask from that model's validated `BIGCHERRY_ATTN_TS` and return MIRRORED+mask. Preserve static `cache_idx` seeding and existing 1341 propagation/transfers. Remove the global variable/setter rather than adding an atomic global. If the existing callback cannot safely support graph inputs, retain legacy MIRRORED for them and stop; no parallel policy registry.
2. **Identity/lifetime.** Validate actual Meta device order, `n_devices`, mask bounds, nonzero active devices, and the all-devices sentinel `active_mask=0`. Admit only exact `attn_inp_kq_mask`, `dsv4_<part>_kq_mask`, `self_kq_mask` plus verified `<backend>#<name>#<n>` scheduler wrappers. Reject unrelated substring matches. Freeze policy for each Meta buffer lifetime; on any legitimate change rebuild the buffer/split-state cache before reuse. Never mutate a cached graph's policy in place. Preserve zero-sized inactive tensors, COMPUTE flags, async transfers and view/alias semantics.
3. **Existing-test extension.** Add A/B/A two-context interleaving, concurrent callback use, unseeded/legacy/all-devices controls, reversed device order, invalid name/mask, same-process repeated requests, cached graph reuse and idempotent patch composition. Assert no global setter remains. Run `test_1341_meta_subset_mirrored.py`, `test_1340_meta_per_device_arena.py`, patch-lint and `meta-memory` composition on pinned b11552.
4. **Hardware only after CPU pass.** Match flag-off/on controls on 2x gfx1100 + gfx1201 (`BIGCHERRY_ATTN_TS=1,1,0`), ctx 49152/245760, 2K/8K/98K prefill, tg128, multi-ubatch, two requests per process, MTP off/on, graph on/off, 1340 common/per-device arena controls. Reuse `BIGCHERRY_META_MEM` for static/indexer/arena bytes and zero-share work counts; require correct logits/KLD/greedy/acceptance, no stale graph, illegal access, missing work or wrong-device traffic. The gfx1030 drafter is **not** an additional Meta attention device.
5. **Terminal decision.** If supported deployments are proven single-policy and cannot exercise distinct Meta contexts, document that limitation and close without implementation. Otherwise require per-instance binding before extending KQ/kpool/QSA coverage. A memory saving is not a reason to accept a correctness regression. Preserve 1341 flag fallback and FKE01 ownership.

**External mechanisms:** upstream llama.cpp b11552's per-device callback/userdata and buffer-local cache are the reuse target. Upstream [#26395](https://github.com/ggml-org/llama.cpp/pull/26395) is an open Meta split-axis bounds/test change, not a fix for this policy lifetime. vLLM [BlockPool](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py) and SGLang [memory_pool.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py) are instance-owned cache-lifetime references, **not** transferable AMD performance claims.

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

2026-10-07 BACK TO BASICS (owner: the memory saving is crucial; 1340's per-device allocator keeps costing decode). WHAT FILLS THE SHARED COMPUTE ARENA (1329 + 1331 traces, build b-arenac-b11402q2, production, ctx 245760, ub 512; reserved 1020.90 MiB; live at the peak): attn_inp_kq_mask [245760 x 512] f16 = 240 MiB (input); indexer_sel [245760 x 512] f16 = 240 MiB (ADD, computed); QSA combined mask [247811 x 512] f16 = 242 MiB (REPEAT, computed); pooled-key mask leaf [61504 x 512] f16 = 60 MiB (input); everything else about 100 MiB (MUL_MAT 24, HC_POST 20, MUL 20, ROPE 12, ...). So ~780 of ~880 MiB live at the peak are four 'context cells x batch tokens' matrices, mirrored in full on all three devices; two are host inputs, two are computed on every device (the R9700 computes masks it never reads). RE-PLAN CAUSE (counters added to 1340, one request, all allocators in the process): node count differs 181, larger than planned 118, planned-external-now-needed 3, leaf count 0; the last 'larger' tensor was attn_inp_kq_mask 83968 > 81920 bytes - it is sized by the current KV length (padded to 256), so plans are redone as the sequence grows; the scheduler's own allocator pays this too. The earlier zero-size-slice explanation was wrong (3 events). CHUNKING WITH 1332 (validated patch, BIGCHERRY_QSA_CHUNK, no new code), arena at ctx 245760 ub 512: chunk 256 -> 556.63 MiB; 128 -> 556.08; 64 -> 578.99 (about -464 MiB on EVERY device; once the computed masks are chunked the peak is the inputs, ~330 MiB, of which the KQ mask is 240 and the pooled-key mask 60). SPEED of chunking at ub 512, ctx 245760, ABBA vs production: chunk 128: 8K prefill 1011.5/1012.2 vs 1075.9/1078.6 (-6%), decode level; 98K prefill 909.6/909.4 vs 960.7/989.1 (-7%), decode 53.6/53.2 vs 57.6/58.3 with lower acceptance; probes 21/24, TV 0.119. chunk 256: 8K prefill 1013.8/1011.0 vs 1063.3/1035.0 (-4..5%), decode 87.6/88.3 vs 84.6/84.1 (acceptance 355/466 vs 349/485); 98K prefill 923.3/923.2 vs 989.1/988.5 (-6.6%), decode 59.1/58.6 vs 57.9/58.4; probes 22/24, TV 0.054. READING: chunking is an available memory-for-prefill trade today (464 MiB per card for ~5-6% prefill); it is not free. NEXT, in order: (A) take the KQ mask and pooled-key mask inputs out of the arena - persistent worst-case tensors placed only on the attention devices with 1341's active mask - which removes 300 MiB from the arena on all devices and the re-plan churn with it; (B) check whether the dense KQ mask is needed at all with the sparse attention kernels (would also free the XTXs); (C) make the two computed masks subset-mirrored so the R9700 neither stores nor computes them. 1340 (MSM02) is the fallback, not the route.

## Change Log

- 2026-10-06T11:48:21.824495+00:00 (created-by): Created by agent
- 2026-10-06T12:36:47.707067+00:00 (updated-by): Updated: section:notes

## Ledger-events




- chg_20261006_144829_flash-next-uses-about-14-gb-l_2832
- 2026-10-06T14:48:32.643443+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-06T14:48:46.318179+00:00 (updated-by): Updated: section:notes
- 2026-10-06T19:37:35.322131+00:00 (updated-by): Updated: section:notes
- chg_20261006_215029_opt-in-the-context-sized-atte_9948
- 2026-10-06T21:50:32.670711+00:00 (updated-by): Updated: section:ledger-events
- chg_20261007_041129_opt-in-the-tensor-split-can-s_3694
- 2026-10-07T04:11:45.401067+00:00 (updated-by): Updated: section:ledger-events
- chg_20261007_133409_the-low-vram-tensor-split-layo_8043
- 2026-10-07T13:34:16.424953+00:00 (updated-by): Updated: section:ledger-events
