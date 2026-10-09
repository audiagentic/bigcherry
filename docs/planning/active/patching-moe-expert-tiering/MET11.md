---
id: MET11
order: 11
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-06T16:42:10.474116+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Load-time expert placement from a usage profile (no reordered model file)

## Description

MET08's usage-placed layout currently needs a second copy of the model in which each layer's experts are physically reordered (tools/lab/flash-next/expert-place.py, 88 GB for UD-IQ4_XS). That was the fastest way to test the idea, not the design to keep. This item moves the placement to load time: the original model file is used unchanged, and a usage profile plus per-card capacities decide which experts each card holds. Owner question 2026-10-07: 'could we not do that at runtime when loading with a profile mapping' - yes.

## Steps

1. Factor the placement rule out of expert-place.py into something both the tool and the C++ side follow exactly (document the rule; reuse the tool's existing --plan-only JSON as the permutation oracle, with no second CLI switch). 2. Patch: profile reader (reuse 1338's STRP reader code shape), permutation builder, permuted block copy at load for the three expert tensors, B2 id translation in the qwen4exp MoE graph (after top-k, before the three range MUL_MAT_ID nodes and any 1336 copy), log line with the per-card hot / cold counts, BIGCHERRY_PATCH_HIT marker. 3. Offline mechanics test (apply on 1281 + 1283, idempotent, fail closed, flag-off path unchanged). 4. Hardware, equivalence first: with the SAME profile and capacities, the load-time placement on the ORIGINAL file must give the same per-card expert sets as the reordered file (compare the logged permutation with expert-placement.json) and, with B1 semantics emulated or with B2, probes inside the envelope against the reordered-file run; with B2 the row split on the original file is untouched by construction. 5. Hardware, layout: the MET08 comparisons on the original file (prefill / decode at 8K, 98K, 202K; ctx 245760; acceptance; fidelity). 6. Load time with and without the placement (reads are no longer sequential). 7. If adopted: delete gguf/placed-128-128-256 and retire expert-place.py's write mode (keep it as a simulator).

## Detailed Solution & Technical Design

New opt-in patch package (ID allocated only after admission; 1343 is already occupied by superseded 1343_mtp_nextn_rereserve), requires 1283 (expert split) and 1281. Entirely opt-in: BIGCHERRY_MOE_EP_PLACEMENT=<profile file> switches it on; unset = today's behaviour, byte-identical (the loader takes the existing path). It only has an effect together with BIGCHERRY_MOE_EP=1. Inputs: (1) the profile in the Strata STRP format that 1338 already reads and writes (ranked (layer, expert) pairs); (2) the per-card expert counts from BIGCHERRY_MOE_EP_TS (e.g. 128,128,256); (3) BIGCHERRY_MOE_EP_TRAFFIC (default equal) = share of the hot experts per card. From these the same rule as expert-place.py builds, per layer, a permutation new_index -> old_expert (hot experts dealt across the cards by weighted round robin until a card is full, cold remainder on the cards with room). MECHANISM, two parts. (A) Weights: when a layer's ffn_gate_exps / ffn_up_exps / ffn_down_exps are loaded, the expert blocks are written to the device buffers in permuted order. Expert blocks are whole byte ranges of a quantised tensor (n_bytes / n_expert each), so this is a block copy with no requantisation; these tensors are copied to VRAM anyway, so it changes the read order, not the amount of I/O. Needs the loader's tensor-data path for split tensors (llama_model_loader::load_all_data -> ggml_backend_tensor_set on the meta buffer, which splits along the expert axis) to read through the permutation. (B) Router consistency, choose one: (B1) permute the rows of ffn_gate_inp.weight the same way at load - the selected ids are then already in the new order, no run-time cost, identical to what the reordered file does; or (B2) leave the router untouched and translate the selected ids after top-k through a per-layer old->new table (one small gather per layer per graph) - selection, routing weights and tie-breaking stay exactly those of the original model. Recommended: B2, because the GPU top-k has no expert-id tie-break (MET09), so B1 can change which of two equal-scored experts is selected, while B2 cannot; the cost of B2 is one I32 gather of n_tokens * top_k elements per layer. The table is a small MIRRORED tensor per layer.

## Code Samples & Guidance



## Files

new engines/llamacpp/patches/<allocated_moe_placement_id>/ (patch.py, patch.toml, SUMMARY.md), tools/tests/patch/test_<allocated_moe_placement_id>.py; vendor files touched by the patch: src/llama-model-loader.cpp (permuted load), src/models/qwen4exp.cpp or src/llama-graph.cpp (id translation), src/llama-model.cpp (flag, profile, per-layer tables); tools/lab/flash-next/expert-place.py (--plan-only; existing)

## Validation

Flag unset: identical to the current build (greedy md5 and speed). Flag set: permutation equals the tool's for the same inputs; fidelity inside the envelope; MET08's speed gates; no extra VRAM beyond the per-layer id tables; load time reported. Lightweight tier for promotion as an opt-in enabler.

## Effort & Risk

Medium. The delicate part is the loader path for split tensors (mmap vs read, the meta buffer's per-device slicing along the expert axis) and making sure all three expert tensors and the id translation use the same table. Wrong tables give plausible but wrong output, so step 4's equivalence check against the reordered file is mandatory before any speed run.

## Standards

Package-only patch, fail-closed anchors, offline test. Off switch = the variable unset. No second permanent mechanism: once this works the reordered file is a test artefact to delete.

## Acceptance Criteria



## Notes

Also the base for MET10 (host tail): the same per-layer table decides which experts are resident and which stay on the host. Profiles come from 1338's recorder (BIGCHERRY_MOE_CACHE_PROFILE_OUT, with BIGCHERRY_MOE_CACHE_LARGE=1 to include prefill routing); a recorder that works inside the tensor split, not only on a single card, would remove that detour and belongs here or in MET08.

## 2026-10-10 PNRO16 consolidation and implementation admission

PNRO16's proposed separate GGUF inventory, placement compiler and routing replay are **closed as duplicates**. `tools/lab/flash-next/expert-place.py::place` and its existing `--plan-only` receipt are the sole offline permutation oracle; `tools/lab/strata/routing-balance.py` is the existing per-token routing replay. `MET08` owns workload comparisons; `MET09` owns split/order numerical differences; `MET07`/1337/1338 own cache/profile; `RPL01` owns topology-level cost modelling. Do not add a new placement registry, compiler, scheduler, cache, or profiling subsystem. The old proposed patch number 1343 collides with `1343_mtp_nextn_rereserve` (superseded but still reserved): allocate an unused package ID only after the gate below passes.

**Concrete preflight, before a loader edit:** validate `caps` are nonnegative integers, exactly one per visible eligible device, and sum to the actual model expert count; `traffic` has the same length and consists of finite strictly positive numbers (reject NaN/Inf); validate STRP magic/version/length, layer/expert bounds, duplicates, and a complete per-layer permutation and inverse. `expert-place.py` currently checks the sum and `min(traffic)<=0`, which admits `caps=[-1,7]` for six experts and NaN traffic. Add negative host fixtures to the existing owner tool's tests, not a second compiler. For the real model, check the GGUF shard set, tensor names, types, shapes, packed expert-block byte strides and profile provenance; `--plan-only` currently accepts no GGUF path and cannot prove any of those. Bind a recorded profile hash and model/shard manifest to the qualification receipt; a metadata-only fingerprint cannot prove identical weight bytes. No automatic placement on a mismatched/unknown receipt.

**Performance model and routing contract:** STRP ranks pairs but does not retain their frequencies. The current `place(ranked,caps,traffic)` is smooth weighted round-robin over ranked experts, **not** a cost-optimal weighted bin-packer. A six-expert host counterexample with weights [100,90,80,70,1,1] and 3/3 slots yields loads 181/161, versus 171/171 for a separately calculated weighted-greedy control. These are synthetic service counts, not BigCherry performance measurements. Reuse real Strata routing traces to compute selected-expert work per device and existing VRAM/fit measurements for each 128/128/256-like candidate; do not infer GPU service time from expert count alone. vLLM EPLB's weighted `balanced_packing` and SGLang EPLB provide a mechanism reference, not an RDNA drop-in.

**Exact load-time data flow:** `GGUFReader`/loader metadata -> validated `old_to_new[layer][expert]` and `new_to_old` tables -> expert-block permutation of gate/up/down at `llama_model_loader::load_all_data`/Meta split AXIS_2, preserving quantized byte blocks -> keep original router rows -> translate only post-top-k I32 IDs before all three 1281 range MUL_MAT_ID nodes -> existing 1283 partial/local GLU/down and one collective. Tables must remain immutable and graph-resident for the owning model/context lifetime, with correct backend placement and no host-pointer capture. Preserve all selected weights and tie-breaking; ensure all three expert tensors and id mapping use the same permutation. Reject unsupported merged gate_up, expert bias/scale layouts, shard mismatch, split mode or fit failure **before** enabling the path; flag-off remains byte-identical. Never silently switch to the reordered-file semantics.

**Cheapest gate:** host fixtures for the negative inputs above, per-layer bijection, packed block offsets and inverse IDs; compare existing `--plan-only` JSON with the proposed C++ mapping on a real model without allocating a second full GGUF copy. Then an opt-in, default-off prototype only if MET08's fixed-work/per-device routing trace shows a >=3% plausible E2E gain after transfer, load-time and memory costs. Validate multi-request/multi-ubatch, graph replay/reallocation, full-vocab/greedy/KLD, MTP acceptance, expected-vs-observed expert/collective work and VRAM at 8K/98K/202K/245760. Promotion requires >=4 independent sessions x >=10 paired ABBA rounds, CI95-low >=3% E2E and <=1% control regression. Close without implementation if no fit-safe profitable candidate survives. No new GPU work was run for this audit.

Sources: llama.cpp PRs #29887 (merged 2026-10-07), #29963 (open as of 2026-10-09); vLLM `vllm/distributed/eplb/policy/default.py`; SGLang `python/sglang/srt/eplb/expert_distribution.py`. These are mechanism comparisons only.

## Change Log

- 2026-10-06T16:42:10.474116+00:00 (created-by): Created by agent
