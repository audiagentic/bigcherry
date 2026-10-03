---
id: PRBE46
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:41.470034+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-VK-001: MoE density-aware Vulkan MMV routing

## Description

TODO, corrected per GPT review (req_e17e0bf5a68c48d5). The proposed 'expert density from src2 directly' heuristic is impossible at this host routing point without a GPU readback/synchronization -- src2 exposes shape metadata only (ne[0]/ne[1]), not runtime expert-ID VALUES (which live in GPU memory and are not host-visible without a sync). The plan's own sketch also incorrectly referenced `ctx->device`, but ggml_vk_use_mul_mat_vec_id(const ggml_cgraph*, int) has no ctx parameter (confirmed at b11126 ggml-vulkan.cpp:7670 -- signature is unchanged, still just (cgraph, node_idx)). Threshold remains undefined pending hardware evidence.

## Steps

1. Replace the invalid 'read src2's runtime expert-ID values' approach with a METADATA-ONLY heuristic using shape data already host-visible without synchronization: src0->ne[2] (number of experts), src2->ne[0] (n_expert_used), src2->ne[1] (batch/token count) -- these are static graph-shape values, not GPU-resident routing data. 2. Keep the function signature as-is: `bool ggml_vk_use_mul_mat_vec_id(const struct ggml_cgraph * cgraph, int node_idx)` -- do NOT add a ctx parameter (no ctx is available or needed at this call site; device/arch info, if needed for the guard, must come from a different existing accessor already used elsewhere in this file, to be identified by reading the surrounding dispatch code). 3. Keep batch<=8 as the unconditional fast path (byte-for-byte unchanged). 4. Add the density-aware extension for batch 9..64 using only the metadata-only signals above (e.g. n_expert_used/n_expert ratio as a static density proxy, since per-token actual expert IDs are not available without sync). 5. Freeze the exact threshold from hardware evidence (Brutus MoE runs at -np 8/9/16/32/64) before finalizing -- this remains a hardware-validation-driven constant, now on a metadata-only formula rather than the previously-invalid runtime-readback approach.

## Detailed Solution & Technical Design

The routing decision lives entirely in one function, ggml_vk_use_mul_mat_vec_id, called from ggml_vk_mul_mat_id (ggml-vulkan.cpp:7683) which then dispatches either ggml_vk_mul_mat_vec_id_q_f16 (MMV path) or ggml_vk_mul_mat_id_q_f16 (full path). Because src2 (the expert-id tensor, `dst->src[2]`) is already available at the decision point, expert density (how concentrated token->expert routing is, vs uniformly spread across all n_expert_used experts) can be computed from it directly without threading new data through the call chain -- this keeps the change localized to one function body plus a driver/arch guard, avoiding new op_params or graph-shape changes. The density heuristic itself (e.g. "if the number of DISTINCT experts touched across this batch's tokens is <= K for some K scaled by batch size" vs "experts nearly saturated") needs empirical tuning on real Qwen MoE traffic (-np 8/9/16/32/64) before its exact formula is finalized -- this plan specifies the mechanism and hook point precisely but the density threshold itself is a hardware-validation-driven constant, not something to hardcode from a guess.

## Code Samples & Guidance

Real b11126 anchor (ggml/src/ggml-vulkan/ggml-vulkan.cpp:7670-7674, verified exact):\n```cpp\nbool ggml_vk_use_mul_mat_vec_id(const struct ggml_cgraph * cgraph, int node_idx) {\n    ggml_tensor * dst = cgraph->nodes[node_idx];\n    ggml_tensor * src0 = dst->src[0];\n    ggml_tensor * src2 = dst->src[2];\n    return (src2->ne[1] <= 8) && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));\n}\n```\nReplacement sketch (density check placeholder, threshold TBD from hardware evidence):\n```cpp\nbool ggml_vk_use_mul_mat_vec_id(const struct ggml_cgraph * cgraph, int node_idx) {\n    ggml_tensor * dst  = cgraph->nodes[node_idx];\n    ggml_tensor * src0 = dst->src[0];\n    ggml_tensor * src2 = dst->src[2]; // [n_expert_used, n_tokens] expert ids\n    const bool type_ok = src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type);\n    if (!type_ok) return false;\n    if (src2->ne[1] <= 8) return true; // unchanged fast path, zero regression risk\n    if (src2->ne[1] > 64) return false; // outside the density-routing envelope this item covers\n    if (!ggml_vk_is_radv_rdna3_or_later(ctx->device)) return false; // reuse this file's existing arch-guard pattern\n    return ggml_vk_moe_density_favors_mmv(dst, src2); // NEW: density heuristic over src2's expert ids\n}\n```\npatch.toml: id="<order>_rd55_vk_moe_density_routing", state="untested", backend="vulkan", plan-item="PRBE46", experiment-contracts=["UP-VK-001-MOE-DENSITY-MMV-ROUTING"].

## Files

ggml/src/ggml-vulkan/ggml-vulkan.cpp (ggml_vk_use_mul_mat_vec_id and its call site ggml_vk_mul_mat_id), tests/test-backend-ops.cpp (MUL_MAT_ID parity at B=9..64), new package patches/<order>_rd55_vk_moe_density_routing/.

## Validation

Offline: patch-lint, patch-rebase-check. Correctness: MUL_MAT_ID output parity between MMV and full paths at every density/batch tested. Hardware (Brutus, not run here): Qwen MoE -np 8/9/16/32/64 on XTX/R9700 RADV, B<=8 controls (must be bit-identical to pre-patch, since that path is unchanged), PP512 and dense-model controls, explicit B=8 vs B=9 boundary comparison, TG aggregate/kernel timing/per-request latency across the target driver matrix.

## Effort & Risk

M (unchanged from item estimate) -- real risk is overfitting the density threshold to one driver/workload; mitigated by the explicit driver/arch guard keeping non-target hardware on the unchanged fixed cutoff.

## Standards

Preserve Vulkan fallback (B<=8 path byte-for-byte unchanged), driver/architecture qualification before promotion, campaign evidence provenance.

## Acceptance Criteria

Acceptance requires output parity, explicit B=8/B=9 boundary coverage, density/driver/architecture controls, and a repeatable target-driver latency/TG benefit without regressing other batches; otherwise retain the existing routing.

## Notes

Supersedes: RD55
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd55

Supersedes RD55; coordinate with reusable-build Vulkan scoping without duplicating ownership.

2026-09-24 relevance at b11126: no existing patch (RD55 grep = no hits). Not upstream-absorbed -- ggml_vk_use_mul_mat_vec_id at ggml-vulkan.cpp:7670 read directly and confirmed to still use the plain `src2->ne[1] <= 8` cutoff with no density awareness. GPT design requests this batch (req_6a45d0917b114112, req_3710dbae73fb49be) were unusable; plan authored directly from verified source. Coordinate with reusable-build Vulkan scoping per the item's original notes.

2026-09-24 GPT review req_e17e0bf5a68c48d5 applied: replaced the invalid runtime-expert-ID-readback heuristic (would require GPU sync) with a metadata-only heuristic (src0->ne[2], src2->ne[0]/ne[1]); corrected the sketch's erroneous ctx parameter -- confirmed ggml_vk_use_mul_mat_vec_id(cgraph, node_idx) at b11126 ggml-vulkan.cpp:7670 takes no ctx argument.

## Change Log

- 2026-09-09T10:56:41.470034+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:54.998308+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.337328+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.109221+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:08:34.398338+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-10T03:09:12.902174+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030930_repaired-two-more-active-patch_7368
- 2026-09-10T03:09:30.729645+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:46:23.047896+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T05:08:20.197102+00:00 (updated-by): Updated: section:description, section:steps, section:notes
