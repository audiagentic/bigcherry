---
id: PRBE62
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:49.294113+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# VK-DRV-002: AMD graphics queue for compute on proprietary Vulkan

## Description

TODO, narrower than originally scoped. Evaluate AMD proprietary-Vulkan graphics-queue dispatch for MoE decode. Relevance at b11126: a related opt-in already exists -- `GGML_VK_ALLOW_GRAPHICS_QUEUE` env var (ggml-vulkan.cpp:4084-4087) lets the compute-queue-family search also accept VK_QUEUE_GRAPHICS_BIT-capable families, but it is a blanket allow-flag with no workload (MoE vs dense) or driver (proprietary vs RADV) conditioning and no per-graph queue selection -- so the low-level plumbing exists but the workload/driver-conditioned selector this item requires does not. Scope narrows to: build the conditioned selector on top of the existing flag rather than inventing queue-family discovery from scratch.

## Steps

1. Run the mandatory anchor-discovery greps before writing any Edit: `git -C work/upstream/llama.cpp.git grep -n -E 'compute_queue|VK_QUEUE_COMPUTE_BIT|VK_QUEUE_GRAPHICS_BIT|queue_family_index|VkDeviceQueueCreateInfo|vkGetDeviceQueue|vkQueueSubmit|vkCreateCommandPool' b11126 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `... -E 'driverID|VkPhysicalDeviceDriverProperties|VK_DRIVER_ID_AMD_PROPRIETARY|VK_DRIVER_ID_MESA_RADV|vendorID' ...`, `... -E 'GGML_OP_MUL_MAT_ID|mul_mat_id' ... tests/test-backend-ops.cpp`, and `... -E 'VkBufferCreateInfo|sharingMode|queueFamilyIndexCount|pQueueFamilyIndices' -- ggml/src/ggml-vulkan`. Also re-confirm the existing flag: `git -C work/upstream/llama.cpp.git grep -n GGML_VK_ALLOW_GRAPHICS_QUEUE b11126 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp` (already confirmed present at ggml-vulkan.cpp:4084-4087).
2. Do not finalize any Edit anchor until queue-family selection, device-queue creation, command-pool/submission, driver-properties query, and buffer-sharing blocks are pasted from step 1's real output.
3. Implement `bigcherry_vk_use_moe_graphics_queue(device, graph)`: true only when BIGCHERRY_VK_MOE_GRAPHICS_QUEUE=1 (new, narrower flag layered on top of the existing GGML_VK_ALLOW_GRAPHICS_QUEUE plumbing -- do not just reuse the broad upstream flag for this workload-specific decision), device is AMD proprietary driver (VK_DRIVER_ID_AMD_PROPRIETARY or the exact equivalent b11126 exposes; RADV explicitly excluded), an alternate queue family supports both GRAPHICS and COMPUTE bits, and the graph is MoE-decode-shaped (contains GGML_OP_MUL_MAT_ID at a decode-sized token/batch dimension, e.g. <=8 -- exact field to be confirmed from step 1's MUL_MAT_ID grep). Dense decode/prefill and large-N MoE prefill graphs must stay on compute queue.
4. Wire queue selection at graph-submission granularity (one lane per whole graph, not per-kernel, to avoid extra cross-queue sync). Default/env-unset path must be byte-for-byte the current compute-family selection.
5. Handle buffer ownership: if the feature is enabled, backend buffers must be safely shared across the two selected queue families (concurrent sharing mode) unless the real b11126 allocator already provides equivalent handling -- confirmed by step 1's VkBufferCreateInfo/sharingMode grep. Never submit an exclusive-family buffer on both families.
6. Add the activation marker (BIGCHERRY_PATCH_HIT patch=1263_prbe62 path=amd_moe_graphics_queue family=<index>) at the real graphics-queue submission point (not just init), gated by BIGCHERRY_PATCH_TRACE, once per process.
7. Tests: MUL_MAT_ID backend-op case at decode width with env on (expect reference correctness + graphics marker); ordinary MUL_MAT decode case (no marker); MUL_MAT_ID prefill-sized case (compute path, no marker); RADV negative control if available (env on but compute fallback); temp-0 identity for a representative MoE model with env off vs on (identical token IDs); dense temp-0 control must also stay identical and never trigger the marker.

## Detailed Solution & Technical Design

Layer a workload+driver-conditioned selector on top of the ALREADY-UPSTREAM GGML_VK_ALLOW_GRAPHICS_QUEUE opt-in (ggml-vulkan.cpp:4084-4087) rather than inventing queue-family discovery from scratch. This item's own acceptance criteria explicitly forbid a global default -- the upstream flag alone does not satisfy that (it's all-or-nothing across all ops), so this remains TODO, but implementation is smaller than initially scoped since the graphics-queue-family search plumbing exists. Real anchors for queue creation, driver-ID query, and buffer sharing must be pasted from step 1 before Edit() anchors are finalized.

## Code Samples & Guidance

patches/1263_prbe62_amdvlk_moe_graphics_queue/patch.toml:
```toml
schema = 1
id = "1263_prbe62_amdvlk_moe_graphics_queue"
order = 1263
state = "untested"
kind = "enhancement"
origin = "local"
backend = "vulkan"
plan-ids = ["PRBE62"]
requires = []
conflicts = []
requires-options = []
forbids-options = []
subsystems = ["vulkan", "queue-dispatch", "moe"]
hardware = ["amd"]
validation-architectures = []
backends = ["vulkan"]
```
patches/1263_prbe62_amdvlk_moe_graphics_queue/patch.py (skeleton -- TODO-VERIFY anchors require step-1 pasted text):
```python
from bigcherry.patcher import Edit, FilePatch

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-vulkan/ggml-vulkan.cpp",
        description="PRBE62 opt-in proprietary-AMD MoE graphics-queue routing",
        edits=(
            Edit(
                id="prbe62-driver-workload-predicate",
                anchor=r"<TODO-VERIFY: real device/queue helper anchor, near existing GGML_VK_ALLOW_GRAPHICS_QUEUE at ~line 4084>",
                rationale="add AMD-proprietary + MoE-decode qualification predicate layered on the existing graphics-queue-allow plumbing",
                mode="insert_before",
                text=r"<TODO-VERIFY: bigcherry_vk_use_moe_graphics_queue() using verified b11126 device/graph types>",
                guard=r"bigcherry_vk_use_moe_graphics_queue",
            ),
            Edit(
                id="prbe62-dual-queue-init",
                anchor=r"<TODO-VERIFY: real queue-family/device-queue creation block>",
                rationale="retain existing compute queue selection; add optional graphics+compute lane only under the new narrower flag",
                mode="replace",
                text=r"<TODO-VERIFY: dual-queue init preserving default-off behavior>",
                guard=r"BIGCHERRY_VK_MOE_GRAPHICS_QUEUE",
            ),
            Edit(
                id="prbe62-queue-submit-select",
                anchor=r"<TODO-VERIFY: real submission block>",
                rationale="select graphics lane only for proprietary-AMD MoE decode graphs; emit activation evidence at submission",
                mode="replace",
                text=r"<TODO-VERIFY: graph-level queue selection + submit + marker>",
                guard=r"BIGCHERRY_PATCH_HIT patch=1263_prbe62",
            ),
            Edit(
                id="prbe62-buffer-family-sharing",
                anchor=r"<TODO-VERIFY: buffer-create block, only if needed per step-1 sharingMode grep>",
                rationale="permit safe compute/graphics family alternation only while the experimental flag is enabled",
                mode="replace",
                text=r"<TODO-VERIFY: concurrent queue-family buffer setup>",
                guard=r"<unique prbe62 buffer-sharing guard>",
            ),
        ),
    ),
]
```

## Files

ggml/src/ggml-vulkan/ggml-vulkan.cpp; possibly a separate buffer-allocation source if VkBufferCreateInfo lives elsewhere (confirm in step 1); tests/test-backend-ops.cpp; patches/1263_prbe62_amdvlk_moe_graphics_queue/{patch.toml,patch.py,SUMMARY.md,validation/producer.py}

## Validation

Offline: `PYTHONPATH=tools python -m bigcherry patch-lint`, `patch-rebase-check --focal-overlay 1263_prbe62_amdvlk_moe_graphics_queue --source bigcherry-tuning`. Unit: the 5 backend-op/control cases in step 7. Hardware (Brutus, not run here): proprietary-AMD-driver R9700, MoE + dense Qwen decode/prefill, output parity, TG/PP and queue-utilization split by workload/driver, via `python -m bigcherry.patch.validation_campaign`.

## Effort & Risk

M / medium-high -- touches Vulkan device/queue init and buffer-sharing semantics; risk contained by opt-in flag, RADV/dense fallback to unchanged compute path, and per-graph (not per-kernel) queue selection.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Never enable globally; retain only a proven workload/driver-specific selector with repeatable benefit and no dense regression.

## Notes

Supersedes: RD79
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd79

2026-09-24 relevance at b11126: GGML_VK_ALLOW_GRAPHICS_QUEUE opt-in flag already upstream (ggml-vulkan.cpp:4084-4087, confirmed via grep) but is a blanket allow with no workload/driver conditioning -- item's own acceptance criteria (never enable globally) means this stays TODO, narrowed to build the conditioned selector on existing plumbing. GPT design request req_63a12ebff6544a0a (batched with PRBE55).

## Change Log

- 2026-09-09T10:57:49.294113+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:59.947699+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.407607+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.225741+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:16:25.617635+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031644_repaired-four-more-active-succ_8062
- 2026-09-10T03:16:44.974520+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:30:37.583971+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
