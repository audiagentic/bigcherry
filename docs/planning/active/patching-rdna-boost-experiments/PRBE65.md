---
id: PRBE65
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:03.403872+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# VK-FA-001: Decouple Vulkan FA occupancy tuning from exact shared-memory capability equality

## Description

TODO. No patch or upstream absorption implements this. At b11126, ggml/src/ggml-vulkan/ggml-vulkan.cpp's FA occupancy heuristic gates on exact equality `maxComputeSharedMemorySize == 65536`, so it silently no-ops on real AMD devices reporting other totals (e.g. 32768). The separate hard legality check (`total_size <= maxComputeSharedMemorySize`) elsewhere is untouched and correct; this item only concerns the occupancy heuristic's trigger condition.

## Steps

1. Read the full FA config function in ggml/src/ggml-vulkan/ggml-vulkan.cpp containing the `== 65536` block (grep `limit_occupancy_shmem` to confirm the enclosing function name and full signature before editing).
2. Replace the exact-equality guard with a threshold/table keyed on the real reported `maxComputeSharedMemorySize`, e.g. treat any AMD device reporting >=32768 as eligible for a scaled occupancy target (scale the 26KiB/30KiB/14KiB constants proportionally to reported size relative to the 64KiB baseline the constants were tuned for), falling back to no occupancy limiting below a safety floor (e.g. <16KiB) rather than guessing.
3. Add a unit/offline test (Python or C++ harness under tools/tests or a new ggml-vulkan-specific test) that calls the FA config function (or a thin wrapper) with mocked `maxComputeSharedMemorySize` values of 65536, 32768, 16384, and confirms limit_occupancy_shmem scales and never exceeds the reported total.
4. Add a non-AMD vendor control (e.g. NVIDIA vendor id) confirming the branch is untouched.
5. Hardware validation: FA PP/TG before/after on an AMD device that already reports 65536 (must be unchanged/neutral -- regression control) plus any AMD device in the fleet reporting a different real total (if none exists, document that this step is blocked pending such hardware and is not claimed).

1. Read the full FA config function in ggml/src/ggml-vulkan/ggml-vulkan.cpp containing the `== 65536` block (grep `limit_occupancy_shmem` to confirm the enclosing function name and full signature before editing).
2. Replace the exact-equality guard with a threshold/table keyed on the real reported `maxComputeSharedMemorySize`, scaling the 26KiB/30KiB/14KiB constants proportionally to reported size relative to the 64KiB baseline; fall back to no occupancy limiting below a safety floor (<16KiB) rather than guessing. Define and record the EXACT expected `limit_occupancy_shmem` output for the documented test points before writing code, per GPT's fix: 64KiB -> unchanged current values (26/30/14 KiB as vec4-count, i.e. `/4/4`); 32KiB -> half those byte targets before the `/4/4` conversion (13/15/7 KiB); 16KiB -> floor triggers, no limiting applied (result.limit_occupancy_shmem left at its unset/default value); non-AMD vendor -> branch entirely untouched (result unset).
3. Add a unit/offline test (Python or C++ harness under tools/tests or a new ggml-vulkan-specific test) that calls the FA config function (or a thin wrapper) with mocked `maxComputeSharedMemorySize` values of 65536, 32768, 16384, and asserts the EXACT expected vec4-count values from step 2 (not just 'scales and never exceeds').
4. Add a non-AMD vendor control (e.g. NVIDIA vendor id) confirming the branch is untouched.
5. Hardware validation: FA PP/TG before/after on an AMD device that already reports 65536 (must be unchanged/neutral -- regression control) plus any AMD device in the fleet reporting a different real total (if none exists, document that this step is blocked pending such hardware and is not claimed).

## Detailed Solution & Technical Design

Data flow: `ggml_vk_get_flash_attn_config` (or equivalent; confirm exact name via grep 'limit_occupancy_shmem' in ggml-vulkan.cpp) computes `result.limit_occupancy_shmem` used later to inflate the shader's declared shared-memory usage so the driver schedules fewer subgroups per SIMD. The bug is that this is currently gated on `== 65536` verbatim. The fix generalizes to `>= <floor>` with size-proportional scaling, computed once from `device->properties.limits.maxComputeSharedMemorySize`, and must never push the computed occupancy shmem size past the value that the separate legality check (`total_size <= maxComputeSharedMemorySize`) would reject -- clamp explicitly.

## Code Samples & Guidance

Anchor (verified present at b11126, ggml/src/ggml-vulkan/ggml-vulkan.cpp, inside the FA config function):
```cpp
    if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize == 65536) {
        if (device->architecture != AMD_GCN && n_rows >= 64 && hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 due to smaller workgroup size
            // Values are guessed, tested on RDNA2
            result.limit_occupancy_shmem = (hsk <= 64 ? 26 : 30) * 1024 / 4 / 4;
        } else if (device->architecture == AMD_GCN && n_rows <= 8 && hsk >= 256) {
            // Same thing for GCN, with an occupancy target of 2 subgroups per SIMD.
            // Here low-batch FA with large head size is affected.
            // n_rows < 4 switch because workgroup size switches from 128 to 256 there.
            result.limit_occupancy_shmem = (n_rows < 4 ? 14 : 26) * 1024 / 4 / 4;
        }
    }
```
Replacement sketch:
```cpp
    const uint32_t amd_shmem_total = device->properties.limits.maxComputeSharedMemorySize;
    if (device->vendor_id == VK_VENDOR_ID_AMD && amd_shmem_total >= 16384) {
        const double shmem_scale = double(amd_shmem_total) / 65536.0;
        if (device->architecture != AMD_GCN && n_rows >= 64 && hsk <= 128) {
            result.limit_occupancy_shmem = uint32_t((hsk <= 64 ? 26 : 30) * 1024 * shmem_scale) / 4 / 4;
        } else if (device->architecture == AMD_GCN && n_rows <= 8 && hsk >= 256) {
            result.limit_occupancy_shmem = uint32_t((n_rows < 4 ? 14 : 26) * 1024 * shmem_scale) / 4 / 4;
        }
        result.limit_occupancy_shmem = std::min(result.limit_occupancy_shmem, amd_shmem_total / 4 / 4);
    }
```
Patch package sketch: `patches/12xx_vk_fa_occupancy_shmem_scale/patch.toml` (id=vk_fa_occupancy_shmem_scale, order in the 12xx-free range -- check `ls patches/ | sort` for the next free order before assigning, kind="enhancement", state="untested", backend="vulkan", experiment-contracts=["VKFA01-OCCUPANCY-SHMEM-SCALE"]); `patch.py` with one `FilePatch(path="ggml/src/ggml-vulkan/ggml-vulkan.cpp", edits=(Edit(id="vkfa-occupancy-scale", anchor=re.escape(<verbatim block above>), mode="replace", text=<replacement>, guard=r"shmem_scale"),))`.

## Files

ggml/src/ggml-vulkan/ggml-vulkan.cpp; patches/12xx_vk_fa_occupancy_shmem_scale/{patch.toml,patch.py,SUMMARY.md}; new offline mock test under tools/tests/.

## Validation

PYTHONPATH=tools python -m bigcherry patch-lint; patch-rebase-check --focal-overlay vk_fa_occupancy_shmem_scale --source bigcherry-tuning; offline mocked-device-trait unit test (step 3/4); hardware: FLASH_ATTN_EXT test-backend-ops parity plus llama-bench FA PP/TG on gfx1030/gfx1100/gfx1201 via python -m bigcherry.patch.validation_campaign (not run here).

## Effort & Risk

S; low blast radius (Vulkan-only, AMD-only branch), risk is picking wrong scale constants without a second real device at a different reported shmem size to validate against -- flag that gap explicitly if only 64KiB-reporting hardware is available.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Separate heuristic from legality, with no shader over-limit and repeatable occupancy benefit on affected devices; fallback on uncertain capability.

## Notes

Supersedes: RD82
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd82

2026-09-24 relevance at b11126: TODO confirmed, no existing patch and current code still hard-gates on == 65536 (verified via git show b11126:ggml/src/ggml-vulkan/ggml-vulkan.cpp). GPT design request: gateway rejected all submissions this session (VAL-AGW-025 / EXT-GPTAUTO-003, agent_task_gateway_overview); plan authored directly from verified source excerpt -- no GPT request id.

2026-09-24 GPT review req_d55aed71224e43a8 applied: NOT-READY -- added exact expected limit_occupancy_shmem values (vec4-count, /4/4 conversion) for 64/32/16KiB and non-AMD test points instead of leaving the numeric policy to the implementer.

## Change Log

- 2026-09-09T10:58:03.403872+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:11.754779+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.420235+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.247665+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:17:38.400709+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031805_repaired-four-more-graph-and-v_2834
- 2026-09-10T03:18:05.338956+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:32:14.639554+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T04:48:16.562912+00:00 (updated-by): Updated: section:steps
- 2026-09-24T04:48:22.409370+00:00 (updated-by): Updated: section:notes
