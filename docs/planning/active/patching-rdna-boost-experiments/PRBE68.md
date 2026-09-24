---
id: PRBE68
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:18.039731+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# VK-SUB-001: Architecture-aware Vulkan submission cap

## Description

TODO. No patch or upstream absorption. At b11126, ggml/src/ggml-vulkan/ggml-vulkan.cpp sets a fixed `device->max_nodes_per_submit = 100` default with only a manual `GGML_VK_MAX_NODES_PER_SUBMIT` env override -- there is no automatic architecture-aware cap today.

## Steps

1. Re-read upstream issue #26679's own data (fetch via `gh issue view 26679 --repo ggml-org/llama.cpp` or web) to see which exact architectures/drivers showed timeout risk and which showed the RDNA4 regression from over-conservative submission caps -- do not guess thresholds without this.
2. In the device-init block (ggml-vulkan.cpp, ~line 4058), replace the fixed default with a function of `device->architecture` / `device->vendor_id` / `device->driver_id`, keeping 100 (or lower, if #26679 shows a lower safe value) for GCN and any driver/architecture combination #26679 flags as timeout-prone, and only raising the default for RDNA4/gfx1201 if #26679's data shows the higher value is safe there.
3. If #26679 does not contain enough evidence to justify raising the default for any architecture, keep the default at 100 unconditionally and add only a debug log of the chosen value plus the architecture that selected it (telemetry-only path) -- state this explicitly if it is the outcome.
4. Add a unit test mocking device->architecture/vendor_id/driver_id combinations and asserting the selected cap.
5. Hardware: long-run stability test (no DeviceLost/timeout) on old AMD/GCN control, gfx1100, gfx1201, plus PP/TG and submission-count telemetry comparison against the current fixed-100 baseline.

## Detailed Solution & Technical Design

This must be evidence-driven from #26679, not guessed. The mechanism itself (reading device->architecture at the same init site that already reads driver_id/vendor_id for other AMD-specific decisions -- see the coopmat bypass at line ~1732 as a precedent for the same pattern) is straightforward; the risk is entirely in picking wrong numeric ceilings. If #26679 evidence is insufficient, ship the conservative-default-plus-telemetry version only (see step 3) rather than inventing a raised ceiling.

## Code Samples & Guidance

Anchor (verified present at b11126, ggml-vulkan.cpp, device init):
```cpp
        // Submit at least every 100 nodes, in case there are workloads without as much matmul.
        device->max_nodes_per_submit = 100;
        const char* GGML_VK_MAX_NODES_PER_SUBMIT = getenv("GGML_VK_MAX_NODES_PER_SUBMIT");
        if (GGML_VK_MAX_NODES_PER_SUBMIT != nullptr) {
            uint32_t max_nodes_per_submit = std::stoul(GGML_VK_MAX_NODES_PER_SUBMIT);
            device->max_nodes_per_submit = std::max(max_nodes_per_submit, 1u);
        }
```
Replacement sketch (conservative-plus-telemetry variant, use unless #26679 evidence justifies a numeric change):
```cpp
        device->max_nodes_per_submit = 100; // conservative default across all architectures pending #26679 evidence
        if (getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=<id> path=vk_submit_cap contract=VKSUB01 arch=%d driver=%d cap=%u\n",
                          (int) device->architecture, (int) device->driver_id, device->max_nodes_per_submit);
        }
        const char* GGML_VK_MAX_NODES_PER_SUBMIT = getenv("GGML_VK_MAX_NODES_PER_SUBMIT");
        if (GGML_VK_MAX_NODES_PER_SUBMIT != nullptr) {
            uint32_t max_nodes_per_submit = std::stoul(GGML_VK_MAX_NODES_PER_SUBMIT);
            device->max_nodes_per_submit = std::max(max_nodes_per_submit, 1u);
        }
```
Patch package sketch: `patches/12xx_vk_submission_cap_telemetry/patch.toml` (kind="enhancement", state="untested", backend="vulkan", experiment-contracts=["VKSUB01-SUBMISSION-CAP"]).

## Files

ggml/src/ggml-vulkan/ggml-vulkan.cpp; patches/12xx_vk_submission_cap_telemetry/{patch.toml,patch.py,SUMMARY.md}.

## Validation

PYTHONPATH=tools python -m bigcherry patch-lint; patch-rebase-check; offline mocked-device unit test; hardware long-run stability (no DeviceLost/timeout/corruption) plus PP/TG telemetry on old-AMD/gfx1100/gfx1201 via python -m bigcherry.patch.validation_campaign (not run here).

## Effort & Risk

M; depends on #26679 evidence quality -- if insufficient, ship telemetry-only (S effort, low risk).

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Require architecture-specific cap that prevents safety failures while avoiding material modern-GPU throughput regression; retain conservative fallback on unknown devices.

## Notes

2026-09-24 relevance at b11126: TODO confirmed, fixed default + manual env override only (verified via git show b11126:ggml/src/ggml-vulkan/ggml-vulkan.cpp ~line 4058). GPT design request: gateway rejected all submissions this session (VAL-AGW-025 / EXT-GPTAUTO-003); plan authored directly -- no GPT request id.

## Change Log

- 2026-09-09T10:58:18.039731+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:29.747464+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.434952+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.268296+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:18:48.866865+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031907_repaired-the-vulkan-submission_2651
- 2026-09-10T03:19:07.580884+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:32:24.544952+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
