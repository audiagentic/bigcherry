---
id: PRBE69
order: 0
plan: patching-rdna-boost-experiments
state: deprecated
created-at: '2026-09-09T10:58:22.589852+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# VK-SUB-002: Bound startup submission-ramp growth by learned safe ceiling

## Description

OBSOLETE. PRBE69's premise is bounding an adaptive startup submission ramp by PRBE68's learned safe ceiling. At b11126, ggml/src/ggml-vulkan/ggml-vulkan.cpp has no adaptive/warmup submission ramp at all -- `device->max_nodes_per_submit` is a single fixed value (100, or the PRBE68-designed architecture-aware default) set once at device init and used unchanged thereafter. There is nothing to 'bound' because no ramp exists to grow past a ceiling.

## Steps

Use PRBE68 cap; instrument warmup/ramp growth over repeated long AMD runs; clamp adaptive ramp before it exceeds architecture ceiling; compare capped/uncapped and architectures without cap; verify no timeout/corruption and stable steady-state PP/TG.

## Detailed Solution & Technical Design

Ensure startup adaptation cannot grow beyond the architecture-specific safe submission cap. Preserve adaptive behavior below the cap and disable the cap only for architectures with no known safety requirement.

## Code Samples & Guidance



## Files

Vulkan submission ramp logic and cap integration; warmup/ramp tests; repeated long-run PP/TG and timeout evidence.

## Validation

No timeout/corruption; ramp trace and cap adherence; stable steady-state PP/TG versus baseline.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Keep only if cap prevents safety regressions without reducing normal throughput; adaptive ramp must never exceed PRBE68's safe ceiling on affected AMD.

## Notes

Supersedes: RD86
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd86

2026-09-24 relevance at b11126: OBSOLETE, premise does not hold. Verified via git show b11126:ggml/src/ggml-vulkan/ggml-vulkan.cpp -- device->max_nodes_per_submit is set once (device init) and never adapted/ramped over the run; no warmup-ramp code path exists anywhere in ggml-vulkan.cpp (grep for 'ramp'/'warmup' near max_nodes_per_submit returns nothing). If a genuinely new adaptive-ramp feature is wanted, that is new-feature scope, not a bound on existing behavior, and should be filed as a fresh item once PRBE68 ships and there is real motivation to add ramping (e.g. a measured cold-start cost that a static cap doesn't address) -- not assumed here. Superseded by PRBE68, which already covers the safety concern (timeout-prone older AMD) via a static architecture-aware cap without introducing new ramp complexity. GPT design request: gateway rejected all submissions this session (VAL-AGW-025 / EXT-GPTAUTO-003); disposition made directly from source inspection -- no GPT request id.

## Change Log

- 2026-09-09T10:58:22.589852+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:33.760959+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.440589+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.274152+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:18:55.271487+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031907_repaired-the-vulkan-submission_2651
- 2026-09-10T03:19:07.594642+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:32:36.978396+00:00 (updated-by): Updated: section:description, section:notes
- 2026-09-24T02:32:57.870595+00:00 (state-transition): State: pending → deprecated
