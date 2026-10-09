---
id: PRVP01
order: 0
plan: patching-rocm-vulkan-provider
state: completed
created-at: '2026-09-09T11:01:08.382625+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Retired: CM1 source transplant already native at b11474

## Description

The former PR #27952 source-capture/1246 transplant is obsolete. Upstream merged #27952 on 2026-09-24 (merge 70c4e1582e37e4fd94104eb09301711a0f2675bc), which is an ancestor of BigCherry's pinned llama.cpp b11474 (b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea; compare ahead 314, behind 0). The pinned Vulkan implementation already contains the int8 CM1 shader, host dispatch, architecture enum, generator, and shared quant functions. PR #27952 changed FIVE files, not the four assumed in this plan: ggml-vulkan-types.h is also coupled.

There is no patches/1246_ro19_vulkan_cm1_pr27952 package in the selected BigCherry tree. Do not create a backport of already-pinned native code, add a second shader implementation, or change source registry identity for a nonexistent package.

## Steps

1. Terminal source-equivalence decision: native CM1 at the pin supersedes 1246 packaging.
2. Preserve the historical reviewed PR head (965e5710...) only as provenance, not as the current merged identity; the final merged head was 5cdaca76....
3. Delegate remaining A-prefetch correctness and pinned route checks to PRVP02; TRVP14 owns optional strict route/telemetry, TRVP15 owns later measured promotion. RRVP02 records the Vulkan implementation pause.
4. Reopen only if a future pin loses the native change or an exact diff proves a new missing mechanism. Verify ancestry and source paths first.

## Files

Native pinned files: ggml/src/ggml-vulkan/{ggml-vulkan.cpp,ggml-vulkan-types.h,vulkan-shaders/mul_mmq_cm1.comp,vulkan-shaders/mul_mmq_cm1_funcs.glsl,vulkan-shaders/vulkan-shaders-gen.cpp}. No BigCherry 1246 patch exists or is required.

## Validation

Source-only: verified merged-commit ancestry, all five source files, and absence of 1246 in the patch catalog/tree. No compilation, Vulkan execution, or hardware measurements occurred in this audit.

## Acceptance Criteria

Completed/superseded: no duplicate transplant, no stale four-file capture claim, and a single native CM1 qualification owner (PRVP02). Any future source drift is handled through the existing pin/source provenance process.

## Notes

2026-10-08T22:03Z audit: upstream #27952 merged 2026-09-24; native pinned implementation and 13-quant shader-generator branch inspected. PRVP02 remains pending for a separate shader A-prefetch safety question; PRVP01 is not its implementation owner.

## Change Log

- 2026-09-09T11:01:08.382627+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:18:04.972854+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.592969+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.455637+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T01:03:20.062956+00:00 (updated-by): Updated: section:validation
- chg_20260910_010342_successor-plans-now-have-expli_8662
- 2026-09-10T01:03:42.607862+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:23:00.294061+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032340_repaired-the-cm1-source-and-in_5642
- 2026-09-10T03:23:40.354015+00:00 (updated-by): Updated: section:ledger-events
