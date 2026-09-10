---
id: TRVP14
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:01:16.012263+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# Expose CM1 as a forceable Vulkan recipe candidate

## Description

Expose the integrated CM1 path as a forceable Vulkan recipe candidate with complete recipe identity, strict fail-loud selection, truthful route telemetry, and correctness before timing.

## Steps

1. Define a complete CM1 recipe covering activation preparation/Q8_1, main CM1 pipeline, and optional split-K/reduction. 2. Add identity for operation, quant, accumulator, S/M/L/tile class, subgroup, device/architecture, driver/ICD, shader digest, upstream/vendor/build identity, dispatch signature, and recipe stages. 3. Implement Vulkan-native can_execute and route selection; keep it separate from generic dispatch and create pipelines outside timed regions. 4. Expose strict force through TRBC03 and emit requested route, actual bound route, rejection/fallback reason, signature, candidate/recipe ID, and shader digest. 5. Compare native versus strict CM1 for MUL_MAT and MUL_MAT_ID before enabling measurement. 6. Unsupported strict force fails explicitly; ordinary auto/replay misses and device/timestamp failures fall back visibly to native. Keep patch 1247 package-only and independently identified from 1246.

## Detailed Solution & Technical Design

A CM1 candidate is a complete Vulkan pipeline recipe, not a shader pointer or HIP launcher. Preserve descriptors, push constants, scratch, reduction, and preparation state; route telemetry is emitted only after authoritative pipeline binding. Separate can_execute from auto_eligible. Strict force is actual CM1 or explicit cannot_execute; auto/replay miss is native plus fallback reason. Diagnostics are non-promotable until qualification, and the Vulkan ABI remains independent of HIP.

## Code Samples & Guidance



## Files

src/ggml/src/ggml-vulkan/bigcherry/vk-autotune-dispatch.*; vk-autotune-record.*; vk-autotune-types.h; candidate registry/recipe and correctness modules; tools/bigcherry/telemetry.py; dispatch/recipe/safety/correctness tests; patches/1247_ro21_vulkan_cm1_force_telemetry/patch.toml, patch.py, SUMMARY.md.

## Validation

Force supported CM1 and prove actual route. Force unknown/ineligible candidates and prove explicit rejection. Compare native/CM1 outputs for S/M/L, tails, aligned/unaligned, multiple K, split/non-split, batched/broadcast, and MUL_MAT_ID expert routing/row IDs/subgroup ballots. Test native mode, replay miss, pipeline creation failure, device-lost/timestamp failure, bounded telemetry, and no false-positive fallback reports. Use control/subject compositions and patch-validation workflow for any real validation; route logs alone are not evidence.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Stable complete-recipe identity exists. Supported strict force proves CM1 executed; unsupported force fails loudly. Auto/replay misses fall back visibly to native and telemetry reports actual bound route. Correctness passes before timing/promotion. Patch 1247 is package-only, independently identified, and has no accidental multi-contract binding.

## Notes

Supersedes: RO21
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro21

Supersedes RO21. Depends on RE30/RE32/RE36 and TRBC01/TRBC03; preserve CM1 non-promotable status until TRVP15. Do not copy HIP ABI or bypass governance.

## Change Log

- 2026-09-09T11:01:16.012263+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:18:13.795384+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.601183+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.469352+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:46:56.478251+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034835_repaired-the-final-six-live-su_3009
- 2026-09-10T03:48:35.249744+00:00 (updated-by): Updated: section:ledger-events
