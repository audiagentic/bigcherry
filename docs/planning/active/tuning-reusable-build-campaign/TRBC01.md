---
id: TRBC01
order: 0
plan: tuning-reusable-build-campaign
state: pending
created-at: '2026-09-09T11:01:42.669987+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
---

# Evaluate adding Vulkan as a tunable build line (patchable, campaign/tune/replay-capable like HIP)

## Description

Evaluate and stage a Vulkan-native tunable build line: stock/patch validation first, then bounded MUL_MAT record→tune→replay without widening HIP abstractions.

## Steps

Preserve completed stock Vulkan lane and backend-aware audit; implement Vulkan-native canonical signature/device-driver-shader provenance and native pipeline record; add explicit complete pipeline candidates after TRBC02 capability ABI; add timestamp tuning transaction, correctness/warmup/winner selection/cache; add backend-namespaced replay with mismatch/native fallback; keep HIP unchanged.

## Detailed Solution & Technical Design

Vulkan candidate is a complete preparation+main+reduction pipeline recipe. Hook immediately before stock MUL_MAT recipe choice, not generic dispatch. Use vk-autotune subtree and separate schemas; native fallback on decline/miss.

## Code Samples & Guidance



## Files

campaign/build backend plumbing; ggml-vulkan vk-autotune types/record/tune/replay; Vulkan audit/tests/manifests.

## Validation

Config/backend isolation, pristine patch/build, real stock smoke; record parser-clean canonical observations, candidates reject unsupported limits, timestamp measurement, replay identity/fallback.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Stock lane remains green and first Vulkan MUL_MAT vertical slice has complete provenance, correctness-first tuning, backend-namespaced replay, and no HIP behavior change.

## Notes

Supersedes: RE30
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-reusable-build-campaign-re30

## Change Log

- 2026-09-09T11:01:42.669987+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:58.867384+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.466793+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.312772+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:39:31.943753+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_034017_repaired-six-vulkanprovider-s_6655
- 2026-09-10T03:40:17.953866+00:00 (updated-by): Updated: section:ledger-events
