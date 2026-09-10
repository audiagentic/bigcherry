---
id: PRVP02
order: 0
plan: patching-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:01:11.838320+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# Integrate CM1 shader generation and Vulkan capability gating

## Description

Integrate PR #27952 CM1 shader generation and capability gating as an available-but-unselected Vulkan candidate with native fallback.

## Steps

Add/register expected CM1 shader variants and host symbols; extend capability snapshot with integer CM support, tile/operand types, subgroup controls/ballot, shared memory, FP4, driver/ICD/device/toolchain identity; match host LDS accounting to shader; gate by capabilities without enum ordinals; test all 12 types/48 variants, q2_k absence, pipeline-failure and unsupported fallback.

## Detailed Solution & Technical Design

CM1 requires coupled shader, generator, host pipeline, Q8_1 preparation, quant paths, subgroup and shared-memory checks. Separate can_execute correctness/capability from selection policy; retain native when unavailable/disabled. Do not claim performance qualification until a real contract executor produces named checks.

## Code Samples & Guidance



## Files

Four 1246 vendor files; TRBC02 capability types/snapshot; Vulkan audit/generator tests; existing capability tooling only.

## Validation

Build expected SPIR-V and host symbols; test q4_0/q4_1/q5_0/q5_1/q8_0/iq4_nl/mxfp4/q3_k/q4_k/q5_k/q6_k/nvfp4, q2_k absent; reject unsupported subgroup/LDS/FP4/driver/pipeline; check --full, sources, patch-lint, rebase and pristine Vulkan build.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

All expected variants compile/link, host/shader LDS agrees, unsupported capability is non-executable, native behavior is unchanged when disabled, no unchecked enum ordinal, and CM1 remains blocked from promotion until contract execution exists.

## Notes

Supersedes: RO20
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rocm-vulkan-provider-ro20

## Change Log

- 2026-09-09T11:01:11.838320+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:18:09.468597+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.596999+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.462256+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:23:25.498614+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032340_repaired-the-cm1-source-and-in_5642
- 2026-09-10T03:23:40.374809+00:00 (updated-by): Updated: section:ledger-events
