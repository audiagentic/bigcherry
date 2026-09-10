---
id: TRBC02
order: 0
plan: tuning-reusable-build-campaign
state: pending
created-at: '2026-09-09T10:58:58.770421+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Vulkan candidate eligibility ABI: can_execute() cannot evaluate real Vulkan capability yet

## Description

Define immutable Vulkan capability snapshot ABI for candidate can_execute and replay compatibility before candidate execution.

## Steps

Add typed snapshot with normalized API/driver/device features/extensions/limits/subgroups/shader/toolchain digest; canonical serialize/digest; pass immutable snapshot to preflight/replay; require hardware and capability digests; test unsupported extension/limit/subgroup/API/shader rejection.

## Detailed Solution & Technical Design

Separate compact hardware identity from capability snapshot; capture once per process/lane. Do not copy HIP structs or infer capabilities from ad hoc JSON.

## Code Samples & Guidance



## Files

vk-autotune types/capabilities, candidate preflight/replay, tests.

## Validation

Capability mismatch returns native fallback with reason; digest stability and unsupported feature/limit tests.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Candidates cannot execute or replay without matching immutable capability and shader digests; unknown/unsupported capability fails closed.

## Notes

Supersedes: RE32
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-reusable-build-campaign-re32

## Change Log

- 2026-09-09T10:58:58.770421+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:07.858220+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.476353+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.326340+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:39:39.185958+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_034017_repaired-six-vulkanprovider-s_6655
- 2026-09-10T03:40:17.972992+00:00 (updated-by): Updated: section:ledger-events
