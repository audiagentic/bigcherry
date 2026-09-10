---
id: PRVP01
order: 0
plan: patching-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:01:08.382627+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P1
---

# Capture and pin PR #27952 Vulkan CM1 source

## Description

Capture immutable llama.cpp PR #27952 final-state source against the current vendor pin as an availability-only packaged patch.

## Steps

Record URL/head 965e57103fce2c4329cdfc2b8300f8f7ed57c9fe, semantic base cc83d7b..., target pin 2578138...; diff final state against pin and classify all four coupled host/shader/generator files; package 1246_ro19_vulkan_cm1_pr27952 with patch.toml/patch.py/SUMMARY; register external source; materialize pristine and prove no default route change.

## Detailed Solution & Technical Design

Capture coupled ggml-vulkan.cpp, mul_mmq_cm1.comp, funcs.glsl and shader generator as one deterministic semantic transplant. Keep state untested, no fabricated evidence, package-only identity, and availability disabled until PRVP02/TRVP qualification.

## Code Samples & Guidance



## Files

config/external-sources.toml; patches/1246_ro19_vulkan_cm1_pr27952/{patch.toml,patch.py,SUMMARY.md}; four vendor Vulkan paths; source/provenance tests.

## Validation

Source/external-source checks, patch-lint, pristine materialization/rebase/apply, compile host and shader generator, exact four-file diff accounting and identity changes on source/pin change.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Immutable provenance, clean packaged application/build, complete coupled diff, no default CM1 enablement or production behavior change, and moving upstream head blocks reuse.

## Notes

Supersedes: RO19
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rocm-vulkan-provider-ro19

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
