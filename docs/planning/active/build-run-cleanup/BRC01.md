---
id: BRC01
order: 1
plan: build-run-cleanup
state: pending
created-at: '2026-09-11T06:44:31.138556+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Remove legacy build/run shims and consolidate bridging layers

## Description

Assess the build/run toolset for legacy compatibility shims, duplicate bridges, and fragmented orchestration. Identify only safe removals or consolidations, preserving supported entrypoints and evidence contracts.

## Steps

1. Inventory legacy shims and bridging modules. 2. Classify each as removable, retain-until-migration, or canonical. 3. Define file-level cleanup and consolidation sequence. 4. Implement only after design review and focused validation.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/bigcherry/**; tools/tests/**; docs/reference/tooling/**; TOOL_DISPOSITION.md

## Validation

Consumer scans, import identity/parity checks, focused CLI tests, build/run workflow tests, hygiene checks, and evidence-contract checks.

## Effort & Risk



## Standards



## Acceptance Criteria

A file-level, dependency-aware cleanup design identifies safe removals and required bridging work without creating a second implementation or breaking supported entrypoints.

## Notes

Initial assessment and design to be supplied by GPT gateway review before implementation.

## Change Log

- 2026-09-11T06:44:31.138556+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260911_064900_added-a-tracked-cleanup-plan-f_3622
- 2026-09-11T06:49:00.213062+00:00 (updated-by): Updated: section:ledger-events
