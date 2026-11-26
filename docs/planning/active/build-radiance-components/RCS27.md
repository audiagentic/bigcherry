---
id: RCS27
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-26T01:36:30.551307+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Model-driven discrete qualification selection and evidence gates

## Description

Extend RCS25 unit_harness.py and unit_families.toml. Resolve actual build groups to narrower qualification families and owned rows, support dependencies and gaps. Explicit arch and matched reference. Add read-only plan, fail closed on missing hashes, reference failures, missing rows, skipped/unconverged replay and resource-only PASS ambiguity. Extend uncovered families with real oracle and workload data.

## Steps

1. Add offline group-to-family selection in existing harness; report incomplete inventory and coverage. 2. Map overlays via origins/targets and actual row contracts. 3. Gate exact reference and candidate evidence, missing/zero/skipped/unconverged cases. 4. Validate selected family compile and matched leased measurements; add missing oracles incrementally.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

See docs/design/radiance-kernel-composition.md; extend existing owners.

## Validation

Synthetic groups, malformed/partial inventory, missing coverage, zero hashes, stale/failed reference, skipped and unconverged replay. Actual selected build, then discrete GPU tests under bounded lease.

## Effort & Risk



## Standards



## Acceptance Criteria

Existing harness and outputs only; every required unit covered or explicitly nonpass; compile-only is diagnostic, no automatic promotion.

## Notes

Detailed design and boundaries: docs/design/radiance-kernel-group-qualification.md. No active related item edited.

## Change Log

- 2026-11-26T01:36:30.551307+00:00 (created-by): Created by agent
- 2026-11-26T01:37:56.167096+00:00 (updated-by): Updated: section:steps, section:notes
