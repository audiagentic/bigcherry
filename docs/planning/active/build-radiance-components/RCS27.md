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

### 2.3.0 compatibility closure (2026-11-27)

Use unit_harness.py --model-graph with actual baseline and candidate saved /graph, unit_families.toml, compose_check and origins; join selected row, layout, operand geometry, rank/architecture and execution evidence, including helper-worker and speculative-draft paths not represented by base graph alone. Missing M64/gated IQR, group256 IQ8R, U6/U7 run256/512, mixed-bit head, U2 draft head, U4 I8-table drafter, Flash-Next w4nl64a8i8h and protected BF16 experts, E4M3 HC/MTP/ngram and packed cache rows must be explicitly covered where used. Provider substitution must pass real model layout and both graph selection and measured issue; native libr3d lacks chained/collective paths and does not cover all required forms. Test refusal as well as success, use existing byte/resource/replay gates; a row name or metadata match is not proof. No retirements on missing/unresolved graph coverage. MEN17 owns integrated generation and quality; no duplicate harness.

## Notes

Detailed design and boundaries: docs/design/radiance-kernel-group-qualification.md. No active related item edited.

2026-11-26: Extended existing unit harness with exact rows and saved Radiance /graph model-band selection; provider, geometry, rank and gaps retained. Require leased physical BDF and actual selftest device identity, artifact digests and observed controls; strict duplicate/missing/skipped hash and per-round timing/layout checks, cold/warm replay. 280 focused tests passed, one skipped. Real configured inventory and explicit IQR row plan resolved embed/tiled, M64/gated/unknown row remain uncovered. Graph importer fixture-tested; no new live graph or revised physical replay due to active GPU cohort. Remaining native oracles and matching reference/leased build validation.

## Change Log

- 2026-11-26T01:36:30.551307+00:00 (created-by): Created by agent
- 2026-11-26T01:37:56.167096+00:00 (updated-by): Updated: section:steps, section:notes
- 2026-11-26T05:07:03.703862+00:00 (updated-by): Updated: section:notes
