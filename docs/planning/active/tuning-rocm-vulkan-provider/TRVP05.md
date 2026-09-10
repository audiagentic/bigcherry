---
id: TRVP05
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:00:32.148319+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# hipBLASLt explicit algorithms

## Description

Implement explicit hipBLASLt algorithm enumeration, reconstruction, capability checks, workspace handling, and stable provider-qualified candidate identity for tuning and replay.

## Steps

1. Implement the shared explicit hipBLASLt selection/execution primitive before vendor_auto consumes it. 2. Enumerate native, auto, and explicit indexes from provider inventory; reconstruct indexes with getAlgosFromIndex and check each with matmulIsAlgoSupported. 3. Capture workspace requirements and reject unsupported or incompatible algorithms before measurement. 4. Cache runtime descriptors by candidate runtime_id, problem signature, hardware fingerprint, and stack fingerprint. 5. Emit stable names containing provider and semantic plan while preventing cross-stack/index reuse without an exact fingerprint. 6. Integrate provider-aware candidate identity with tuning/catalog/replay and run inventory, shape-rejection, correctness, and replay validation.

## Detailed Solution & Technical Design

The explicit route is the common capability and execution primitive for both HIPBLASLT_EXPLICIT and HIPBLASLT_AUTO. Candidate discovery enumerates provider inventory, reconstructs provider indexes through getAlgosFromIndex, validates each with matmulIsAlgoSupported, records workspace, and classifies rejection causes. Descriptor caches are namespaced by runtime_id/problem signature/device and stack/toolchain fingerprint. Candidate identity includes provider, semantic plan, algorithm/index, workspace, and fingerprint; an index from another build, device, or stack is never accepted. Native remains a distinct candidate and existing tuner ranking is preserved.

## Code Samples & Guidance



## Files

tools/bigcherry/tuning/catalog.py and schema/candidate identity; hipBLASLt runtime selection/descriptor path; provider dispatch and replay/cache tests; inventory and fingerprint evidence.

## Validation

For representative problems, generated explicit candidates equal provider inventory before shape-specific rejection; every rejected candidate has a classified unsupported/incompatible reason. Verify descriptors and workspace are cached and reused only for matching runtime_id/signature/hardware/stack fingerprints. Confirm provider-qualified names and algorithm identity survive tuning, serialization, and replay, while a candidate from a different build/fingerprint is rejected. Confirm existing tuner ranking and native candidate behavior remain unchanged.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

The shared explicit primitive drives both explicit and auto routes. Unsupported or incompatible indexes fail closed and are classified. Candidate identity includes provider, algorithm/index, semantic plan, workspace, and exact provider/stack fingerprint; cross-fingerprint transfer is rejected. Inventory, shape rejection, correctness, tuning, serialization, and replay gates pass without changing native ranking semantics.

## Notes

Supersedes: RO10
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro10

Supersedes RO10. TRVP04 must consume this primitive; TRVP06 independently validates both routes. Do not bypass ledger/planning governance.

## Change Log

- 2026-09-09T11:00:32.148319+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:23.102568+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.551129+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.390746+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T01:02:43.283004+00:00 (updated-by): Updated: section:validation
- chg_20260910_010342_successor-plans-now-have-expli_8662
- 2026-09-10T01:03:42.532314+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:42:28.131527+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034253_repaired-trvp04-06-so-the-acti_1663
- 2026-09-10T03:42:53.395709+00:00 (updated-by): Updated: section:ledger-events
