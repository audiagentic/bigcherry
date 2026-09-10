---
id: TRVP04
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:00:27.277814+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# hipBLASLt vendor_auto candidate

## Description

Implement the hipBLASLt vendor_auto provider route as a deterministic candidate built on the shared explicit primitive. Preserve a stable BLAS problem translation while making provider selection, workspace, launch, correctness, and fallback observable.

## Steps

1. Add provider candidate side-table/routes for NATIVE, HIPBLASLT_AUTO, HIPBLASLT_EXPLICIT, and CK without changing the mathematical BLAS plan. 2. Reuse one ggml_hip_make_hipblaslt_problem authority from can_execute through workspace, launch, and logging, including dtype/layout/conversion handling. 3. Implement provider-managed auto selection using the shared explicit capability/selection primitive; include provider workspace and conversions in the timed region. 4. Make auto fallback deterministic and persist requested provider, resolved provider, and reason; explicit unsupported requests must fail closed with a classified error. 5. Keep graph_safe and deterministic false until independently proven, and make replay verify resolution semantics rather than only the final provider name. 6. Run correctness, timing, tuning, and replay evidence on representative signatures and preserve the predecessor evidence link.

## Detailed Solution & Technical Design

The runtime keeps the mathematical BLAS plan stable and adds a provider-qualified candidate side table. A single ggml_hip_make_hipblaslt_problem translation authority feeds eligibility, workspace sizing, launch, and telemetry so conversions and descriptor fields cannot diverge. The HIPBLASLT_AUTO route consumes the same capability result as explicit candidates. Auto may select a supported provider candidate or take a deterministic, observable fallback; the record must contain requested provider, resolved provider, fallback reason, provider workspace, candidate identity, stack/device fingerprint, and graph/deterministic flags. Explicit provider requests never silently fall through to native. Until graph capture and determinism are demonstrated, candidates remain non-graph-safe/non-deterministic.

## Code Samples & Guidance



## Files

hipBLASLt provider route and ggml_hip_make_hipblaslt_problem translation in the dispatch/runtime sources; tuning schema/catalog and candidate identity; provider/replay/graph-safety tests; qualification evidence for representative BLAS signatures.

## Validation

Force blas:hipblaslt:auto and verify it resolves, executes, and is numerically correct across representative dtypes/layouts, records effective provider hipblaslt, and survives tuning and replay. Verify conversions and provider workspace are included in timing. Exercise unsupported and overflow cases to confirm classified fail-closed behavior; exercise deterministic auto fallback and confirm requested/resolved/reason fields persist. Confirm auto cannot silently fall through when explicitly forced, and replay rejects resolution-semantic mismatches.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

All eligibility, workspace, launch, and logging paths use the same problem translation. Auto selection and fallback are deterministic and observable with requested/resolved provider plus reason. Overflow, unsupported, and explicit-provider failures are classified and fail closed. Representative correctness, timing, tuning, and replay evidence passes; graph_safe/deterministic remain false unless separately proven.

## Notes

Supersedes: RO09
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro09

Supersedes RO09. Preserve the shared explicit primitive boundary with TRVP05 and independent oracle gate TRVP06. Do not alter patch 1225 or bypass ledger/planning governance.

## Change Log

- 2026-09-09T11:00:27.277814+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:18.595728+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.546928+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.286737+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:42:34.364351+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034253_repaired-trvp04-06-so-the-acti_1663
- 2026-09-10T03:42:53.380629+00:00 (updated-by): Updated: section:ledger-events
