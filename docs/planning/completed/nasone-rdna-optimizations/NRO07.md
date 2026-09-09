---
id: NRO07
order: 7
plan: nasone-rdna-optimizations
state: superseded
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Hybrid ROCm TOP_K kernels

## Description

Port and qualify the HIP-specific TOP_K implementation from nasone commit `7f3e1e4d0b166cb681b2c01503370e610a5b423d` (`ROCm: add hybrid TOP_K kernels`). BigCherry currently optimizes downstream use of MoE routing weights (for example RD17) but does not carry an AMD-specific implementation of the TOP_K selection operation itself. On current `b10705`, non-CUB HIP falls back to full bitonic argsort plus copy, so a selection-specific algorithm can remove substantial unnecessary work for small k.

PNRO06 is the foundation implementation. NRO08 applies the later wave32-native reduction/tuning commit on top. Keeping the two separate allows BigCherry to determine whether the hybrid algorithm itself wins before attributing additional gain to wave32 reduction changes.

## Steps

1. Freeze `7f3e1e4d...` source and audit its entire TOP_K dispatch against `b10705` `top-k.cu`.
2. Add HIP-only selection kernels while preserving CUDA/CUB branches exactly.
3. Implement ordered-float key conversion with explicit NaN/Inf/tie policy consistent with existing TOP_K semantics.
4. Add specialized TOP-1 reduction, n-ary multi-pass selection, and radix selection only where source dispatch uses them; preserve bitonic fallback for unsupported shapes.
5. Make implementation initially selectable/traceable for A/B. Do not replace all HIP TOP_K blindly until exact-index correctness is proven.
6. Create CPU/reference fixtures across k, ncols, nrows, ties, negative values, infinities, and duplicate values. Verify index set/order semantics expected by callers.
7. Profile scratch size, shared-memory use, atomics, waves/block, and number of passes.
8. Capture real MoE/QSA TOP_K signatures and weight measurements by call count/time; synthetic microbench results are insufficient for priority decisions.
9. Compare hybrid versus existing bitonic path at k=1 and routing-like k values (e.g. 2/4/8/10 where models exercise them).
10. Keep PNRO07 disabled during the NRO07 causal arm.

## Detailed Solution & Technical Design

The source avoids sorting all columns. It maps IEEE float values into monotonic unsigned keys, selects the bucket containing the k-th threshold through radix/n-ary passes, compacts values above/equal to that threshold, and carries `(index,value)` pairs between passes where needed. TOP-1 has a cheaper reduction path. This changes algorithmic complexity and temporary-storage behavior, not merely block geometry.

Tie semantics are load-bearing. Existing callers generally need valid top-k indices, but deterministic ordering among equal values may affect routing identity. Tests must compare exact expected behavior against the current implementation and identify whether the source changes stable tie ordering. Any divergence must be explicitly accepted or corrected before performance work.

No runtime candidate ID should be invented in the matmul tuning schema. TOP_K is a separate op family; experiment selection can use an env/control lever local to this patch until a general non-matmul dispatch framework exists.

## Code Samples & Guidance

Primary target is `ggml/src/ggml-cuda/top-k.cu`. HIP-specific code must be surrounded by `GGML_USE_HIP`; CUDA CUB behavior must compile to the exact pre-patch logic.

## Files

- `docs/planning/active/nasone-rdna-optimizations/PNRO06.md`
- `patches/1256_nro07_topk_hybrid/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}`
- shared NRO static tests; future TOP_K correctness fixture runner.

## Validation

Static: non-HIP preprocessor preservation, idempotent apply, selector fallback. Reference tests: exact top-k index set and required ordering for random/hostile vectors; k=1..ncols boundaries; multi-row.

Hardware: gfx1100 primary, with real MoE and QSA signatures if available. Performance: kernel time, scratch bytes, end-to-end routing/model lane, call-weighted effect.

## Effort & Risk

High. Large kernel code, tie/NaN semantics, shared-memory/atomic coordination, and multiple shape-dependent algorithms. A fast wrong routing choice can pass superficial numerical checks while changing model behavior.

## Standards

Exact routing correctness before performance; HIP-only containment; native fallback; no conflation with PRBE14 or downstream MoE fusion.

## Acceptance Criteria

- Exact TOP_K semantics pass all reference/tie/boundary fixtures.
- Non-HIP builds are unaffected.
- Unsupported shapes fall back safely.
- At least one real high-cost gfx1100 signature establishes a repeatable kernel win with no model correctness regression.
- PNRO07 remains a separate causal increment.

## Notes

This is genuinely distinct from PRBE14: NRO07 computes selected expert indices; RD17 folds already-computed routing weights into a later projection.

Superseded by: PNRO06
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from nasone hybrid ROCm TOP_K commit; P0.

## Ledger-events


- Pending: ag-ledger MCP unavailable in authoring session.
- 2026-09-09T11:24:52.422017+00:00 (updated-by): Updated: section:notes
- 2026-09-09T11:42:59.313658+00:00 (state-transition): State: pending → superseded
- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:57:59.860792+00:00 (updated-by): Updated: section:ledger-events
