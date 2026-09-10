---
id: PNRO02
order: 0
plan: patching-nasone-rdna-optimizations
state: pending
created-at: '2026-09-09T10:52:07.096423+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P0
---

# Fuse residual ADD into internal AllReduce completion

## Description

Qualify residual ADD fusion into internal AllReduce completion as an independently disableable experiment. It may reuse PNRO01's finish abstraction but must not claim Q8's transfer-volume effect.

## Steps

1. Keep PNRO01 as shared source identity and dependency; implement an exact conservative matcher for reduction→optional reshape-only chain→single mirrored ADD.
2. Require same shape/type, mirrored residual/output, one consumer, and provider fused-entry support.
3. Pass residual only after backend/rank validation; skip ADD only after fused AllReduce succeeds and clear skip on failure.
4. Cover both operand positions, use-count/aliasing, extra consumers, and provider fallback.
5. Test exact-FP32 first; test BF16/Q8 only under their own representation policies.
6. Measure kernel count, eliminated ADD launch, finish time, graph submission, and decode/prefill.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

patches/1251_nro02_allreduce_fused_residual; Meta matcher; AllReduce finish API; static graph/GPU tests; evidence

## Validation

Positive direct/reshape-ADD fixtures; negative wrong wiring, extra consumer, non-mirrored state, shape/type mismatch, provider failure, and second-apply fixtures. Fused/unfused output equivalence under the same wire mode; eliminated-launch activation evidence; balanced performance with non-target regression gate.

## Effort & Risk



## Standards

Exact graph matching; one-consumer proof; fallback preservation; dependency-aware A/B; no epilogue generalization before this pattern is proven.

## Acceptance Criteria

Fusion activates only for the exact safe graph pattern, all negative fixtures retain ordinary ADD, correctness passes per representation, and performance evidence is causal. PNRO02 remains independently disableable from PNRO01 Q8 selection.

## Notes

Supersedes: NRO02
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-nasone-rdna-optimizations-nro02

Supersedes: NRO02
Inherited semantic scope: preserve graph ownership/use-count, mirrored split, failure clearing, and independent-wire-mode qualification.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T10:52:07.096423+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:36.096098+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.057911+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.695018+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:26:03.477332+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.275437+00:00 (updated-by): Updated: section:ledger-events
