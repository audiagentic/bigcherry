---
id: BCOP15
order: 15
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T04:44:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Assess compact active-lane dispatch for range-owned MoE experts

## Description

Backfill of the earlier MET02 audit. Range-aware zero/skip semantics can remain dispatch-heavy when a tier owns a small fraction of routed experts. Assess compact `(output lane, local expert)` dispatch using existing MUL_MAT_ID grouping/workspace rather than another router.

## Steps

1. Confirm MET02 range semantics and current implementation status on the live branch.
2. Profile 0/10/25/50/75/100% local ownership across ub1-512 on gfx1100/gfx1201.
3. Prototype global->local translation plus compact active-lane list in existing backend workspace; scatter results back to original lanes.
4. Preserve current precision selection and current upstream MMQ tail/allocation safety.
5. Gate through existing HIP autotune rather than a second dispatcher.

## Related

MET02, HIP autotune, QFP10; upstream MUL_MAT_ID work.

## Acceptance Criteria

- Compact dispatch promotes only for >=5% end-to-end or >=10% MUL_MAT_ID kernel improvement on a representative <=25%-local lane.
- No required dense lane regresses >2%.
- No duplicate router/dispatch registry is created.
