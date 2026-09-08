---
id: NRO02
order: 2
plan: nasone-rdna-optimizations
state: pending
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Fuse residual ADD into internal AllReduce completion

## Description

Port the residual-fusion portion of nasone commit `e06dcf6300718227cb8cfda9e61fb12ccb693418` as a separate BigCherry experiment. The target graph pattern is a Tensor-Parallel reduction whose reduced F32 tensor feeds, optionally through reshape-only nodes, a single mirrored residual `ADD`. Instead of completing AllReduce and launching a second elementwise ADD, the AllReduce finish kernel writes `reduced + residual` directly and the Meta scheduler skips exactly the consumed ADD node.

This is deliberately a child of NRO01 in the first draft because NRO01 introduces a unified finish abstraction used by exact/BF16/Q8 paths. The scientific question remains independent: NRO02 must be benchmarked with identical wire representation on both control and subject. It cannot claim Q8's transfer-volume effect.

## Steps

1. Freeze the shared source commit but register provenance through NRO01 only; NRO02 is a local atomic derivative to satisfy the external-source registry's commit-uniqueness rule.
2. Implement an exact, conservative Meta graph matcher: reduction boundary, zero or more no-op reshape nodes, one ADD, one consumer chain, same shape/type, mirrored residual, mirrored output.
3. Add the residual pointer to the internal AllReduce finish interface only after every backend/rank has independently validated the same graph relation.
4. Preserve the ordinary AllReduce API and fallback. If any matcher predicate fails, perform the existing reduction and execute the ADD normally.
5. Skip the ADD only after the fused AllReduce call returns success. Failed provider dispatch must clear the skip decision.
6. Validate operand commutativity intentionally. ADD is mathematically commutative, but graph ownership/use-count semantics are not; tests must cover both operand positions and reject unrelated views/extra consumers.
7. Test exact-FP32 first, then BF16/Q8 only after their own numerical policy passes. The fusion itself should not introduce additional error beyond the selected wire representation.
8. Measure kernel count, AllReduce finish time, eliminated ADD time, graph-submission effects, and end-to-end decode/prefill.

## Detailed Solution & Technical Design

The source change extends the backend communication vtable with an optional fused-add entry point. The Meta backend inspects the next subgraph, identifies a safe residual ADD, passes per-backend residual/output tensors into the communicator, and temporarily clears the ADD compute flag on the next subgraph when fusion succeeds.

The critical invariant is single ownership: the reduced tensor and intermediate reshape chain must not feed another consumer that still expects the unfused value. Shape/type equality alone is insufficient. Split state is also load-bearing: both residual and ADD output must be mirrored so each backend's local residual is semantically identical to the post-AllReduce graph operation.

Do not generalize this to arbitrary epilogues in the first experiment. No bias, MUL, activation, in-place aliasing, or non-F32 residuals. Broader epilogue fusion can be a successor after this exact pattern is proven.

## Code Samples & Guidance

Implement a finish kernel equivalent to:

```cpp
out[i] = reduce(local[i], peer[i]) + residual[i];
```

and a Meta-side matcher that returns false on extra consumers, non-mirrored split state, incompatible type/shape, missing next node, or provider without the optional fused entry point.

## Files

- `docs/planning/active/nasone-rdna-optimizations/NRO02.md`
- `patches/1251_nro02_allreduce_fused_residual/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}`
- shared NRO patch tests.

## Validation

Static graph fixtures must include positive direct-ADD and reshape-ADD patterns plus wrong wiring, extra consumer, non-mirrored residual/output, mismatched shape/type, provider failure, and second-application cases.

GPU correctness compares fused versus unfused output using the same wire mode. Exact-FP32 should be bitwise/operation-order characterized; BF16/Q8 use the parent representation's accepted tolerance. Performance must report eliminated launch count as causal activation evidence.

## Effort & Risk

Medium-high. Kernel arithmetic is simple; graph surgery and node skipping are the risk. A false-positive matcher can silently remove a required operation, so promotion requires aggressive negative-pattern coverage.

## Standards

Exact graph matching, one-consumer proof, fallback preservation, dependency-aware A/B, no threshold/hypothesis duplication outside the future Experiment Contract.

## Acceptance Criteria

- Fused path activates only on the exact safe graph pattern.
- All negative graph fixtures execute the original ADD.
- Fused and unfused outputs meet the representation-specific correctness gate.
- Subject reduces launch count and establishes a positive performance effect without >1% non-target regression.
- NRO02 remains independently disableable from NRO01's Q8 selection.

## Notes

The source commit is intentionally not duplicated in external-source metadata; NRO01 is the source-identity owner and this item records the atomic decomposition.

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created as atomic child of nasone e06dcf630; P0.

## Ledger-events

- Pending: ag-ledger MCP unavailable in authoring session.
