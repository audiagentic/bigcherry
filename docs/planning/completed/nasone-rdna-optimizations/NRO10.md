---
id: NRO10
order: 10
plan: nasone-rdna-optimizations
state: superseded
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# Increase Meta compute-container view headroom for recurrent/MTP graphs

## Description

Evaluate nasone commit `91de35039a1a2a366fde0bc6c5b515bbf7e3cabb`, which raises `ggml_backend_meta_alloc_ctx_tensors_from_buft()` compute-container headroom from 16 to 128 static-tensor views. Hybrid recurrent models with MTP snapshotting can create substantially more than 16 views per static tensor between evaluations; insufficient context capacity is a correctness/capacity failure, not a performance hypothesis.

## Steps

1. Confirm the `b10705` Meta allocator still uses fixed headroom and derive the actual maximum view count for current recurrent+MTP graphs.
2. Build a deterministic boundary fixture that allocates 15/16/17 and higher views per static tensor through the same Meta context mechanism.
3. Port the minimal constant/rationale only if the failure is reproducible or current graph construction proves the capacity bound can be exceeded.
4. Prefer a derived/bounded formula over a magic 128 if the allocator exposes the required dimensions safely; otherwise retain source constant with an explicit upper-bound rationale.
5. Measure additional metadata memory, not just success/failure.
6. Verify non-recurrent and non-MTP graphs are behaviorally unchanged.

## Detailed Solution & Technical Design

The source rationale estimates recurrent snapshot views at roughly `2*(n_rs_seq+1)` per recurrent layer sharing the buffer. The static context must reserve enough tensor-object metadata for views created in compute containers after static allocation. This does not allocate the underlying model tensor data 8x larger; nevertheless metadata growth must be measured and bounded.

## Code Samples & Guidance

Target is one constant in `ggml/src/ggml-backend-meta.cpp`. Keep the experiment minimal and resist bundling unrelated Meta changes.

## Files

Planning-only initially. Future package may be a single-file correctness patch with a dedicated allocation boundary test.

## Validation

Boundary allocation test, real recurrent+MTP graph construction, repeated eval/reset, memory accounting, and controls with ordinary dense graphs.

## Effort & Risk

Low-medium code risk, medium proof risk. An arbitrary larger constant can hide an unbounded-growth bug; capacity should be justified from graph structure.

## Standards

Affirmative capacity test, resource accounting, no performance claim needed for correctness promotion.

## Acceptance Criteria

- Existing 16 limit is shown insufficient for a supported configuration or a proven bound demonstrates insufficiency.
- New headroom covers declared maximum with margin and bounded metadata cost.
- No non-target behavioral regression.

## Notes

This is intentionally not materialized in the initial P0 patch set despite small code size; reproduce the boundary first.

Superseded by: PNRO09
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from nasone Meta headroom commit; P1.

## Ledger-events


- Pending: ag-ledger MCP unavailable in authoring session.
- 2026-09-09T11:25:08.654212+00:00 (updated-by): Updated: section:notes
- 2026-09-09T11:43:15.240501+00:00 (state-transition): State: pending → superseded
- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:57:59.880887+00:00 (updated-by): Updated: section:ledger-events
