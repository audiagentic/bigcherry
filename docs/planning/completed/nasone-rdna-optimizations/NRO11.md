---
id: NRO11
order: 11
plan: nasone-rdna-optimizations
state: superseded
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Add ctx_other model devices to speculative scheduler backends

## Description

Track nasone commit `84b31d4cdb696cd09f04219de5f9a9b8d5ec6ac7`, a correctness fix for speculative/assistant models that share tensors through `ctx_other`. If the draft model's own device list does not include a GPU holding a shared target tensor (for example output/tok_embd), its scheduler may lack a backend capable of executing that preallocated tensor. The source adds missing backends for devices used by the other model.

## Steps

1. Reproduce the abort/failure with a target/draft device-list mismatch under a shared-tensor speculative configuration.
2. Enumerate `model_other->devices`, deduplicate against current model devices/backends, and initialize only genuinely missing devices.
3. Preserve backend order assumptions; document where newly added backends appear relative to ACCEL/CPU backends.
4. Verify lifecycle/destruction ownership of added backend instances.
5. Test same-device target/draft as no-op, subset/superset/disjoint device lists, Meta-wrapped tensor split, and single GPU.
6. Verify scheduler chooses the correct backend for shared tensors and no extra copies are introduced unexpectedly.

## Detailed Solution & Technical Design

This is orchestration correctness, not a speed patch. Device identity must come from the initialized other model, not CLI guesses. Deduplication should compare backend device handles, and failure to initialize a required backend should fail explicitly rather than silently scheduling through CPU.

## Code Samples & Guidance

Primary source surface is `src/llama-context.cpp` context construction immediately after ordinary model device backends are added and before generic ACCEL backends.

## Files

Planning-only initially; future single-patch package plus scheduler/shared-tensor integration tests.

## Validation

Positive mismatched-device fixture must fail on control and pass on subject; same-device/no-ctx_other controls unchanged. Validate output and copy topology, not merely absence of abort.

## Effort & Risk

Medium. Small code but scheduler/backend lifetime and ordering are sensitive.

## Standards

Correctness fix requires affirmative scheduler evidence and output parity; no throughput threshold needed for acceptance.

## Acceptance Criteria

- Shared tensors execute on a backend valid for their allocation under mismatched target/draft device lists.
- No duplicate backend creation or lifetime leak.
- Controls unchanged.

## Notes

Pairs naturally with NRO12 in a DFlash/DSpark validation campaign but should remain an independent patch/hypothesis.

Superseded by: PNRO10
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from nasone ctx_other backend fix; P1.

## Ledger-events


- Pending: ag-ledger MCP unavailable in authoring session.
- 2026-09-09T11:25:14.325022+00:00 (updated-by): Updated: section:notes
- 2026-09-09T11:43:19.797109+00:00 (state-transition): State: pending → superseded
- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:57:59.887171+00:00 (updated-by): Updated: section:ledger-events
