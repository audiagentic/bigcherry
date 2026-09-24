---
id: PRBE33
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:45.334008+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-STREAM-003: Dedicated scratch for concurrent branches

## Description

UPSTREAM-ABSORBED. b11126's ggml_backend_cuda_context gives each (device,stream) pair its own ggml_cuda_pool instance (pools[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS], common.cuh:~1552-1562, lazily created via new_pool_for_device(device, curr_stream_no)). Concurrently executing branches selected by the native fork/join mechanism (ggml_backend_cuda_graph_optimize / curr_stream_no dispatch, see PRBE34) therefore already allocate temporaries from separate per-stream pools, which is the dedicated-scratch-without-aliasing property this item asked for -- it is a native allocator-level solution, not the fork's dedicated reused scratch buffer. Fork patch 1215 independently implements its own concurrent-branch scratch handling; needs the same rebase/redundancy check as PRBE32.

## Steps

1. Confirm at the current pin that ggml_backend_cuda_context::pool(device) still indexes pools[device][curr_stream_no] (common.cuh) and that this is used by the concurrent-branch dispatch path in ggml_backend_cuda_graph_optimize/graph_compute (ggml-cuda.cu ~4200-4360, where curr_stream_no is set from concurrent_event->stream_mapping[node] before any allocation-using op runs).
2. Read patches/1215_rd394041_amd_stream_moe_overlap/patch.py for any Edit() touching scratch/pool/allocator code for concurrent branches; run patch-rebase-check to see if that hunk is now redundant against the native per-stream pool.
3. No new implementation required for PRBE33 itself; if 1215's scratch hunk conflicts or duplicates, note it for a future 1215 rebase (do not edit the patch here).

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

ggml/src/ggml-cuda/common.cuh (evidence only); ggml/src/ggml-cuda/ggml-cuda.cu (evidence only); patches/1215_rd394041_amd_stream_moe_overlap/patch.py (verification target).

## Validation

Offline only: grep-verify the pool()/pools[][] evidence above; run patch-rebase-check --focal-overlay 1215_rd394041_amd_stream_moe_overlap --source bigcherry-tuning. No hardware run needed for this disposition.

## Effort & Risk



## Standards

Correctness prerequisite; isolate branch lifetimes; optimize memory only after safety.

## Acceptance Criteria

No scratch aliasing or output corruption under stress; memory cost is bounded; optimization is not promoted before safety proof.

## Notes

Supersedes: RD41
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd41

append

2026-09-24 relevance at b11126: UPSTREAM-ABSORBED. Verified via git -C work/upstream/llama.cpp.git show b11126:ggml/src/ggml-cuda/common.cuh: pools[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] gives each (device, curr_stream_no) pair its own ggml_cuda_pool, and the native fork/join concurrency dispatch (ggml-cuda.cu ~4285-4311) sets curr_stream_no per node from concurrent_event->stream_mapping[node] before compute, so concurrent branches already allocate from separate pools -- structurally equivalent to the dedicated-scratch goal here, via the allocator rather than a bespoke buffer. Same caveat as PRBE32: fork patch 1215 ports an independent scratch mechanism and needs a redundancy/rebase check, not new qualification. GPT design gateway was unavailable this session (4 rejected submission attempts across PRBE32-34, see PRBE32 notes for request ids) -- disposition written directly from verified upstream source.

2026-09-24 GPT review req_c18183e0a9034c94 applied: WRONG disproven -- per-stream ggml_cuda_pool is op-temp only, native graph_optimize interleaving (ggml-cuda.cu ~4801) is a different mechanism than 1215's dedicated concurrent_scratch buffer; reopened to pending, RD41 scratch retained as still-needed, qualification wiring deferred to PRBE34.

## Change Log

- 2026-09-09T10:55:45.334008+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:54.282206+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.276223+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.024905+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:58:24.631748+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025843_amd-stream-successors-prbe323_9820
- 2026-09-10T02:58:43.103972+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:28:03.133494+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:notes
- 2026-09-24T02:28:08.335931+00:00 (updated-by): Updated: section:notes
- 2026-09-24T02:28:19.217467+00:00 (state-transition): State: pending → superseded
- chg_20260924_023553_re-scoped-11-rdna-boost-planni_1625
- 2026-09-24T02:36:01.939817+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:35:09.830498+00:00 (state-transition): State: superseded → pending
- 2026-09-24T04:36:02.097102+00:00 (updated-by): Updated: section:notes
