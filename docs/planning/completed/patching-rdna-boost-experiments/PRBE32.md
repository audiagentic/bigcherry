---
id: PRBE32
order: 0
plan: patching-rdna-boost-experiments
state: superseded
created-at: '2026-09-09T10:55:40.618469+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# AMD-STREAM-002: Per-(device,stream) BLAS handles

## Description

UPSTREAM-ABSORBED. b11126's ggml_backend_cuda_context already declares independent per-(device,stream) state natively: cublas_handles[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS], cublas_workspaces[...], streams[...], and pools[...], all indexed by curr_stream_no (ggml/src/ggml-cuda/common.cuh:1447-1571). cublas_handle() lazily creates and binds a handle+workspace scoped to (device, curr_stream_no) exactly as this item asked for. This is native upstream infrastructure, not fork-derived, and is exercised by the upstream generic fork/join concurrency mechanism in ggml_backend_cuda_graph_optimize(). Fork patch 1215 (untested) independently ports the same capability from AMD PR#36; it predates convergence with upstream and needs a rebase check (see notes) rather than qualification as new work.

## Steps

1. Confirm at the current pin (grep `cublas_handles\[` and `cublas_handle()` in ggml/src/ggml-cuda/common.cuh) that the per-(device,stream) handle/workspace declarations described here are still present and unchanged from b11126.
2. Read patches/1215_rd394041_amd_stream_moe_overlap/patch.py and check whether any of its Edit() anchors touch ggml_backend_cuda_context's cublas_handles/cublas_workspaces declarations or ggml_cuda_op_mul_mat's handle-selection code; if so, run `patch-rebase-check` to see if those anchors still apply cleanly against the now-native upstream code (likely a no-op conflict/duplicate since upstream already has this).
3. If 1215's per-stream-handle hunk is now a no-op or conflicts, file a follow-up note (not a plan item here) that 1215 should be split/rebased so only its still-needed piece (PRBE34's MoE-shared-expert scheduling) remains; do not edit the patch from this item.
4. No new implementation is required for PRBE32 itself.

## Detailed Solution & Technical Design

Capability owner: patching

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

ggml/src/ggml-cuda/common.cuh (evidence only, no change needed); patches/1215_rd394041_amd_stream_moe_overlap/patch.py (verification target for step 2).

## Validation

Offline only: grep-verify the anchor evidence above still holds at the pinned commit; run `patch-rebase-check --focal-overlay 1215_rd394041_amd_stream_moe_overlap --source bigcherry-tuning` to see whether 1215's handle-related hunks still apply / are redundant. No hardware run needed for this disposition.

## Effort & Risk



## Standards

Concurrency correctness before performance; explicit handle ownership; no ordinary-path penalty.

## Acceptance Criteria

Concurrent GEMMs are correct and independent with no leaks/races; any promotion requires useful overlap without material single-stream regression; otherwise retain as prerequisite evidence.

## Notes

Supersedes: RD40
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd40

append

2026-09-24 relevance at b11126: UPSTREAM-ABSORBED. Verified via git -C work/upstream/llama.cpp.git show b11126:ggml/src/ggml-cuda/common.cuh lines ~1447-1571: ggml_backend_cuda_context declares cublas_handles[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS], cublas_workspaces[...][...], streams[...][...], and pools[...][...], all indexed by curr_stream_no; cublas_handle() lazily creates+binds a handle scoped to (device, curr_stream_no) exactly matching this item's acceptance criteria. This is native upstream (not the AMD fork), already exercised by the generic fork/join concurrency mechanism in ggml_backend_cuda_graph_optimize() (see PRBE34 notes). Fork patch 1215 independently ports the same capability and predates this upstream convergence -- needs a rebase/redundancy check, not new qualification work. GPT design request: attempted 4x (req_fcabed9cee0a4998, req_7e9f4036aae94268, req_b465358100a549c7, req_c68b065818504676), all rejected by the gateway (VAL-AGW-025 gateway-ownership-queue-error, queue overloaded with concurrent batch sessions) -- disposition and plan written directly from verified upstream source instead since this is a straightforward supersession, no design synthesis was actually needed.

2026-09-24 GPT review req_c18183e0a9034c94: READY

## Change Log

- 2026-09-09T10:55:40.618469+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:50.778200+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.272314+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.017739+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:58:17.807856+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025843_amd-stream-successors-prbe323_9820
- 2026-09-10T02:58:43.089488+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:27:32.548252+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:notes
- 2026-09-24T02:27:43.395254+00:00 (updated-by): Updated: section:notes
- 2026-09-24T02:27:48.742236+00:00 (state-transition): State: pending → superseded
- chg_20260924_023553_re-scoped-11-rdna-boost-planni_1625
- 2026-09-24T02:35:57.909534+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:33:35.115835+00:00 (updated-by): Updated: section:notes
