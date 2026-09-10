---
id: RGC01
order: 1
plan: run-gpu-collectives
state: pending
created-at: '2026-09-09T10:47:57.287397+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: P1
---

# Pin MTP head tensors to a single GPU via --override-tensor to avoid cross-GPU AllReduce on the draft path

## Description

The naive tensor override produced a negative result, but the source explicitly leaves split-mode discrimination and possible graph-builder work open.

## Steps

1. Implement the still-valid future scope.
2. Run the stated acceptance and evidence gates.
3. Preserve predecessor provenance and record successor evidence under this ID.

## Detailed Solution & Technical Design

Capability owner: run

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

successor-specs/run-gpu-collectives-gp12.md

## Validation

Historical evidence and constraints: Preserve frozen notes/reviews/evidence on predecessor; IDs: GP03.

Active dependencies: Frozen dependencies: none recorded.

Reference handling: Rewrite forward references (2); preserve historical references (1) on predecessor.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close the recorded gap and pass the frozen scope's stated implementation, correctness, performance, or evidence gate.

## Notes

Supersedes: GP12
Migration: capability-rebaseline-v3-2026-09
Successor key: run-gpu-collectives-gp12

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): GO for experiment only — sufficient design for the next -sm layer / -sm none classification runs (hold model/build/override/workload fixed; outcome is crash-scope classification first, perf only for configs that load). NOT sufficient authority to start graph-builder patching yet. If -ot "nextn\..*=..." works under -sm layer/-sm none but specifically fails under production -sm tensor, spin off a NEW patching-owned item for independently-placeable NEXTN/MTP graph construction rather than silently enlarging this item's scope. Execution order: ranked #1 (cheap, high-value, unblocks whether MTP work stays CLI-level or needs real patch work).

REAL HARDWARE RESULT 2026-09-10 (crash-scope classification, per this item's own next-step and the dev-gpt review's GO-for-experiment verdict): ran the identical -ot "nextn\..*=ROCm0" override (same qwen3.8-27B-Q8_0 MTP gguf, same bc-build-rccl binary, HIP_VISIBLE_DEVICES=0,1) under both -sm none and -sm layer, the two modes GP12 flagged as the next thing to check. -sm none: OOM at model load (27GB q8 model does not fit on one 24GB XTX) -- expected, unrelated to the override, not informative. -sm layer: MODEL LOADED SUCCESSFULLY, server reached 'model loaded'/listening/health=ok with the override applied -- NO CRASH. This confirms GP12's hypothesis: the ggml-backend.cpp:940 pre-allocated-tensor-in-wrong-buffer crash is SPECIFIC to -sm tensor's graph-construction assumptions, not a general NEXTN/MTP-tensor-pinning failure. Consistent with GP03's own finding that -sm tensor is the production-recommended split mode for this box -- so a real fix (if this path is pursued further) needs actual llama.cpp graph-builder changes to treat NEXTN/MTP tensors as an independently-placeable subgraph under -sm tensor specifically, not a CLI-only fix. Per the dev-gpt review's explicit instruction, this finding on its own does NOT authorize starting that graph-builder patch work in this item -- a separate patching-owned item should be filed for it if pursued.\n\nBLOCKED (separate issue, not this item's own scope): attempted the real completion-bench throughput/acceptance comparison under -sm layer (bigcherry.bench.server_completion) to get real numbers for the loads-successfully case, per GP12's original validation methodology. Blocked by an unrelated bench-harness/vendor-source drift: server_completion.py requires Prometheus counters named accepted_tokens_total/draft_tokens_total/drafts_total, but the current vendor llama.cpp server (tools/server/server-task.cpp) emits differently-named counters (spec_decode_num_accepted_tokens_total, spec_decode_num_drafts_total) and none of the locally built binaries on Brutus (checked bc-xtx, bc-build-multi, bc-build-multi-tune, bc-build-tune-4b, bc-build-corefix, bc-build-allfixes, bc-build-multi-replay, bc-build-rccl) emit either metric at runtime even after a real completion request was fired (draft_n/draft_n_accepted came back None on the /completion response too). This is a real, separate tooling gap between the completion-bench harness and the current llama.cpp vendor pin/build -- worth its own investigation (possibly RQW01/RHA01-adjacent, or a fresh item) before completion-bench can be trusted for any future MTP evidence work, not something to route around by hand-parsing /completion output. Did not attempt a workaround here since falsifying/reverse-engineering the correct current metric wiring is outside this item's scope.\n\nDisposition: crash-scope question ANSWERED (real, hardware-verified, consistent with GP12's own hypothesis). Throughput comparison DEFERRED, blocked on the completion-bench/vendor metrics-naming gap above -- flag to user/next session as a real, separate blocker rather than closing this out with fabricated numbers.

## Change Log

- 2026-09-09T10:47:57.287397+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:03:56.793204+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events



- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.775870+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:26.075341+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:02.602666+00:00 (updated-by): Updated: order=1, priority='P1'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.916836+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:13:42.062040+00:00 (updated-by): Updated: section:notes
- chg_20260910_001348_confirmed-on-real-hardware-tha_5870
- 2026-09-10T00:13:48.233417+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.238883+00:00 (updated-by): Updated: section:ledger-events
