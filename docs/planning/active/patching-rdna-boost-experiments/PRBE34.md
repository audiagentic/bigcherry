---
id: PRBE34
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:55:49.488869+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# AMD-STREAM-004: Overlap MoE shared expert on auxiliary stream

## Description

IMPLEMENTED-AS-PATCH, needs rebase. Patch 1215_rd394041_amd_stream_moe_overlap (state=untested) implements MoE shared-expert overlap on an auxiliary stream during small-batch Qwen MoE decode (RD42), and already has substantial real hardware evidence recorded in this item's own notes below (+2.38% mean gfx1100 gain over 10 paired rounds, byte-exact bit_identical logits, profiler-confirmed 82% stream overlap). However, b11126 upstream now natively provides the generic fork/join concurrency substrate (ggml_cuda_concurrent_event, stream_mapping, per-stream cublas_handle/pool, curr_stream_no dispatch in ggml_backend_cuda_graph_optimize) that 1215 partially reimplements from scratch (see PRBE32/PRBE33 -- both superseded as upstream-absorbed). The remaining real gap is that upstream's own generic detector is hardcoded to a 3-way `attn_norm`-named fan-out (QKV), not the MoE shared-expert 2-way fan-out this item targets. Disposition: keep qualifying/promoting patch 1215 as-is is still valid (it works, per the extensive real evidence below), but before promotion a rebase check should confirm whether 1215's stream/scratch plumbing duplicates upstream and should be trimmed to just the MoE-fan-out detection predicate riding on the native mechanism -- smaller, less regression-prone, and avoids maintaining a second concurrency substrate.

## Steps

1. This item's substantive qualification work (activation, bit_identical, backend_reference-adjacent, and 10-round CI95 performance evidence) is essentially done per the notes below; remaining formal step is wiring run_rd39_42_contract_qualification() (fully designed in notes, GPT-approved req_3e42043eb71a4a92) into validation_campaign.py so patch-verify-evidence can compute eligible_for_validated_state.
2. Before final promotion, run a rebase/redundancy check: read patches/1215_rd394041_amd_stream_moe_overlap/patch.py, identify which Edit()s duplicate now-native upstream code (per-stream cublas_handle/pool -- see PRBE32/33) vs which are the still-necessary MoE-fan-out-specific piece (the shared-expert diamond detection and its stream_mapping entries).
3. If duplication is confirmed, scope (as a SEPARATE future item, not implemented here) a slimmer patch that only adds a MoE shared-expert predicate to the existing ggml_backend_cuda_graph_optimize() fan-out/join detector (real anchor: the `if (!strstr(root_node->name, "attn_norm")) continue;` gate and `min_fan_out`/`max_fan_out` constants at b11126 ggml-cuda.cu, function ggml_backend_cuda_graph_optimize, roughly lines 4650-4760) instead of 1215's bespoke stream/scratch plumbing -- reusing the native ggml_cuda_concurrent_event/stream_mapping/join_events machinery already exercised for QKV.
4. Complete the formal validation.toml + Experiment Contract wiring for 1215 (design already scoped in notes) regardless of the rebase outcome, since the real evidence already gathered stands on its own.

## Detailed Solution & Technical Design

Real anchored context (b11126 ggml/src/ggml-cuda/ggml-cuda.cu, function ggml_backend_cuda_graph_optimize): the existing generic detector computes `fan_out[src]` per source tensor, requires `count >= min_fan_out && count <= max_fan_out` (both hardcoded to 3), and only proceeds `if (strstr(root_node->name, "attn_norm"))`. A MoE shared-expert diamond has fan_out==2 (routed-expert branch + shared-expert branch off the FFN input norm) rather than 3, and the root node is an FFN/MoE input norm, not attn_norm. Extending this detector to also recognize the 2-way MoE case (e.g. widen min_fan_out to 2 and add an alternate name/shape predicate identifying the FFN-input norm feeding both a MUL_MAT_ID (routed experts) and a plain MUL_MAT/dense path (shared expert)) would let 1215's MoE overlap ride the same stream_mapping/concurrent_event/join_events/cublas_handle()/pool() machinery already used for QKV, instead of reimplementing it. This is real, scoped follow-on design (not implemented this session) -- flagged here so it is not lost, but PRBE34's own disposition (qualify patch 1215 as-is) does not require it: 1215 already builds and has real positive hardware evidence independent of this optimization.

## Code Samples & Guidance

Real anchor (verify with: git -C work/upstream/llama.cpp.git grep -n 'attn_norm' b11126 -- ggml/src/ggml-cuda/ggml-cuda.cu): `if (!strstr(root_node->name, "attn_norm")) { continue; }` inside ggml_backend_cuda_graph_optimize's fan_out loop, adjacent to `const int min_fan_out = 3; const int max_fan_out = 3;`. A future slimmer patch would add a second branch here recognizing the MoE shared-expert root (FFN input norm name pattern + fan_out==2) alongside the existing attn_norm/fan_out==3 branch, both feeding the same downstream ggml_cuda_concurrent_event construction code already present (~line 4756 onward).

## Files

patches/1215_rd394041_amd_stream_moe_overlap/patch.py (rebase-check target); patches/1215_rd394041_amd_stream_moe_overlap/README.md (already has extensive real evidence, see item notes); tools/bigcherry/patch/validation_campaign.py (needs run_rd39_42_contract_qualification(), design already recorded in notes); config/experiment-contracts.toml [contract.RD39-42-STREAM-MOE-OVERLAP] (already exists, needs qualification runner wiring only).

## Validation

Offline: PYTHONPATH=tools python -m bigcherry patch-lint patches/1215_rd394041_amd_stream_moe_overlap; patch-rebase-check --focal-overlay 1215_rd394041_amd_stream_moe_overlap --source bigcherry-tuning. Hardware (on Brutus, not run here): the remaining real step is executing run_rd39_42_contract_qualification() once authored, on tierM-qwen35b-a3b-moe-mtp per the corrected contract binding, to produce a formal eligible_for_validated_state verdict -- the underlying evidence (bit_identical, activation, 10-round CI95 performance) already exists per this item's notes and does not need to be regathered.

## Effort & Risk



## Standards

Dependency-aware scheduling; workload predicate; join correctness; no unconditional overlap.

## Acceptance Criteria

Exact eligible workload overlaps safely with byte-identical output and >1% E2E TG gain; all other paths retain existing behavior; prerequisites are explicit.

## Notes

Supersedes: RD42
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd42

2026-09-24 relevance at b11126: IMPLEMENTED-AS-PATCH (1215_rd394041_amd_stream_moe_overlap, state=untested), with extensive real prior evidence already in this item's own notes (10-round CI95 performance +2.38%, byte-exact bit_identical logits, profiler-confirmed overlap -- see the 2026-09-12 entries above). New finding this session: PRBE32/PRBE33 (the RD40/RD41 prerequisite pieces this item depends on) are now UPSTREAM-ABSORBED natively at b11126 (per-stream cublas_handle/pool, generic fork/join concurrency detector in ggml_backend_cuda_graph_optimize -- real anchors: ggml/src/ggml-cuda/common.cuh ~1447-1571, ggml-cuda.cu ~4504-4800). Upstream's own generic detector is hardcoded to attn_norm/fan_out==3 (QKV), not the MoE shared-expert 2-way case, so patch 1215's own MoE-overlap piece is still real, not-yet-upstream work -- but its stream/scratch substrate likely duplicates the now-native mechanism and should get a rebase check before promotion (design sketched above). GPT design gateway was unavailable this session (4 rejected attempts, VAL-AGW-025 queue-ownership errors -- see PRBE32 notes for request ids); the rebase-extension design above was written directly from verified upstream source rather than GPT-reviewed, and should get a GPT design pass before anyone actually implements the slimmer patch.

## Change Log

- 2026-09-09T10:55:49.488869+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:12:58.630166+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.280175+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.031786+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:58:31.227383+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025843_amd-stream-successors-prbe323_9820
- 2026-09-10T02:58:43.116897+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:29:04.400409+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation
- 2026-09-24T02:29:32.124976+00:00 (updated-by): Updated: section:notes
- chg_20260924_023553_re-scoped-11-rdna-boost-planni_1625
- 2026-09-24T02:36:06.238415+00:00 (updated-by): Updated: section:ledger-events
