---
id: QFP08
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:57.065205+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# MTP draft-prefill overlap on the 6900 XT drafter (TTFT)

## Description

Scope narrowed 2026-10-04 (FMTP01 Gate 0 review, owner-approved): decode-time draft/verify overlap, sync elimination and hidden-state handoff moved to the FMTP plan (FMTP01 Gate 0 owns measurement; options F/C/B there). QFP08 now owns only draft-prefill overlap: pipeline the draft context's prompt processing chunk by chunk behind target prefill (draft chunk i runs while the target processes chunk i+1, fed the target hidden rows of chunk i). Objective: long-context TTFT; no change to decode or acceptance. Expected 5-30% of draft-related TTFT, bounded by the currently serialized draft prefill and CPU/PCIe contention.

## Steps

1. Measure draft-prefill time vs target prefill at 10K/80K/240K prompts on profile v3 (server timings + rocprof).
2. Design the chunk pipeline at the server/common speculative prompt seam: target chunk submit -> hidden rows ready (event, not context synchronize) -> draft chunk on ROCm3 while next target chunk runs.
3. Implement as a patch behind an env flag; fail closed to serial on any error.
4. Validate: TTFT ABBA, greedy identical, acceptance unchanged, no target prefill slowdown.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

TTFT ABBA at 10K/80K; greedy identical; draft acceptance unchanged; target pp t/s not reduced.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

GPT design req_93c6774520f7454a (draft-prefill overlap). Online (RV4214): SGLang parallel-spec notes MTP/EAGLE need hidden-state transport; DPDraft RFC supports keeping the drafter TP=1 on its own GPU (our setup). Do not reuse the rejected 1215 concurrency approach.

2026-10-04 synctrace on profile v2 (~80K, 256 decode tokens; flashnext-v2-synctrace, sync-tracer.c counts hipStreamSynchronize call sites): the decode-time host waits are dominated by the MTP draft loop - draft-context graph_compute via common_speculative_draft/process (3060+, 946, 765 calls; ~12 syncs per draft call) and llama_get_embeddings_nextn_ith -> llama_context::synchronize (2 x 1032 calls: a full sync every step to read the MTP hidden state). 864 further calls are prompt-checkpoint state copies (state_seq_get_data) during the fill, not decode. Targets: async draft graph (no per-call syncs), event-based hidden-state handoff instead of a context-wide synchronize, double-buffered hidden state. Next instrument: add per-call blocked-time to sync-tracer.c so sites are ranked by wait time, not count.

## Change Log

- 2026-10-03T15:20:57.065205+00:00 (created-by): Created by agent
- 2026-10-03T16:30:23.311740+00:00 (updated-by): Updated: section:notes
- 2026-10-04T03:26:36.373482+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:validation
- 2026-10-04T03:26:55.973470+00:00 (updated-by): Updated: section:description
