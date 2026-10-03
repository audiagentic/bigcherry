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

# MTP draft/verify overlap on the 6900 XT drafter

## Description

Overlap MTP drafting on the 6900 XT with target work instead of strict draft -> verify serialisation. Draft costs ~6.5 ms per 3-token draft at 10K and ~9.2 ms at 80K = 15-17% of an MTP step (profile ABBA logs); that is the ceiling. MTP drafters need target hidden states, so full cross-round overlap is not free: start with double-buffered hidden-state transfer and overlapping transfer/sampling, and draft-prefill overlap at prompt time.

## Steps

1. Trace one MTP step timeline (draft, transfer, verify) on profile v2. 2. Prototype draft-prefill overlap. 3. Hidden-state double buffering.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

ms/step ABBA; acceptance unchanged; greedy identical.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

GPT design req_93c6774520f7454a (draft-prefill overlap). Online (RV4214): SGLang parallel-spec notes MTP/EAGLE need hidden-state transport; DPDraft RFC supports keeping the drafter TP=1 on its own GPU (our setup). Do not reuse the rejected 1215 concurrency approach.

2026-10-04 synctrace on profile v2 (~80K, 256 decode tokens; flashnext-v2-synctrace, sync-tracer.c counts hipStreamSynchronize call sites): the decode-time host waits are dominated by the MTP draft loop - draft-context graph_compute via common_speculative_draft/process (3060+, 946, 765 calls; ~12 syncs per draft call) and llama_get_embeddings_nextn_ith -> llama_context::synchronize (2 x 1032 calls: a full sync every step to read the MTP hidden state). 864 further calls are prompt-checkpoint state copies (state_seq_get_data) during the fill, not decode. Targets: async draft graph (no per-call syncs), event-based hidden-state handoff instead of a context-wide synchronize, double-buffered hidden state. Next instrument: add per-call blocked-time to sync-tracer.c so sites are ranked by wait time, not count.

## Change Log

- 2026-10-03T15:20:57.065205+00:00 (created-by): Created by agent
- 2026-10-03T16:30:23.311740+00:00 (updated-by): Updated: section:notes
