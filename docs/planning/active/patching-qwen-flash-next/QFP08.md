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

## Change Log

- 2026-10-03T15:20:57.065205+00:00 (created-by): Created by agent
