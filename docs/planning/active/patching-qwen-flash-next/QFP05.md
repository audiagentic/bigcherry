---
id: QFP05
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:39.453089+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# 1297 MTP draft vocabulary trim (BIGCHERRY_DRAFT_VOCAB_N) and draft LM-head follow-ups

## Description

Patch 1297_draft_vocab_trim (evaluated, in production profile, opt-in BIGCHERRY_DRAFT_VOCAB_N=65536): the MTP draft output.weight (Q8_0, 675 MB, 248K vocab) was ~31% of 6900 draft time at 80K; the head is trimmed to the first N rows with a -inf scatter for the rest. Fixes: flatten head input; fall back to plain head when ggml_nelements(cur)==0 (MTP prompt replay). Follow-ups: frequency-ranked vocab order (FR-Spec) instead of first-N; requantize the draft output matrix smaller (ik_llama.cpp --mtp-requantize-output-tensor); 48K trim.

## Steps

1. Fold review. 2. Frequency-ranked row order (needs a token-frequency table from real traffic). 3. Requantized/smaller draft head A/B. 4. 48K vs 64K acceptance/speed.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Acceptance and ms/step ABBA per change; target output unaffected (draft only).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Evidence: -8% ms/step at 10K, -7% at 80K (flashnext-trim-ab-2). Draft now ~6.5 ms per 3-token draft at 10K, ~9.2 ms at 80K (15-17% of step). GPT review in flight: req_af96db92b8e34cfe.

## Change Log

- 2026-10-03T15:20:39.453089+00:00 (created-by): Created by agent
