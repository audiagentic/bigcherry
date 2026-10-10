---
id: QFP51
order: 51
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-10T21:23:38.446485+00:00'
breadth: ''
skill: advanced
created-by: claude
priority: P2
work: M
---

# Scheduler split plan reused across same-shape graphs (MTP verify widths 2-8 and prompt batches)

## Description

Split out of QFP42 (closed 2026-10-11 as done by 1359), where it had been folded in as the PRBE56 follow-up. It was never part of removing the per-batch wait and is not done.

Every graph is split again from scratch: `ggml_backend_sched_split_graph` (ggml/src/ggml-backend.cpp) assigns backends, builds the splits and their inputs each call, although consecutive graphs of the same shape split the same way. Measured host cost today (run ht1, Flash-Next 24K, 512-token prompt batches, 1319 / 1320 timing): graph build and allocation 11.4 ms a batch on the target, never reused (0 of 76); the Meta backend's own rebuild about 4 ms a batch. For MTP verify steps (widths 2..8) the split cost recurs every decode step and has not been measured separately.

Since 1359 the prompt path keeps one batch queued ahead, so this host time no longer leaves the cards idle in prefill; what is left to gain there is the mid-length gaps (kernel-gap-stats.py, run pipe4: 293 gaps of 2-10 ms a card, 1.28 s of a 28.1 s prefill). In decode the host time is on the critical path every step.

## Steps

1. Measure first, per call: split / build CPU time, time blocked in target against draft, bytes and copies a token, for MTP verify widths 2..8 and for a 512-token prompt batch (1319 / 1320 / 1325 timing exist; add what is missing).
2. If material: cache a TEMPLATE keyed by shape and topology, backend-id mapping and split boundaries, and rematerialise per instance the tensor pointers, `split->inputs` and graph-owned views.
3. Lightweight tier for the result.

## Detailed Solution & Technical Design

Constraints carried over from the QFP42 source audit:
- Reusing a raw `ggml_backend_sched_split` is unsafe: graph splitting mutates `node->src[]`.
- The existing `prev_backend_id != split_backend_id` check already avoids a same-backend synchronise; removing that guard is not a new win and must not be claimed as one.
- Remove a cross-device wait only where a trace shows it is not a real dependency.
- llama's own graph reuse (`llm_graph_result::can_reuse`) never fires for these prompt batches because the QSA mask shapes follow the growing context; that is a separate lever (fixing the shapes) and is noted here only so it is not rediscovered.

## Code Samples & Guidance



## Files



## Validation

Greedy identity at 8K / 24K / 98K and a long decode (a plan cache is a reordering of host work only); host timing before and after per call; ABBA on one binary; repeated runs at 98K.

## Effort & Risk



## Standards



## Acceptance Criteria

- The per-call split cost is measured for verify widths 2..8 and for a prompt batch.
- The template cache is promoted (lightweight tier) or rejected with that measurement.
- No raw split object is reused.

## Notes

Origin: PRBE56, folded into QFP42 on 2026-10-08, split out again 2026-10-11 on the owner's instruction. Related: QFP16 (target verify submit host time 5-7 ms a round), QFP46 (prefill concurrency contracts).

## Change Log

- 2026-10-10T21:23:38.446485+00:00 (created-by): Created by claude
