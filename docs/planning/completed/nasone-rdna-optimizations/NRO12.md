---
id: NRO12
order: 12
plan: nasone-rdna-optimizations
state: superseded
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Replicate lm_head for DFlash2/DSpark under tensor split

## Description

Track correctness fix `22ed83e9aaba8eab1cd730ee57cd37f3623a42c8` for DFlash2/DSpark tensor-parallel speculative decoding. These drafters can rank the full vocabulary inside the graph. When the draft has no output head of its own and shares the target lm_head, a tensor-split output projection is incomplete on each device; the ranker requires a full replicated output tensor.

## Steps

1. Detect draft architecture/capability from GGUF metadata before target model load, not by filename.
2. Add an explicit `output_replicated` model/load parameter and carry it through model construction.
3. Mark output weight and bias mirrored in Meta split-state policy when required.
4. Make DFlash model loading set the same property when selector/Markov metadata proves full-output ranking.
5. Cover both shared-target lm_head and draft-with-own-lm_head; the latter must not force unnecessary target replication.
6. Verify memory cost and fail clearly if replication cannot fit.
7. Test tensor split 2+ GPUs, layer split control, and ordinary MTP/Eagle/no-spec controls.

## Detailed Solution & Technical Design

The fix is topology semantics. A full-vocabulary TOP_K/argmax cannot operate correctly on one shard unless the algorithm performs a distributed selection; this source chooses replication instead. Bias split state must follow weight state, otherwise adding axis-0 bias to mirrored logits is invalid.

Memory is part of correctness/operability: replicating a large lm_head can materially increase per-GPU VRAM. Record the delta and consider future distributed-top-k as a separate optimization if replication is too costly.

## Code Samples & Guidance

Do not hard-code DFlash2 names. Source helper inspects `general.architecture` and tensors such as selector/Markov weights, and returns false when the draft already owns `output.weight`.

## Files

Planning-only initially; future package touches common speculative init, public model params, Meta split-state, model state, DFlash model load, and tests.

## Validation

Control must reproduce wrong/abort behavior on a real/synthetic DFlash2 tensor-split topology. Subject must show full output tensor on every rank, exact candidate selection versus single-GPU/reference, and correct bias placement. Record VRAM increase.

## Effort & Risk

High due cross-layer API plumbing and memory implications, though core algorithm is straightforward.

## Standards

Affirmative TP topology proof; exact candidate-selection correctness; explicit resource accounting.

## Acceptance Criteria

- Full-vocabulary draft ranking matches reference under tensor split.
- No replication when not required.
- Output weight+bias split states remain coherent.
- Memory cost is recorded and within declared operational limit.

## Notes

NRO11 may be required in some shared-target device-list configurations but is not inherently a source-code prerequisite.

Superseded by: PNRO11
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from nasone DFlash2/DSpark TP fix; P1.

## Ledger-events


- Pending: ag-ledger MCP unavailable in authoring session.
- 2026-09-09T11:25:20.346382+00:00 (updated-by): Updated: section:notes
- 2026-09-09T11:43:24.942830+00:00 (state-transition): State: pending → superseded
- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:57:59.894603+00:00 (updated-by): Updated: section:ledger-events
