---
id: MET06
order: 6
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-02T04:45:10.449925+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: L
---

# 1285 expert_slice_loader: runtime per-expert slicing of a standard GGUF (--expert-placement)

## Description

Remove the offline-repack requirement once MET03-MET05 prove value. llama_model_loader synthetic tensors: compact destination [ne0,ne1,n_tier] per tier filled by raw expert-slab copies (src offset old_expert*nb[2]); router rows permuted at load; CPU tiers get compact CPU memory (not zero-copy mmap). Separate --expert-placement <json> flag (common parser -> POD structs via llama_model_params); not an -ot extension. Reject placement that overlaps an -ot expert override.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Standard Q6 GGUF + placement JSON produces byte-identical tier tensors to the offline repack (MET03); greedy parity and KLD identical to the repacked run.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

-ot selects buffer type per whole tensor and cannot express partitioning.

## Change Log

- 2026-10-02T04:45:10.449925+00:00 (created-by): Created by agent
