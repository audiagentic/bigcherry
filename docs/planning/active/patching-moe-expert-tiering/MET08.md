---
id: MET08
order: 8
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-06T14:53:05.705421+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Usage-placed expert layout: dense + attention on the XTXs, hot experts across all three cards, cold on the R9700

## Description

Owner's layout (2026-10-06): tensor-split the dense weights and attention (all KV) on the two XTXs, deal each layer's high-usage experts over all three cards leaving room on the XTXs, put the cold remainder on the R9700. Built from 1283 (expert split + BIGCHERRY_MOE_EP_TS), tools/lab/flash-next/expert-place.py (lossless per-layer expert reorder from a usage profile) and -ts 1,1,0. Measurements so far are recorded in MET04: prefill equal to +1% against the row split; decode a few percent higher but following draft acceptance, which changes with the partition; 128/128/256 did not fit at ctx 245760 before 1341 (R9700 full), 132/132/248 and 136/136/240 fit.

## Steps

1. Re-run 128/128/256 at ctx 245760 with 1341 on (1440 MiB returned on the R9700). 2. A decode measure that does not depend on acceptance: several prompts (code, prose, tool JSON) and ms per verify step, ABBA against production on the ORIGINAL file. 3. Hot traffic weighted to the XTXs (2:2:1) - needs a second reordered copy (disk: 41 GB free, remove the 1:1:1 copy first). 4. Explain why the row split gives a different text on the reordered file (MET09). 5. Per-device selected-expert counters in 1283 so balance is measured inside the split. 6. Decide: adopt as a deployment layout (reordered model file + profile flags), keep as an option, or reject.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

patches/1283_qwen4exp_expert_parallel, tools/lab/flash-next/expert-place.py, queue-moe-ep.sh, gpu-usage-sampler.sh, tools/lab/strata/routing-balance.py

## Validation

Adoption needs: fits at the production context, prefill not below production at 8K / 98K / 202K, decode not below production on a multi-prompt set, fidelity inside the envelope at each depth, and a reproducible recipe for the reordered file (profile + caps + traffic recorded in expert-placement.json).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Depends on MSM02 / MSM03 for memory headroom on the R9700. The reordered file is a derived artifact (gguf/placed-128-128-256), not a new quantisation.

## Change Log

- 2026-10-06T14:53:05.705421+00:00 (created-by): Created by agent
