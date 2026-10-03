---
id: QFP03
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:25.602966+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: S
---

# 1294 deterministic TOP_K ties (lowest-index tie-break in HIP radix top-k)

## Description

Patch 1294_topk_deterministic_ties (evaluated, in production profile): HIP radix TOP_K picked tied QSA cells in atomic order (ReLU-sum scores give many exact zeros), so long-context greedy output differed between server starts. Lowest-index tie-break makes 32K/80K output identical across starts. Speed neutral. Conflicts with 1256.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Same prompt, two server starts, greedy identical at 32K and 80K; ms/step unchanged.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Upstream fact (GPT online search): QSA HIP's >1K-context cliff was TOP_K falling back to CPU, fixed by upstream radix TOP_K, so TOP_K is on the critical path. Distinct from PNRO06/PNRO07 (TOP_K speed). GPT review in flight: req_af96db92b8e34cfe.

## Change Log

- 2026-10-03T15:20:25.602966+00:00 (created-by): Created by agent
