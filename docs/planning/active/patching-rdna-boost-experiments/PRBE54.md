---
id: PRBE54
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:14.838615+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# UP-KV-001: Dedicated F16 dequant path for Q4/Q5 KV

## Description

Qualify dedicated F16 dequantization for Q4/Q5 KV cache inputs, with per-KV-type promotion only when AMD measurements justify it.

## Steps

Recheck PR #27140; implement vectorized half2 Q4/Q5-to-F16 dequant where absent; test Q4_0/Q4_1/Q5_0/Q5_1 long-context Qwen against Q8_0/F16/decode controls at 8K/32K/64K/128K and q_rows 128/512/2048; measure dequant time, PP/TG, and memory.

## Detailed Solution & Technical Design

Replace generic elementwise dequant bottlenecks with vectorized half2 conversion for lower-bit KV types while preserving exact/tolerant numerical semantics and existing fallback.

## Code Samples & Guidance



## Files

Shared CUDA/HIP KV dequant kernels; per-KV-type correctness tests; long-context benchmark manifests and evidence.

## Validation

Exact/tolerant dequant and model output; dequant kernel time, PP/TG, memory; promote per type only where AMD benchmark wins.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only a KV type with output/dequant correctness and repeatable AMD PP/TG benefit versus Q8_0/F16/decode controls; retain generic path otherwise.

## Notes

Supersedes: RD64
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd64

## Change Log

- 2026-09-09T10:57:14.838615+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:28.184233+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.372555+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.168472+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:11:58.576713+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031217_repaired-four-more-active-succ_7909
- 2026-09-10T03:12:17.962856+00:00 (updated-by): Updated: section:ledger-events
