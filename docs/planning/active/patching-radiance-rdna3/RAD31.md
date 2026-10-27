---
id: RAD31
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-27T02:35:18.290573+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: M
---

# Compressed (FP8) helper result plane: offline quality and transfer study

## Description

Research only. The combined FP16 result plane is ~20 MiB at 4096 rows x 2560; E4M3 with per-row or per-128 scales would halve it (MORI quantized combine, intranode.hpp:310-385, 650-715). BF16 combine already failed the quality gate, so accuracy risk is high. Only worth doing if PWR/RAD21 profiling shows the result transfer on the prefill critical path.

## Steps

1. Captured final combined rows: G0 FP16, G1 E4M3 per-row, G2 E4M3 per-128.
2. Reconstruction error, KLD, top-1, encode/decode cost, bytes, critical-path saving.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Go only if KLD/top-1 pass the existing gates and the transfer saving is on the critical path.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Communication format, unrelated to the model's FP8 expert weights. Gated on RAD21 profiling.

## Change Log

- 2026-10-27T02:35:18.290573+00:00 (created-by): Created by agent
