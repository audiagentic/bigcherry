---
id: RAD31
order: 0
plan: patching-radiance-rdna3
state: superseded
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



BACKLOG TRIAGE 2026-11-04 (MERGE into RAD41; revalidated against main 6eb31a708, verdict CONFIRM): Carry RAD31's FP16-versus-E4M3 scaled-row comparisons and quality limits as RAD41 capture-harness variants. Evidence: docs/planning/active/patching-radiance-rdna3/RAD41.md:14-31 is pending and explicitly lists E4M3 per-row/per-128, CPU offline capture error, bytes, combined-row error, KLD and 8-session gates. These encompass the remaining research-only FP8 output-plane steps in RAD31.md:14-32; no independent compression implementation or qualification gate is lost. Original content kept below for history.

## Change Log

- 2026-10-27T02:35:18.290573+00:00 (created-by): Created by agent
- 2026-11-04T04:09:53.475356+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:09:57.466037+00:00 (state-transition): State: pending → superseded
