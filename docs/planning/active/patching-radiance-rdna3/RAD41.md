---
id: RAD41
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-01T11:26:14.112101+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# Result row formats and sparse result transfer: bytes against error, measured on captures

## Description

The engine's link carries every helper's result rows (a 2048-row step: ~10.5 MB a head plane, ~4.6 MB the tail's), and that read is ~4 ms of a ~9 ms MoE layer on the R9700's Gen4 x4 link. Formats tried 2026-11-01: f32 partials (2x bytes, -14% at 8 sessions, KLD inside the noise band), fp16 with a power-of-two row exponent (bf16 bytes, 1,884 vs 2,064 tok/s, neutral), bf16 (current). Candidates to shrink bytes: helpers write only the rows they touched plus a compact index (the tail touches a minority of rows), fp8 E4M3 with a per-row or per-128 scale, a per-row-scaled int8. Decide with an offline harness on real captures, not a full KLD per variant.

## Steps

1. Harness: replay captured requests (runs/rad23-cap14, rad23-cap14t) through the worker, dump each helper's f32 partial rows; encode/decode each candidate format on the CPU; report bytes a step and error (rel L2, max abs, per-row worst) against f32, and the error of the summed row the engine adds.
2. Sparse transfer: count touched rows per helper per step from the captures; model the bytes of a compact (index + row) form against dense planes for heads and tail.
3. Pick at most two candidates by bytes x error; implement behind the protocol (one format, migrate all readers/writers, no compatibility mode), then the 42K probe, the 8-session run and one KLD at 2048-token steps.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Harness table for all formats; chosen format reaches lower engine-link bytes with error at or below bf16's; KLD inside the noise band (RAD39); 8-session throughput at or above the current 2,369 tok/s.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

f32 is not reopened on its own: it only pays if bytes stop mattering. The CPU combine already writes one fp16 plane; a narrower combined row is part of this item.

## Change Log

- 2026-11-01T11:26:14.112101+00:00 (created-by): Created by agent
