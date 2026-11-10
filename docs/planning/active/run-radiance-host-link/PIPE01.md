---
id: PIPE01
order: 0
plan: run-radiance-host-link
state: pending
created-at: '2026-11-10T02:09:11.799110+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# Tail layer call: overlap the activation pull and result push with the GEMM by two staggered token halves on copy engines

## Description

On flashnext-4gpu cold steps the 6900 XT tail is the critical path (11.4 ms per layer call: activation pull 1.25, GEMM 7.88, result push 2.25 ms, strictly serial, link-bound at ~3.6 GB/s on Gen3 x4). HL01 showed the copy engines carry both directions at once (1.06x the slower direction) without slowing a concurrent compute kernel, whereas the pull/push kernels add ~0.5 ms contention. Two staggered token halves could hide ~1.75 ms of link time (the first pull and last push stay exposed); splitting costs ~1.1 ms (GEMM time is ~linear in rows; weights re-read), so the bound is ~0.65 ms per layer call (~5% of a cold step).

## Steps

1. Replay mode: run one captured request as two token halves on two streams, staggered, with the pulls and pushes as hipMemcpyAsync from packed contiguous buffers; compare wall time per layer call with the single request (producer/packing kernels included).
2. Bit-identity: per-token reduction order is unchanged (token halves, not expert chunks); compare final helper output bit for bit on the 40 captures.
3. Go if the tail layer call improves by >=0.3 ms repeatably; then cold-TTFT A/B (>=2%, 95% interval) with doc prefill, long fill and decode as guards.
4. If the bound is not met, try 3-4 chunks only if the weight re-read penalty is measured small.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Replay hash identical to the single-request path on all 40 captures; tail layer-call wall time, p50/p95, interleaved; end-to-end cold TTFT paired A/B with a confidence interval; doc prefill, 8-session fill and 1/6-stream decode at parity.

## Effort & Risk



## Standards



## Acceptance Criteria

Tail layer call >=0.3 ms faster in replay, bit-identical output, cold 27k TTFT improved with the 95% interval excluding zero, no repeatable regression above 2% elsewhere, no new spills or occupancy loss.

## Notes

Source: GPT review (2026-11-10) and runs/rad49-gpt-review/followups/PIPE01-notes.md. Expert-chunking is not used: a token's tail contributions span experts, so result pushes cannot start per expert chunk without changing the summation order. Host memory only (no P2P).

## Change Log

- 2026-11-10T02:09:11.799110+00:00 (created-by): Created by agent
