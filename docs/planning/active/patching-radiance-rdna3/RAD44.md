---
id: RAD44
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-11-01T22:24:03.382106+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Parked: multi-row packed-f16 W4 decode GEMV (lane_moe_gemv_w4mr) - deep analysis before revisiting

## Description

lane_moe_gemv_w4mr (branch perf/w4-gemv-mr, PR #506 closed unmerged, kept for revisit) combines W4H2's packed-f16 arithmetic with the FP8 multi-row block structure: one block per (expert, 64 outputs), the expert's routed rows four at a time, each 16-byte weight load decoded once per pass. On 12 real 6-stream (24-row) decode captures on one XTX it measured 267+162 = 429 us per request against W4H2's 185+155 = 340 us (old W4 GEMV 516 us, grouped GEMM ~343 us). Parked, not discarded: rows average ~1.5 per expert on a head helper at 6 streams, and the XTX's 96 MB Infinity Cache likely already absorbs W4H2's repeated per-route weight reads. The kernel may win at higher rows per expert (8+ streams, 33-256-row steps that today fall to the grouped GEMM, MTP verify at more streams) or with a different structure.

## Steps

1. GPT deep analysis of the kernel and the measurements (request in the ag-agents-gateway session for this item).
2. Measure rows-per-expert and distinct experts per helper request across stream counts and step sizes from TRACE/captures.
3. Roofline: bytes read from DRAM vs Infinity Cache for W4H2 and W4MR (rocprofv3 counters on the worker replay, not the engine).
4. Try the structural variants GPT recommends on replay (replay-kernel-time.sh), each against W4H2 on identical captures, at 24, 48, 96 and 256 rows.
5. Only if a variant wins in its target regime: route that regime to it (by rows per expert or step size), end-to-end A/B on step time, KLD.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Replay kernel time against W4H2 on identical captures in each row regime; end-to-end decode step time A/B; KLD at the chunk minimum with the kernel on every step.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner direction 2026-11-01: park and deep-analyse rather than discard. Kernel changes stay in the bigcherry moe-remote component (managed). The worker harness's W4 CPU reference is broken (rel_l2 ~1.0 for the default kernel, RAD15), so correctness is by KLD until it is fixed.

## Change Log

- 2026-11-01T22:24:03.382106+00:00 (created-by): Created by agent
