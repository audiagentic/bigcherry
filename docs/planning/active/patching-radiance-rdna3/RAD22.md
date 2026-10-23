---
id: RAD22
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-23T05:23:14.763211+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P3
---

# moe-remote: replicate high-variance hot experts on both XTX helpers and route each to the less-loaded replica

## Description

Per-step helper imbalance in the Flash-Next helper layout (2026-10-23 trace, combine off, 2 runs x 2 arms): the busier helper gets +16% routes over the mean on the median step, and the slower helper's GPU time exceeds the per-step mean by ~9-10% (prefill 525 vs 480 ms, decode 5,400 vs 4,870 ms over the bench). Totals are balanced (1.38M vs 1.46M routes), so the skew is per-step variance that static ownership (R3_MOE_WORKER_BALANCE, already shelved) cannot fix. Tensor-splitting every expert across both XTX was assessed and shelved (see the model page). This item: copy a small set of high-variance hot experts onto both helpers and send each route to the replica on the less-loaded helper that step, keeping full-width GEMMs.

## Steps

1. Harness first: from TRACE/route captures, simulate per-step assignment with K replicated experts (K = 8..128) and report the critical-path saving vs the VRAM given up (displaced experts fall back to the R9700).
2. Choose replicas by load variance, not mean heat (DeepSeek EPLB eplb.py: replicate, then pack by estimated load).
3. Engine side (rad0002 publish): per step, assign each replicated expert's routes to the helper with fewer routes so far; protocol carries the choice.
4. Helper: hold the replicas; no other change.
5. A/B vs the parent build in helper-ab.json; KLD unchanged (same math, bit-identical per route).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

engines/radiance/components/moe-remote/ (r3_moe_worker.cpp, moe_mailbox_ops.hip, moe_lane_common.h)
engines/radiance/patches/rad0002_moe_remote_handoff/
tools/lab/radiance/heat-placement/ (simulation from traces)

## Validation

Simulation shows a critical-path saving larger than the extra R9700 fallback cost; end-to-end A/B pp and tg not worse with complete separation for the claimed gain; output bit-identical.

## Effort & Risk



## Standards



## Acceptance Criteria

A measured end-to-end gain (prefill or decode) over the parent build with no regression, or a recorded negative result with the simulation numbers.

## Notes

Upper bound from the trace: ~10% of helper GPU time, about 2-3% end to end. Lower priority than RAD21 (round-trip overlap). Prior art: DeepSeek EPLB (github.com/deepseek-ai/EPLB), vLLM/SGLang redundant experts, FasterMoE expert shadowing. GPT review req_bigcherry_e3de4a2ecbd9 (2026-10-23).

## Change Log

- 2026-10-23T05:23:14.763211+00:00 (created-by): Created by agent
- 2026-10-23T05:23:32.642210+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria, section:notes
