---
id: RHP17
order: 0
plan: patching-radiance-heat-placement
state: pending
created-at: '2026-11-10T00:10:41.824087+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Device-speed and phase-aware expert placement: minimise each layer's slowest-device finish time (offline simulation first)

## Description

The live planner (worker_heat.h propose) ranks experts by heat and minimises the tail's heat share; devices have only byte budgets, no speed or PCIe model, so it reported no_improvement_or_margin 30/30 on cold prompts while the 6900 XT tail was ~73% busy. Both GPT reviews: optimise the per-layer critical path C(expert, device, phase) = dispatch + transfer + compute + combine, consider a cold-latency and a sustained-throughput profile, and replicate a few hot tail experts only when it pays. A smaller tail is not the answer within today's RAM (7 GiB tail: cold TTFT -26% but 8-session fill -21%).

## Steps

1. Fit per-device stage costs by rows per expert (captures + helper stage logs; tail vs XTX ratios differ per stage).
2. Offline simulator over captured requests and heat: current placement vs per-layer-max objective within the same byte budgets.
3. Only if the simulation shows recoverable time: extend the planner objective (managed moe-remote change), live A/B on cold, doc and long-fill workloads.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Simulation predicts and a live A/B confirms a cold-prompt TTFT gain with no loss on doc prefill, 8-session long fill or decode; KL unchanged (ownership changes alter reduction grouping, so gate it).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Evidence: runs/rad49-4gpu-coldtail, rad49-4gpu-tailsweep, rad49-4gpu-longfill; findings runs/rad49-findings-20261109.md. Refs: HybriMoE sec. IV-B, DeepSeek EPLB balanced_packing/replicate_experts, MoE-Infinity phase policies.

## Change Log

- 2026-11-10T00:10:41.824087+00:00 (created-by): Created by agent
