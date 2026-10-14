---
id: RAD07
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-14T22:16:25.441629+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Per-model radiance docs: what we run, best results, what is on, what was shelved

## Description

engines/radiance/README.md is a 1,144-line dated lab log mixing models, kernels and hardware; there is no per-model statement of current state. Create a clean, maintained doc set: docs/reference/engines/radiance/README.md (overview, links to parity assessment, runbook, platform design) and docs/reference/engines/radiance/models/<model>.md for qwen3.8-flash-next and qwen3.8-27b.

## Steps

1. Move RADIANCE_PLATFORM_DESIGN/PARITY/RUNBOOK under docs/reference/engines/radiance/ (update every link).
2. Per-model page sections: container and quantisation; reference quality vs the full model (Flash-Next owner table: mean KL 0.0789 all / 0.0239 text+assistant, median 0.0037/0.0029, P99 1.65/0.32, P99.9 5.33/1.03, top-1 92.85%/94.57%); layouts tried with best prefill/decode and run IDs; patches and settings on, and why; tried and shelved, with reason; open levers linked to plan items.
3. Keep engines/radiance/README.md as the dated lab log; each entry links to its model page.
4. Doc rule: every measurement or decision updates the model page in the same change.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Links resolve; each model page states current best layout and numbers that match the latest recorded runs.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Recording against model-specific docs is an owner requirement (2026-10-15).

## Change Log

- 2026-10-14T22:16:25.441629+00:00 (created-by): Created by agent
