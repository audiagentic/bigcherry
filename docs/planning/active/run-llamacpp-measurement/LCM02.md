---
id: LCM02
order: 0
plan: run-llamacpp-measurement
state: pending
created-at: '2026-11-19T22:17:26.106083+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P3
work: S
---

# llama.cpp engine feature states and host-facts start receipts in run records

## Description

The llama.cpp parts removed from the Radiance-only MRC02/MRC03 slices on 2026-11-20, preserved for this item: tag archive/mrc03-llamacpp-adapter (10c812fa6: feature_registry.llamacpp_states, a llama.cpp benchmark feature-state adapter, engines.LLAMACPP_RECORD) and tag archive/mrc02-llamacpp-wrapper (b949cc30d: host-facts start-receipt requirement for the llama wrapper).

## Steps

1. Declare llama.cpp's optional features in the platform feature registry (MRC03) and have the llama.cpp producer emit feature states per run (no separate 'legacy upstream identity' engine name: use core.engines.LLAMACPP).
2. Require the sealed host-facts start receipt (MRC02) on llama.cpp measured starts.
3. Reuse the archived commits where they fit the no-legacy rule.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

llama.cpp run records carry feature states and a start receipt; guards (engine literals, tooling boundaries) pass.

## Effort & Risk



## Standards



## Acceptance Criteria

llama.cpp records meet the same MRC02/MRC03 contract as Radiance records, with no compatibility identity.

## Notes



## Change Log

- 2026-11-19T22:17:26.106083+00:00 (created-by): Created by agent
- 2026-11-19T22:17:55.528715+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria
