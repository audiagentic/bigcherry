---
id: VTP46
order: 0
plan: run-dense-27b-virtual-tp
state: deprecated
created-at: '2026-11-23T20:42:14.800583+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Decode attention (h256 gqa6) spill-free on gfx1100: split the PV accumulator across warps by head dimension, query and K fragments through LDS

## Description

Withdrawn.

## Steps

None.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

None.

## Effort & Risk

None.

## Standards



## Acceptance Criteria

None.

## Notes

2026-11-24: started with the resource model (step 2) once the accumulate-once baseline is merged and released.

Withdrawn: not justified by our own measurements. Attention decode spills stay tracked under the attn-decode slices and the VTP37 spill scan.

## Change Log

- 2026-11-23T20:42:14.800583+00:00 (created-by): Created by agent
- 2026-11-23T20:43:07.491746+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T20:48:32.456232+00:00 (state-transition): State: pending → in_progress
- 2026-11-23T20:54:34.802032+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-11-23T21:10:52.197017+00:00 (updated-by): Updated: section:notes
- 2026-11-23T21:11:08.007424+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T21:11:17.486954+00:00 (state-transition): State: in_progress → deprecated
