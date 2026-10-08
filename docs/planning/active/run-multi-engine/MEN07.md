---
id: MEN07
order: 8
plan: run-multi-engine
state: pending
created-at: '2026-10-08T20:36:41.926254+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: L
---

# First patch on the second engine, through the same lifecycle

## Description

With radiance as the second engine the patch engine needs no new language support: its sources are C++, HIP and CMake, which the existing anchor dialects cover. What is needed is the per-engine structure from MEN08 and one real change taken end to end.

## Steps

1. Pick one concrete change wanted on radiance (for example an activation marker, or a kernel library entry for a card it does not cover) after MEN01 and MEN02.
2. Author it as patches/radiance/<id>/ with patch.toml, patch.py, tests and SUMMARY.
3. Take it through patch-lint, the per-engine composition check, a Brutus build, an ABBA with activation evidence (MEN06) and the lightweight promotion tier.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-lint, patch-rebase-check for the radiance engine and the mechanics test pass; the built server shows the marker.

## Effort & Risk



## Standards



## Acceptance Criteria

One validated patch on radiance, promoted through the same tier as llama.cpp patches.

## Notes

Deliberately last and conditional: MEN01 must show the relevant code is source, and there must be a specific change to carry. Do not build this ahead of need.

Conditional on there being a specific change to carry. Depends on MEN08 and MEN04.

## Change Log

- 2026-10-08T20:36:41.926254+00:00 (created-by): Created by agent
- 2026-10-08T20:48:09.886352+00:00 (updated-by): Updated: section:title, order=8, section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
