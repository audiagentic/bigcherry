---
id: RCS06
order: 6
plan: build-radiance-components
state: pending
created-at: '2026-10-18T12:57:50.674536+00:00'
breadth: ''
skill: basic
created-by: agent
priority: P2
work: S
---

# Move the n-gram helper into its own component

## Description

r3_ngram_helper.hip, r3_ngram_host.cpp, r3_ngram_api.h and r3_ngram_test.cpp have no dependency on the MoE path. Make them the ngram-helper component, status evaluated and default off, with its own tests and README. Per the owner (2026-10-13) the R9700 is compute, not table storage, so no new GPU-side n-gram runs are planned; keep the code on main for revisiting.

## Steps

1. git mv into engines/radiance/components/ngram-helper with component.toml and CMake.
2. Register in engine.toml; keep output helpers/libr3_ngram_helper.so.
3. Run r3_ngram_test.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Component builds and r3_ngram_test passes; runtime home unchanged.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Independent of RCS02-RCS05; can go first as the template for a component move.

## Change Log

- 2026-10-18T12:57:50.674536+00:00 (created-by): Created by agent
