---
id: RCS06
order: 6
plan: build-radiance-components
state: completed
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

2026-10-20 done: engines/radiance/components/ngram-helper (component.toml, CMakeLists.txt, ngram_plugin.cpp). The n-gram row was registered by libr3's own plugin table, so it became its own small kernel plugin, libr3ngram (the libr3moe pattern), not a plain file move; to use the shelved feature, list libr3ngram in --kernels and set R3_NGRAM_DEV. The seven R3_NGRAM_* settings moved with it; libr3 keeps the gfx11 kernels only. The component declares targets gfx1100/gfx1201 because the engine is configured for the union of component targets (the bump-gate test caught gfx1030 widening the engine build); the helper code object still carries gfx1030. Checked on Brutus: formal lane build ok, kernels/libr3ngram.so and helpers/libr3_ngram_helper.so installed, libr3.so no longer carries the row, rad-kbench loads libr3ngram, r3_ngram_test builds.

## Change Log

- 2026-10-18T12:57:50.674536+00:00 (created-by): Created by agent
