---
id: MEN25
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-10-23T01:21:28.357440+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: S
---

# patch-promote supports radiance (engine-aware promotion and patch sets)

## Description

tools/bigcherry/release/patch_promote.py hard-codes ENGINE = LLAMACPP and promotes into [patch-set.validated-enhancements], so a radiance patch (engines/radiance/patches/<id>, sets in engines/radiance/recipes.toml such as serving-core) cannot be promoted with `bigcherry patch-promote`; the first radiance promotion (rad0008_hc_upmix_rt_sweep, RAD18, 2026-10-23) had to be done by hand. Owner rule: every engine follows the same build -> patch -> test -> validate -> promote -> release workflow through bigcherry commands (CLAUDE.local.md 'Engines are managed alike', MEN10-MEN14).

## Steps

1. Take --engine from the CLI context instead of the module constant; resolve the engine's recipes.toml and patches root through core.engines.
2. Make the target production patch set engine-declared (llamacpp: validated-enhancements; radiance: serving-core) rather than a literal.
3. Use the engine's evidence and rebase-check commands (radiance: --source radiance); keep the same evidence line requirements (Model, Mechanics, Activation, Identity, A/B).
4. Tests for a radiance promotion and for refusing an unknown engine.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

tools/tests release tests cover a radiance promotion end to end on a fixture recipe; patch-lint and check pass after promoting a radiance fixture patch; llama.cpp promotion behaviour unchanged.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Found while promoting rad0008 (RAD18). That promotion's evidence record is releases/evidence/rad0008_hc_upmix_rt_sweep-promotion.md, written in the same format patch-promote produces.

## Change Log

- 2026-10-23T01:21:28.357440+00:00 (created-by): Created by agent
