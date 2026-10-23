---
id: MEN25
order: 0
plan: run-multi-engine
state: completed
created-at: '2026-10-23T02:14:26.236854+00:00'
breadth: ''
skill: intermediate
created-by: agent
work: M
---

# Engine-agnostic patch-promote: engine settings in engines/<name>/engine.toml [promotion]

## Description

bigcherry patch-promote only knew llama.cpp (hardcoded ENGINE, validated-enhancements, source bigcherry, b<N> pin), so rad0008 had to be promoted by hand. Make it work for any engine, with engine-specific values in the engine's own folder, not in the generic script (owner, 2026-10-23).

## Steps

1. [done] engines/<name>/engine.toml [promotion] patch-set/source, parsed into EngineLayout (core/engines.py).
2. [done] patch_promote.py takes the engine; patch tree, recipes, production set/source, pin version (engine's version source) and release package from the layout; fails closed without [promotion].
3. [done] patch-promote requires --engine (cli/engine_scope.py); lifecycle skill updated.
4. [done] Tests: radiance promotion into serving-core, missing [promotion] refused.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/bigcherry/release/patch_promote.py, tools/bigcherry/core/engines.py, engines/llamacpp/engine.toml, engines/radiance/engine.toml, tools/bigcherry/cli/engine_scope.py, tools/tests/release/test_patch_promote.py, tools/tests/cli/test_engine_scope.py, tools/tests/core/test_engine_literals.py, .claude/skills/bigcherry-patch-lifecycle/SKILL.md

## Validation

release 271 OK, core 298 OK, cli 2 pre-existing failures (same on main). CI green. Merged PR #390.

## Effort & Risk



## Standards



## Acceptance Criteria

patch-promote promotes a radiance patch into serving-core and a llama.cpp patch into validated-enhancements with no engine named in tools/bigcherry/release; an engine without [promotion] is refused.

## Notes

Ledger event chg_20261023_020136_bigcherry-patch-promote-now-wo_3569.

## Change Log

- 2026-10-23T02:14:26.236854+00:00 (created-by): Created by agent
- 2026-10-23T02:14:38.432956+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:notes
- 2026-10-23T02:15:01.158624+00:00 (updated-by): Updated: section:acceptance_criteria
- 2026-10-23T02:15:19.907693+00:00 (state-transition): State: pending → completed
