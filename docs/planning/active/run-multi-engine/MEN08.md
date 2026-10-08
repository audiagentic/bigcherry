---
id: MEN08
order: 4
plan: run-multi-engine
state: pending
created-at: '2026-10-08T20:47:01.932040+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Repository structure so two engines can be patched in parallel

## Description

Today the patch system has one namespace: patches/<id>/ (152 packages), one overlay (src/), one vendor tree, one composition per source, one pin transition marker, one CI audit over everything, and one config/recipes.toml that every patch PR touches. Two engines can only be worked on at the same time, by different slices or agents, if none of those are shared.

## Steps

1. Namespace by engine: patches/<engine>/<id>/, src/<engine>/ overlay, vendor/<engine>/, tools/tests/patch/<engine>/. The 152 existing packages move to patches/llamacpp/ in one mechanical change with every reference updated; no aliases or fallback lookups for the old paths.
2. Patch identity is (engine, id): registry, catalog, lint, composition, rebase check, dispositions and evidence are keyed by it; requires and conflicts never cross engines.
3. Split configuration so engines do not contend for one file: config/engines/<engine>.toml holds that engine's pin, sources, patch sets and experiments; config/recipes.toml keeps only what is shared (platforms, hosts, trees).
4. CI per engine with path filters: a PR that touches only patches/radiance/ runs the radiance lint, composition and compile check, not the llama.cpp audit, and the reverse.
5. Pin bumps per engine with independent transition markers and rebase reports (with MEN04).
6. Lab: work/builds/<engine>/, tree-activity leases and the queue's build rows carry the engine; the GPU locks stay shared because the cards are.
7. Release: one release-please component per engine, so a batch of promotions on one engine releases without the other.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

patches/, src/, config/recipes.toml, tools/bigcherry/patch/registry.py, tools/bigcherry/patch/rebase.py, tools/bigcherry/patch/offline_check.py, tools/bigcherry/patch/patchset.py, .github/workflows/patch-offline-validation.yml, release-please config

## Validation

Before/after production source-tree hash for llama.cpp is identical; the full offline suite passes; two slices, one per engine, each change a patch and merge without touching a common file.

## Effort & Risk

L. Mechanical but wide: every patch path, about 200 test files and the CI workflow change at once. Risk is in missed path references, so the equivalence proof is the unchanged llama.cpp production tree hash plus the full offline suite. Do it as one change with no behaviour change mixed in, at a moment when few patch PRs are open.

## Standards



## Acceptance Criteria

A radiance patch PR and a llama.cpp patch PR can be authored, checked by CI and merged in parallel with no shared file between them and no cross-engine CI cost.

## Notes

Packaging change only: no patch's mechanism or output changes, so no validation is invalidated (owner rule). Do after MEN01 and MEN02 confirm radiance is staying; do before MEN07.

## Change Log

- 2026-10-08T20:47:01.932040+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261008_212949_added-a-like-for-like-single-c_9270
- 2026-10-08T21:30:09.553704+00:00 (updated-by): Updated: section:ledger-events
