---
id: MEN07
order: 7
plan: run-multi-engine
state: pending
created-at: '2026-10-08T20:36:41.926254+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P3
work: L
---

# Patch system over a second upstream (only when there is a change to carry)

## Description

The patch engine (anchored edits, guards, idempotence, requires/conflicts, composition check, rebase report at a pin bump) edits text against a pinned tree and is not tied to C++. It already has a language switch (c, cmake, none). What ties it to llama.cpp is the single patches/ root, the single vendor tree and the recipes that select patches for it.

## Steps

1. Python dialect for anchor matching (strip comments and docstrings so anchors cannot attach to them), with tests equivalent to the C ones.
2. engine field in patch.toml; patch selection, composition and patch-rebase-check run per engine against that engine's pinned tree.
3. A container build lane applies the composed tree as an overlay layer on the pinned image and records the resulting digest.
4. First real package: one change we actually want on the vLLM side, taken through lint, composition, build, ABBA and promotion.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-lint, patch-rebase-check and the mechanics tests pass for a Python package; the overlay image runs and its marker appears.

## Effort & Risk



## Standards



## Acceptance Criteria

One validated patch on the second engine, promoted through the same lightweight tier.

## Notes

Deliberately last and conditional: MEN01 must show the relevant code is source, and there must be a specific change to carry. Do not build this ahead of need.

## Change Log

- 2026-10-08T20:36:41.926254+00:00 (created-by): Created by agent
