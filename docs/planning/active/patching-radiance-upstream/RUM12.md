---
id: RUM12
order: 0
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-21T23:41:27.558357+00:00'
breadth: ''
skill: advanced
created-by: claude
work: M
priority: P1
---

# Named start reuses the last measured plan; stat-based model identity

## Description

Starts must be quick and reuse what is already measured: a named serve/start reuses the most recent good measured manifest for its config instead of measuring again, and never hashes a whole model.

## Steps

- Move the named-config index from the run dir to a work-root store keyed on the resolved settings, ordered cards and VRAM, power profile, config inputs and model path/size/mtime; drop the engine, helper and bc-serve builds and the installed runtime path from the key.
- Identify the model by canonical path, size and mtime; never read it.
- A stored plan the replay checks refuse is measured once afresh and replaces the entry; an engine that fails to come up on a stored plan drops the entry.
- Measure the second named start of flashnext-r9700 on cards against the replay time.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Hardware-free: stored_manifest_cache_test (key, stat identity, store), serving_test (cross-run-dir reuse without measuring, fall-through to one fresh measurement on a different engine build), session_test (entry dropped when the engine fails on a stored plan). On cards: the second named start of flashnext-r9700 about the replay time (44 s with the work root), no measuring, no fingerprint.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes



## Change Log

- 2026-11-21T23:41:27.558357+00:00 (created-by): Created by claude

## Acceptance

A second named start with unchanged inputs reuses the stored manifest from any run dir and any build, runs no measuring child and reads no model bytes; a stored plan that cannot be reused measures once and replaces it; each relaxed check and what still protects it is listed in the PR.
- 2026-11-21T23:41:35.527696+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance
