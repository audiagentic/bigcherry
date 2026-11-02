---
id: RSC01
order: 1
plan: run-serving-configuration
state: pending
created-at: '2026-11-02T00:52:58.694609+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P0
work: M
---

# Serving configuration schema, full settings inventory, effective manifest and explain (no serving-path change)

## Description

Owner direction 2026-11-01/02: launching must be easy to run and manage, use standard formats, give maximum customisation with minimum complexity, and contain no hand-picked values; everything not a genuine user/policy choice is discovered, derived or measured, per environment and layout. Today ~450 knobs live on 8 surfaces (launcher flags and literals, layouts.toml, 86 component settings, 73 R3_* env vars, 116 radiance options, recipes, 63 lab JSON configs) with no single resolver or explanation. This item defines the typed TOML schema and the classification of every setting, and produces an effective-configuration manifest with per-value provenance and an explain command, without changing how serving runs.

## Steps

1. Inventory and classify every setting across all surfaces (machine-readable, in git).
2. Typed TOML schema with unknown-key rejection; site/profile/CLI layers.
3. Resolver producing the effective manifest with provenance from today's sources (reproducing the current launch exactly).
4. `bigcherry config validate|explain` and `serve --explain --dry-run` (read-only).

## Detailed Solution & Technical Design

Layers, in typed precedence (facts and hard constraints are types, not overridable preferences): compiled engine contract < container metadata < discovered environment < site TOML (hard caps, permitted hardware, security) < serving profile (objective, sessions, context, retention, quality) < explicit CLI (`--set key=value`, typed and validated) ; the calibration cache fills only values left `auto` and never overrides constraints; experiment arms are a separate namespace used only by the lab runner. Every setting is classified as user/policy, discovered, derived, measured/tuned, experiment/diagnostic, or retire (GPT assessment 2026-11-02: radiance 70/4/13/14/14/1, components 2/10/9/42/18/5), including patch-owned settings such as R3_MOE_WORKER_SKIP_OWNED. Each resolved value records key, value, origin, inputs, constraints, confidence (exact/measured/estimated), evidence, override and lifecycle (startup-only, step-boundary, live). Component lifecycle statuses (experimental..production) stay and keep gating production use.

## Code Samples & Guidance



## Files



## Validation

Card-free tests: schema types, unknown keys, precedence, hard-cap conflict refusal, provenance completeness; the manifest resolved from today's launcher inputs equals today's actual engine args and helper environments byte for byte.

## Effort & Risk



## Standards



## Acceptance Criteria

Every setting classified with an owner; schema and explain in place; the resolved manifest reproduces the current launch exactly.

## Notes

Source: GPT configuration assessment 2026-11-02 (gateway request req_main_fa1c9eed1169), reviewed. Runs in parallel with RHP stage 1; does not block it. Owner rules: [no hand-picked resource constants], no backward-compatibility shims.

## Change Log

- 2026-11-02T00:52:58.694609+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:53:44.556553+00:00 (updated-by): Updated: section:ledger-events
