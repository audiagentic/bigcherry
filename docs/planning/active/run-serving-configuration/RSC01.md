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

# C++ serving component: schema, strict loader, resolved manifest, explain (one implementation in the release package)

## Description

Owner direction 2026-11-01/02: launching must be easy to run and manage, use standard formats, give maximum customisation with minimum complexity, and contain no hand-picked values; everything not a genuine user/policy choice is discovered, derived or measured, per environment and layout. Today ~450 knobs live on 8 surfaces (launcher flags and literals, layouts.toml, 86 component settings, 73 R3_* env vars, 116 radiance options, recipes, 63 lab JSON configs) with no single resolver or explanation. This item defines the typed TOML schema and the classification of every setting, and produces an effective-configuration manifest with per-value provenance and an explain command, without changing how serving runs.

## Steps

1. New bigcherry radiance component engines/radiance/components/serving: a static library built with the engine, in C++ only.
2. One versioned, typed schema in C++ covering service intent, layout, memory, the engine section and per-helper sections. Unknown keys are errors. Types, ranges and enums are checked, and cross-field rules apply (helpers match the layout; budgets fit the detected VRAM and RAM). A JSON Schema is exported from it for docs and tooling, never maintained by hand.
3. Strict loader. Input files are TOML (vendored header-only toml++); the precedence is schema defaults with a stated origin < discovered facts < site < profile < `--set`. Hard caps can only tighten. Nothing is read from shell scripts or environment variables, and there is no silent default.
4. Output: a resolved manifest (canonical JSON via nlohmann/json, which is already linked, plus a content hash) with per-value provenance.
5. Linked into the engine (RSC07), r3_moe_worker (RSC06) and the bc-serve supervisor (RSC02). `bc-serve --validate|--explain [--json]` is the only explain implementation.
6. The Python `bigcherry config` commands become thin clients that call bc-serve; there is no schema or resolver in Python.
7. config/serving-settings.toml stays as the reference inventory the C++ schema is built from. It is not read at run time.
8. Equivalence test: for flashnext-2xtx-rdna2-tail, the manifest carries exactly the values today's launcher passes. The test extracts the launcher's argv and envs, and is replaced when RSC02 deletes the launcher.

## Detailed Solution & Technical Design

Owner decisions 2026-11-02 (RV4224):
- Start-up is a strict loader plus a manifest that each process loads itself, not a command-line converter. The engine reads it through a managed --manifest patch (option a).
- The loader lives in bigcherry's radiance C++ code, so the release package (bin/radiance, radiance_home/helpers/r3_moe_worker, plugins and bc-serve) is one language with one implementation, not split between C++ and Python.

The first Python resolver (serving_config.py, config_cmds.py) is dropped. Its inventory and parity-test idea carry forward. Layer rules and provenance fields are unchanged: facts and hard constraints are types, not preferences; the calibration cache fills only `auto` values and never overrides constraints; experiment arms stay in a separate namespace.

## Code Samples & Guidance



## Files



## Validation

Card-free C++ tests (component test target): schema types and ranges, unknown-key rejection at every level, precedence, hard-cap refusal, missing-required refusal, provenance completeness, canonical manifest hash stability, and launch equivalence with today's launcher for the default layout. `bc-serve --explain` runs on the real host.

## Effort & Risk



## Standards



## Acceptance Criteria

One C++ schema and loader serve the engine, the helpers and the supervisor. Unknown keys fail closed, and there are no fallbacks. The manifest is hashed and explainable and equals today's launch values. Python carries no copy of the schema or resolver.

## Notes

Source: GPT configuration assessment 2026-11-02 (gateway request req_main_fa1c9eed1169), reviewed. Runs in parallel with RHP stage 1; does not block it. Owner rules: [no hand-picked resource constants], no backward-compatibility shims.

## Change Log

- 2026-11-02T00:52:58.694609+00:00 (created-by): Created by agent

## Ledger-events


- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:53:44.556553+00:00 (updated-by): Updated: section:ledger-events

- chg_20261102_035356_serving-start-up-plan-one-str_4985
## Reviews

- RV4224
- 2026-11-02T03:51:36.726474+00:00 (updated-by): Updated: section:title, section:steps, section:detailed_solution, section:validation, section:acceptance_criteria
- 2026-11-02T03:52:49.575691+00:00 (updated-by): Updated: section:title, section:steps, section:detailed_solution, section:validation, section:acceptance_criteria
- 2026-11-02T03:54:00.264966+00:00 (updated-by): Updated: section:ledger-events
