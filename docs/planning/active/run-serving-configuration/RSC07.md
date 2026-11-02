---
id: RSC07
order: 13
plan: run-serving-configuration
state: pending
created-at: '2026-11-02T03:52:04.869272+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P0
---

# Engine --manifest: managed radiance patch loading the engine section through radiance's own option table

## Description

Owner decision 2026-11-02 (recorded on RSC01), option (a): the engine reads its configuration from the resolved manifest, not from generated flags. Radiance is upstream (core/config.cpp, about 229 options, already linked to nlohmann::json). A managed patch adds `--manifest <path>`, which loads the manifest's engine section.

## Steps

1. A managed patch (bigcherry-patch-author workflow, fail-closed anchors) on core/config.cpp adds `--manifest`. Each key in the engine section maps to an entry in radiance's existing option table (same names without the leading dashes) and goes through the same parse and validation code as the CLI, so behaviour is identical.
2. Unknown keys, type errors, and mixing --manifest with other configuration flags are all errors.
3. Engine-side plugin settings (bc_moe_remote hook, n-gram, record/heat) come from the same manifest via RSC06's typed structs.
4. The patch carries its own tests (option-table coverage: every manifest key resolves to an option) and a rebase check.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Patch verify (apply, idempotence, anchors). Card-free: every option accepted by the CLI is accepted as a manifest key, with the same parsed Config. On hardware: a manifest launch versus today's argv launch gives byte-identical single-stream text and step time within noise.

## Effort & Risk



## Standards



## Acceptance Criteria

The engine starts with only `--manifest` (and `--role`). The parsed configuration equals the argv equivalent. Unknown keys fail closed. The patch is managed and rebases cleanly.

## Notes

A patch on upstream is a carried cost that we accept, so that no command-line converter remains. Keep the patch thin: reuse the option table, add no second schema in C++.

## Change Log

- 2026-11-02T03:52:04.869272+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_035356_serving-start-up-plan-one-str_4985
- 2026-11-02T03:54:15.306393+00:00 (updated-by): Updated: section:ledger-events
