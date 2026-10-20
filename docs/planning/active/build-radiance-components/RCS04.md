---
id: RCS04
order: 4
plan: build-radiance-components
state: completed
created-at: '2026-10-18T12:57:42.399279+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Extract moe-remote and moe-harness components from libr3

## Description

Create engines/radiance/components/moe-remote and moe-harness, each with component.toml, CMakeLists, README and tests; libr3 keeps only the gfx1100 engine kernel overrides. Register all components in engine.toml [components]. Move each [setting.*] to the component that reads it, with its status and evidence unchanged.

## Steps

1. Create the components and move files (git mv, history kept).
2. Update CMake so every component builds against the engine installed from the lane build; outputs keep their runtime paths.
3. Update the pin-bump components-build, component-plugin-abi and component-tests gates to iterate all components.
4. Re-point evidence mechanism ids (libr3:R3_MOE_FP16 -> moe-backend-gfx1100:R3_MOE_FP16 etc.) and the activation markers; the RAD06 evidence must be merged first and then re-pointed as a metadata move, not re-run (the mechanism and output are unchanged).
5. Update docs/reference/engines/radiance and the model page setting table.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

bigcherry build of the radiance lane produces the same runtime home contents as before; patch-lint and the component lint pass; one Flash-Next smoke and one A/B show no change.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Depends on RCS01-RCS03 and the RAD06 evidence PR. Moving a setting between components is a wiring change and does not invalidate its proof.

Added 2026-10-19: (1) the build profile chooses which components are built (stock = none; a MoE profile = moe-remote plus the backends for the cards present), so engine.toml [components] lists all and the recipes' build profiles select; the pin-bump components-build gate builds the selected set. (2) Rename decision (owner asked why 'helper'): the process is an expert worker; rename binary r3_moe_worker -> r3_moe_worker and R3_MOE_WORKER_* -> R3_MOE_WORKER_* in this same change, migrating every caller, doc and evidence mechanism id at once (no shims). (3) Takes the 1.8.0 pin: built-against v1.8.0, the fp8a8 native rewrite is shelved in native-shelved/.

## Change Log

- 2026-10-18T12:57:42.399279+00:00 (created-by): Created by agent
