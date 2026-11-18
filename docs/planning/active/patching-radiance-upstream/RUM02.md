---
id: RUM02
order: 4
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:00.655993+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P1
work: L
---

# Extract injected BigCherry implementations into owned components and narrow hooks

## Description

Move large source-creation literals and duplicated policy from patch.py into ordinary owned source files. Keep only concrete typed integration hooks; reuse RCS13 callable inventory and RCS01 ownership mapping. Preserve behavior before mixing extraction with upstream ports.

## Steps

- Complete RUM04's native-boundary decision and removal first; do not extract rad0009/rad0011 bodies whose current purpose is obsolete.
- Build the keep/delete map from actual callers and RSC13 inventory: keep remote execution/admission, measured memory/plan checking, exact rank constraints, shared group adapter and qualified diagnostics.
- Move only surviving CPU implementation and tests from raw patch strings into existing serving/moe-remote owners and declared CMake targets.
- Keep necessary architecture template adapters as small owned headers with typed operand/handle views; preserve RAD_OP dependency declarations and fused route/combine semantics.
- Reduce patch.py to concrete linkage/include/call-site changes. Remove the old created payload in the same change, and hash/link all owned source inputs.
- Separate moved lines from real deletion and verify a no-behavior-change slice only where extraction is still warranted.

## Detailed Solution & Technical Design

Extraction is not the first objective: preserve fewer concepts. rad0009 startup translation should mostly disappear through upstream YAML; measuring/expected-plan hooks from rad0011 may survive. Classify first, delete native duplication second, extract actual BigCherry behavior third.

Retain ordinary direct typed calls and the existing extern-C plugin contract where needed. Do not add an extension event bus, service locator, callback factory or generic lifecycle manager. The first supported case is one engine and disjoint helpers, so full multi-engine helper ownership is not a prerequisite.

Source movement improves patch review but does not shrink compiled code. Net reduction comes from removed startup bridge/projections, proven upstream replay/governor reuse and redundant internal callers, not reorganizing the same text.

## Code Samples & Guidance

Sequence:
```text
native ownership classification -> remove obsolete bridges/mirrors
-> enumerate actual surviving callers -> extract surviving implementation
-> minimal target attachment sites -> focused boundary proof
```
See Astra removal map for exact surviving versus removable source clusters.

## Files

- engines/radiance/patches/rad0002_moe_remote_handoff/patch.py
- engines/radiance/patches/rad0009_serving_manifest/patch.py
- engines/radiance/patches/rad0011_plan_memory/patch.py
- engines/radiance/components/serving/bc_serving_v2.cmake
- engines/radiance/components/moe-remote/CMakeLists.txt
- engines/radiance/engine.toml

## Validation

Canonical patch apply/idempotence for the changed hooks, one touched-component build and the existing callback-boundary tests. Compare old/extracted semantic projections or outputs for one fake-engine case; do not run GPU performance tests for source relocation alone.

## Effort & Risk

L: template adapters and callback visibility can create hidden coupling. Extract in small component-bound slices, preserve symbol/ABI boundaries, and avoid introducing a new abstraction merely to relocate code.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

Surviving large implementations in changed packages move only where doing so reduces repeated upstream attachments or build drift. Obsolete implementations are deleted first. Owned files participate in build identity/packaging; typed hooks preserve behavior. Moving every payload or relocating on the old pin is not an initial-start gate.

## Notes

Prerequisites RUM01/RUM03/RUM04. The Astra source review ASTRA-CONSOLIDATION.md supersedes extraction-first ordering. Coordinate RCS13 inventory and RCS01/RCS08 component ownership. Remove obsolete native bridges/replay/policy before moving surviving code; movement is not net deletion.

## Change Log

- 2026-11-18T01:28:00.655993+00:00 (created-by): Created by codex
- 2026-11-18T01:37:27.808050+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:42:49.165174+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:56:30.966408+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:16:37.229196+00:00 (updated-by): Updated: order=4, section:steps, section:detailed_solution, section:code_samples, section:notes
- 2026-11-18T02:22:18.965936+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:25:00.590374+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:25:21.136200+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:28:04.354691+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
