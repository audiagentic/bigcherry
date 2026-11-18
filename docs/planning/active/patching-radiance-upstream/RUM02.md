---
id: RUM02
order: 2
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

- Use RSC13's callable inventory to enumerate source-creation blocks in rad0002, rad0009 and rad0011; record original symbols, owning components and upstream call sites before editing.
- Extract ordinary CPU implementation into owned .cpp/.h files under existing serving/moe-remote components; keep architecture template adapters close to their actual ABI needs rather than introducing a broad extension framework.
- Link owned files through one declared CMake target/source list. Include source/headers in engine build identity and release SDK inputs; do not copy them into upstream or generated build trees without hashing.
- Retain only includes, target linkage and concrete declare/route/combine/lifecycle call sites in patch.py. Remove corresponding create=True implementation payloads in the same slice.
- Keep the externally loaded plugin C ABI explicit and versioned. Internal linked engine glue uses direct typed calls and bound references; no virtual event bus or name-based registry in token execution.
- Submit extraction separately where baseline compilation is feasible, then reanchor the small hooks during the target port. If baseline and target must change together, document the unavoidable coupling.

## Detailed Solution & Technical Design

Current rad0002 embeds abi/bc_moe_remote.h, core/place/bc_moe_remote.cpp and architecture adapters as large Python raw strings; rad0009 embeds the startup bridge and emitted-option globals. Move implementation bodies, not merely the strings into another file that still gets copied into upstream. Serving already has bc_serving_v2.cmake as one source-list integration point; extend that pattern only for needed owned sources.

Suggested ownership: serving owns config/plan admission; moe-remote owns session/ticket/owner-map adapter; existing libr3/libr4d collective code owns backend execution; upstream keeps model-building and lifecycle calls. Do not add a new top-level 'extension service' component.

The MoE after_route adapter currently depends on model templates, RadOperand slicing and declared operation handles. Keep a narrow typed adapter that binds these handles during declare; do not serialize an upstream model struct or pass a map of arbitrary pointers. Preserve rad_op_reads/writes declarations, fused gather semantics and zero-work policy branches.

One direct call at a lifecycle/step boundary is acceptable. Resolve policy, layout and ownership once; do not replace compile-time kernel specialization with per-element callbacks. A smaller patch diff is not by itself a speedup; count removed duplicated policy/map work and measure only if execution changes.

## Code Samples & Guidance

Proposed linked ownership (illustrative names, not existing APIs):
```cpp
// Construct once for this admitted engine generation.
bc::RemoteExecutionState remote(admitted_contract, placement_view);
// Bind operation handles during architecture declare.
bc::RemoteOps ops = bc::declare_remote_ops(builder, geometry, remote);
// Concrete upstream call site; no parsing or profile lookup here.
bc::after_route(ctx, ops, bound_step, routed_views);
```
Keep existing versioned extern-C plugin boundary where shared objects call across it. Do not add a new generic callback table for linked code.

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

Large implementation payloads no longer live in changed patch.py packages; every owned file participates in build identity and packaging. Hook signatures are typed and lifecycle-owned. Extraction preserves existing behavior and removes duplicate implementations rather than adding alternate paths.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM01. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisite RUM01. Coordinate RCS13 inventory and RCS01/RCS08 boundaries. Cleanup opportunities: duplicated global policy getters, repeated row-layout switches and repeated include paths only when their callers are migrated in this slice.

## Change Log

- 2026-11-18T01:28:00.655993+00:00 (created-by): Created by codex
- 2026-11-18T01:37:27.808050+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:42:49.165174+00:00 (updated-by): Updated: section:effort_risk
