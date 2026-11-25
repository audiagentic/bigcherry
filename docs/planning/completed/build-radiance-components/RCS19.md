---
id: RCS19
order: 0
plan: build-radiance-components
state: completed
created-at: '2026-11-25T00:40:42.311181+00:00'
breadth: ''
skill: intermediate
created-by: lead
work: S
priority: P1
---

# Kernel composition gates: provenance and anchors, compile, link and export, qualification; source ownership records and patched-source freshness

## Description

Managed patches and native rewrites both transform the same upstream kernel text. The gfx1201 fused-tile patch passed every gate and still could not build because its edits and the merged gfx1100 rewrites collided (anchors vanished, a type was defined twice); only a full lane build found it. Make every kernel change a unit with one owner per line of source, checked by a machine in seconds.

## Steps

1. Gate 1 (seconds, offline): apply the ordered patch set (bc patches included) to the pinned source, run the real rewrite check against it; at least one match per declared alternative group (exactly one only for explicitly exclusive variants); declared dependencies; region overlap; existing source-tree identity. 2. Gate 2: compile the affected translation units for gfx1100 and gfx1201 (host and device passes, architecture macros, explicit template instantiations) with failing fixtures: duplicate type, repeated template default argument, device-only definition clash, conflicting conditional compilation. 3. Gate 3: link the affected family or a contract fixture and check duplicate definitions, unresolved symbols, exported names, external device symbols. 4. Gate 4: resource, byte identity, selftest and replay for the affected family and the units that include a changed shared header. 5. Ownership records in patch metadata: affected units (expanded through the include graph), owned regions, consumed anchors. 6. Naming: architecture suffix only where two architectures must both define a name; uniqueness proven by gate 3; upstream and ABI names never renamed. 7. Freshness: verify the merged patched-input CMake dependency (#1546), with a test that changing an upstream header regenerates output without deleting the build directory. 8. First customer: the gfx1201 fused tile through the standard lane.

## Detailed Solution & Technical Design

Design: docs/design/radiance-kernel-composition.md section 4 (S1 to S5).

Integration: tools/bigcherry/patch/compose_check.py existing materializer/production_ids/gate1 and compile gates reused by release/pin_bump.py and engine_adapters/radiance_bump.py. Move authoritative rewrite gate to exact selected patched tree after coverage/materialization, before costly compile; pristine triage advisory only for transformed sources, retain pristine ABI/coupling checks. Exact target/source/experiment/patch order and existing source identity checks, each supported composition, patch-created sources and missing/unselected package distinguished. Use compiler depfiles/canonical compile commands and actual host/device macros/includes/arch; changed indirect/generated headers and upstream-only changes invalidate dependents; unknown widens compile set. Follow RCS15 supported gfx11/gfx1030 providers, not forced gfx12 compilation. Gate1 ownership/alternative, Gate2 real host/device, Gate3 link/export/row selection and Gate4 correctness/resource/replay via RCS25 discrete harness distinct. Structured pass/fail/skipped/unverified/not-applicable, required skipped cannot count as pass.

## Code Samples & Guidance



## Files

tools/bigcherry/release (patch checks), tools/tests/patch, engines/radiance/patches, skills/bigcherry-patch-author, docs/design/radiance-kernel-composition.md

## Validation

Fixtures: a vanished anchor, a doubled anchor and an unmatched required alternative group or explicitly exclusive group with multiple matches fail gate 1; a duplicate type, a repeated template default and a device-only definition clash fail gate 2 and not gate 1; every existing patch set passes; the regeneration test passes; both architectures lane-build with the gfx1201 package applied and the gfx1100 resource records unchanged against the shipped release.

## Effort & Risk

Small. Risk: the gate needs the bc patches applied first to avoid false failures; keep the allowed-failure list explicit and tested.

## Standards



## Acceptance Criteria

Every managed patch set applies to the pinned source, every required anchor resolves under its declared variant, and every transformation has ownership and dependency metadata. Compile and link checks reject invalid declarations, duplicate definitions and symbol conflicts for both architectures before a lane build. No family is qualified until the affected and dependent kernels pass their correctness and resource gates.

Actual pin-bump path (not only manual CLI) stops at correct composition before expensive lane on known collision, vanished/duplicate anchor, overlap/alternative failure; patch-created file succeeds only for exact selected composition. Host/device/default/duplicate symbol fixture, changed indirect/generated header and stale CMake input regenerate, missing compiler/incomplete closure not clean, actual link/export and family qualification. Bind exact selection+source+gate identities to MEN16 resume; no parallel materialization if exact tree proof allows reuse. Existing RCS19 gate obligations retained.

Match real rewrite alternative semantics, reject zero required matches and ambiguous explicitly exclusive groups, keep valid multiple declared edits; test real candidate composition, upstream-only indirect headers, no missing compiler => pass. Verify existing #1546 change with incremental upstream-header test, then RCS25 affected discrete correctness and final integrated gate only at release.

## Notes

Stage 0 of the native RDNA3 migration brief (additive gfx12 body, distinct identifiers) is the same work.

2026-11-25 first customer: the gfx1201 fused-tile package. Resolved on a branch by keeping the patch's shared-header edits, making the five libr3 header blocks find-1-or-0 with a paired identity block (the rewriter's paired-run rule keeps upstream drift failing), renaming the patch's own tile to the gfx12-suffixed names and wrapping the tile and its dispatch so the gfx11 build ignores them. Lane build on both architectures: gfx1100 records unchanged against the shipped release (0 added, 0 removed, 0 changed), gfx1201 adds 11 fused kernels with 0 spills and the 44 plane variants unchanged. The anchor gate (step 1) would have caught this before the lane build.

Review (2026-11-25): the existing rewrite checker accepts an optional group when at least one block matches; exactly-one-of-pair and region-overlap checks are new tooling, not the existing check run earlier. Whole-file ownership is exclusive (a file owned by an overlay is edited by no patch or rewrite); region ownership applies only to unowned files. Assignment: bigcherry-2 builds gate 2a (syntax-only host and device passes for the affected units on both architectures, fixtures: duplicate type, repeated template default argument, device-only definition clash, and the undisguised gfx1201 collision) and gate 2b (real compile); the lead builds gate 1, the ownership records and the freshness dependency.

2026-11-25 cross-programme bump review: docs/design/radiance-bump-integration.md is the shared implementation-order and compatibility specification; MEN24 owns sequence and timing, domain implementation remains here. No duplicate checker, bump driver, manifest launch or legacy fallback. Current pin-bump.run_phase_drift calls radiance_bump.source_drift_checks/rewrite_drift on pristine upstream, not necessarily compiled composition. In-flight node/compose-check 28d5a315f and node/rcs19-freshness 5344042687 already implement first slices; check merge status, reuse; no duplicate checker. Observed compose checker uses local textual include walk and Report.ok excludes skipped; complete actual coverage and explicit gate states.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. #1546 real CMake freshness dependency merged; reuse and verify, do not implement digest freshness again. #1554 rad0065 promoted; r4d_gemm_8a8_prefill.h changed upstream candidate, indirect dependents need gates even without IQR .hip change. Current source_drift_checks still pristine; compose checker branch not yet merged in audit. Correct alternative semantics: at least one match per existing paired-run rule; exactly-one only if variant group explicitly declares exclusivity, deliberate multiple valid edits allowed. Real checker/metadata fixture, no blanket exact-one. RCS25 discrete unit harness after affected compile, no full lane prerequisite.

2026-11-25 lean build sweep at 79c61a758. One existing CMake/Ninja build graph, one canonical matched release lane, existing tests and release records. No new cache, build system, manifest, hash scheme or generic framework. Reuse actual evidence; only test changed behavior and dependency closure. #1546 libr3 patched-source freshness merged but libr2 CMakeLists.txt:28–36 calls the shared r3_rewrite.py and only watches rewrites.txt, units.txt and one native rule; a patched source/header, rewriter change or new native rule can leave stale generated code in existing directories. libr3 also omits explicit configure dependency on r3_rewrite.py and exclude.txt; native rule glob does not track added/removed files. Fix actual input lists in both existing CMake paths, including upstream unit listing and shared generator/declaration inputs if not already tracked. Use normal CMake dependencies (CONFIGURE_DEPENDS for changing glob membership where necessary); r3_rewrite.py already avoids rewriting unchanged files and removes stale source outputs, retain it. One small incremental test/real affected-unit build proves changed header/rule regenerates and unchanged generated files do not rebuild; cover libr2 path too, no digest watcher or new generator. Actual pin-bump rewrite check remains on pristine source: integrate the existing compose gate after exact selected patch materialization, reuse checked tree. Do not run syntax-only and real compile twice for the same changed unit when real compile covers both host/device branches; syntax-only useful as cheap early diagnostic. BU03 owns separate warm/cold discrepancy.

2026-11-25 small implementation slice: both libr2/libr3 now watch actual upstream source/header glob membership and rewrite inputs; stub generator is a link dependency. Existing patched source copied into isolated scratch and both CMake/Ninja configurations passed (libr2 3 units, libr3 115). On a changed upstream header ninja build.ninja regenerated both copies; unchanged repeat did not rerun CMake or touch the copy; added/removed header detected. Generated link rules contain r3_gen_stubs.py dependency. First source fixture lacked four patch-created headers and correctly refused libr3 configure; used complete existing patched source for this proof, not a claim of fresh selected recipe composition or full compile. #1552 composition checker now merged; remaining work is integrate exact selected patched tree into pin-bump, cover shared libr2 rewriter, upstream/owned dependency changes, use canonical CMake compile options/targets rather than reconstructed flags, and distinguish skipped from passed. No GPU or full plugin build in this slice.

2026-11-25 closed. Gates: patch-compose-check with gates 1 (anchors on the patched source), 2a (syntax-only host and device, gfx1100 and gfx1201), 2b, 3 (#1552); per-patch unit ownership, the shared-unit record and the naming lint (#1580); the pin-bump compile phase runs gates 1 and 2a before components-build, stopping with NATIVE_REWRITE_DRIFT or COMPOSITION_COMPILE_FAILED, and a missing compiler stops, never passes (#1584); patch-author and bump-radiance skills carry the gates. Exact-identity binding to resume records was not built (no fingerprinting machinery, owner direction): git commit ids and plain comparisons only.

## Change Log

- 2026-11-25T00:40:42.311181+00:00 (created-by): Created by lead
- 2026-11-25T00:41:22.694997+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T00:42:02.400302+00:00 (updated-by): Updated: section:notes
- 2026-11-25T00:52:35.276328+00:00 (updated-by): Updated: section:title, section:steps, section:validation, section:acceptance_criteria
- 2026-11-25T01:08:36.372161+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4605
- 2026-11-25T02:04:12.989071+00:00 (updated-by): Updated: section:notes, section:detailed_solution, section:acceptance_criteria
- RV4631
- 2026-11-25T02:39:54.073606+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- 2026-11-25T02:52:41.601297+00:00 (updated-by): Updated: section:steps, section:validation
- 2026-11-25T05:20:55.808306+00:00 (updated-by): Updated: section:notes
- 2026-11-25T05:32:23.302601+00:00 (updated-by): Updated: section:notes
- 2026-11-25T09:14:15.155854+00:00 (updated-by): Updated: section:notes
- 2026-11-25T09:14:28.045520+00:00 (state-transition): State: pending → completed
