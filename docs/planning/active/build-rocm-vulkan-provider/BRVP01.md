---
id: BRVP01
order: 7
plan: build-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T10:59:53.142667+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P2
---

# BuildPlan identity for HIP and Vulkan toolchains

## Description

Resume Vulkan scope (2026-10-02, GPT design session ses_62f7004ea2d24bba). Make compile identity distinguish source/build options plus Vulkan SDK, glslc/SPIR-V toolchain and generated shader capability set. RADV/AMDVLK ICD identity is runtime identity, not build identity.

## Steps

1. Fingerprint VULKAN_SDK / glslc / SPIR-V-Headers / Vulkan headers in the Vulkan BuildPlan (content-level: glslc --version + realpath, header identity).
2. Keep runtime ICD/device capability under RRVP02/RRVP03.
3. Prove an in-place glslc/toolchain change invalidates build reuse.
4. No gfx1100/gfx1201/gfx1030 Vulkan compile targets: c061 emits device-neutral SPIR-V and runtime-gates device capabilities.

## Detailed Solution & Technical Design

Capability owner: build

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

tools/bigcherry/build/builds.py; campaign/build.py; campaign/lane.py; campaign/workers.py; core/provenance.py; existing BuildPlan/build_plan_id/effective_build_id reuse and RRVP02 probe attestation.

## Validation

Same source + shader toolchain reuses one build across RADV/AMDVLK; a glslc/header change changes build_plan_id; HIP identity is unaffected by Vulkan runtime selectors.

## Effort & Risk

M / medium: cache-identity change; HIP builds must not depend on irrelevant Vulkan tools.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

When resumed, the existing build identity/reuse machinery incorporates the explicit compile-affecting projection and rejects mismatched/incomplete cache reuse; no duplicate identity system is introduced.

## Notes

Supersedes: RO04
Migration: capability-rebaseline-v3-2026-09
Successor key: build-rocm-vulkan-provider-ro04

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): GO once RRVP02 is frozen. Design is sufficient as-is: define an explicit per-backend COMPILE-AFFECTING projection of RRVP02's identity rather than hashing the entire RRVP02 object blindly (matches RO04's original stack_name/resolved_stack_fingerprint/build_stack_fingerprint three-way split). "Persist full attestation beside metadata" means the build-time RRVP02 probe record specifically — NOT RRVP03's runtime-loaded attestation, which is a separate later stage. Exact cache-identity mismatch must reject reuse (no warn-and-reuse), per this item's own already-frozen validate_reuse requirement. Execution order: ranked #8 — binds the compile-affecting projection into build/cache identity, unlocking RRVP03.

CONFIRMED via deeper repo-validated dev-gpt review (2026-09-10): fits existing code directly -- extend the existing BuildPlan/build_plan_id/effective_build_id/reuse machinery rather than introducing a second build-identity system. The codebase already distinguishes requested build identity from effective configuration/runtime-bundle identity, which this item's stack_name/resolved_stack_fingerprint/build_stack_fingerprint split maps onto cleanly. Execution order shifts to #7 in the revised sequence.

PAUSED 2026-09-10 (user directive): Vulkan is out of scope for now -- plans may continue to be updated/reviewed, implementation is paused. Design above stands for when resumed (depends on RRVP02, also paused).

Supersedes: RO04
semantic-carryforward: concrete identity/reuse gates restored 2026-09-10.
Vulkan remains paused by directive; this is not closure.

## Change Log

- 2026-09-09T10:59:53.142667+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:55.975074+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.524607+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:44.258237+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:07.424515+00:00 (updated-by): Updated: order=8, priority='P2'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.974624+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.406863+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:19:00.040229+00:00 (updated-by): Updated: order=7, section:notes
- 2026-09-10T00:25:57.108601+00:00 (updated-by): Updated: section:notes
- chg_20260910_002605_paused-all-vulkan-provider-imp_6846
- 2026-09-10T00:26:05.875846+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:27:52.861830+00:00 (updated-by): Updated: section:files, section:validation
- 2026-09-10T02:41:03.776847+00:00 (updated-by): Updated: section:steps, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_024129_build-and-external-fix-success_1105
- 2026-09-10T02:41:29.969279+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-02T12:34:38.010202+00:00 (updated-by): Updated: section:title, section:description, section:steps, section:validation, section:effort_risk
