---
id: RRBC02
order: 9
plan: run-reusable-build-campaign
state: pending
created-at: '2026-09-09T10:59:03.207047+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# Replace platform.vulkan-linux's targets placeholder with real device/ICD identity

## Description

Replace Vulkan platform targets placeholder with normalized runtime device/ICD identity and capability provenance, distinct from compile-target identity.

## Steps

Define vendor/device class/UUID policy, ICD/driver/API/features/capability/shader digest; include in platform/build/runtime provenance; keep targets legacy-only or remove semantic use; add descriptor/mismatch tests; require driver/ICD and shader-cache state on Vulkan Experiment Contracts and never pool corpora across ICDs.

## Detailed Solution & Technical Design

HIP AMDGPU_TARGETS remains compile identity. Vulkan identity is runtime capability/provenance; ordinals/BDF/paths are evidence only. RRBC02 is single identity authority consumed by campaigns/contracts/replay.

## Code Samples & Guidance



## Files

recipes/config/provenance/build/campaign identity and Vulkan tests.

## Validation

Config parsing, stable identity under device reorder, missing ICD/API rejection, HIP compatibility, real stock artifact identity/capability digest, per-ICD evidence separation.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Vulkan artifacts carry normalized stable runtime identity and capability digest; driver/ICD is mandatory provenance; no ordinal-based reusable identity or cross-ICD pooling.

## Notes

Supersedes: RE33
Migration: capability-rebaseline-v3-2026-09
Successor key: run-reusable-build-campaign-re33

External dev-gpt holistic review (2026-09-10, req_9f60aaa2ae5f4b88): RE-SCOPE BEFORE CODING. The predecessor (RE33) content wrongly mixes declarative platform constraints, observed software identity, and device capabilities into one item. Re-scope this item to own ONLY Vulkan platform constraints/selection policy (vendor/device-class constraints, ICD selection policy, minimum Vulkan API/capability requirements) — it must consume RRVP02's identity/capability outputs, never invent a parallel Vulkan identity digest. New dependency added: RRBC02 -> RRVP02 (this item cannot start before RRVP02 is frozen). Remove the Vulkan `targets` config field entirely rather than retaining it as a compatibility placeholder (project doctrine: no legacy shims). RRBC02 references RRVP02 fingerprints/snapshots directly. Naming resolution: keep RRBC02 for the RE33 lineage and RRVP02 for the RO03 lineage — do not rename/merge one into the other, they are genuinely separate authorities (RRBC02 = declarative constraint/policy; RRVP02 = probed ResolvedStackIdentity + CapabilitySnapshot). Experiment corpora must retain an explicit ICD/driver identity partition per EC15's folded-in requirement — never pool across ICDs. Execution order: ranked #10, after RRVP02/BRVP01/RRVP03 (moved later than RE33's original position since it now depends on RRVP02).

CORRECTION from deeper repo-validated dev-gpt review (2026-09-10): the prior "remove targets entirely" recommendation was incomplete/wrong. Platform.targets currently feeds BOTH _resolve_architectures() / candidate-generation identity AND HIP compile targets -- it is not a pure legacy placeholder that can simply be deleted. This item therefore needs a backend-aware CONFIG SCHEMA MIGRATION that separates HIP compile/catalog architectures from Vulkan platform/device constraints, not a field removal. RRVP02 remains the sole probed runtime/software identity authority (that overlap resolution still holds); this item still owns declarative Vulkan platform constraints/policy and still depends on RRVP02. Execution order shifts to #9 in the revised sequence (after RRVP03, before RHA01).

PAUSED 2026-09-10 (user directive): Vulkan is out of scope for now -- plans may continue to be updated/reviewed, implementation is paused. Design above (backend-aware config schema migration separating HIP compile identity from Vulkan platform constraints, depends on RRVP02) stands for when resumed.

## Change Log

- 2026-09-09T10:59:03.207047+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:16:11.720645+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.480385+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:07:51.075586+00:00 (updated-by): Updated: section:notes
- 2026-09-10T00:08:08.881022+00:00 (updated-by): Updated: order=10, priority='P3'
- chg_20260910_000828_reviewed-and-re-planned-all-pe_3612
- 2026-09-10T00:08:28.947678+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.333474+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:19:06.494411+00:00 (updated-by): Updated: order=9, section:notes
- 2026-09-10T00:25:59.745578+00:00 (updated-by): Updated: section:notes
- chg_20260910_002605_paused-all-vulkan-provider-imp_6846
- 2026-09-10T00:26:05.866232+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:27:59.179507+00:00 (updated-by): Updated: section:files, section:validation
- 2026-09-10T03:37:58.055369+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_033817_repaired-vulkan-provenance-and_3027
- 2026-09-10T03:38:17.192078+00:00 (updated-by): Updated: section:ledger-events
