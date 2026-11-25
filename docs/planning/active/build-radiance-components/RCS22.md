---
id: RCS22
order: 0
plan: build-radiance-components
state: pending
created-at: '2026-11-25T00:40:56.481749+00:00'
breadth: ''
skill: advanced
created-by: lead
work: L
priority: P2
---

# Loadable kernel-family plugins composed from a manifest (decided at the RCS15 gate)

## Description

Replace one plugin per architecture by one small plugin per kernel family per architecture, composed at load from a manifest, so a family builds, gates, versions and rolls back on its own.

## Steps

0. Decision gate: the RCS15 baseline says which gfx11 tree survives. 1. Row ownership model: stable row ids, one owning family, complete callback set (launch, layout, relayout, scratch, init and fini, tuning) plus concurrency, collective and chaining metadata; a foreign callback is moved to an engine-owned common library or declared as a versioned dependency, never stubbed; deterministic row order independent of load order. 2. Plugin contract: one versioned descriptor entry point; library kept loaded for every plan and in-flight operation; local symbol scope, hidden visibility, explicit versioned exports, no constructor-order dependence. 3. Identity: host_abi_id, row_contract_id, device_abi_id, artifact_id, composition_id; plan-cache keys on composition and dispatch identity. 4. Device code: first version without relocatable device code or external device symbols; per-device module registration with failure cleanup; code-object inspection at release; no unloading under GPU work or captured graphs; no live swapping. 5. Release: one sealed atomic manifest, whole-release rollback first. 6. Stages: 22a one independent family (link, load, register, test), 22b several families, 22c sealed release against the monolithic provider with mixed-architecture starts.

## Detailed Solution & Technical Design

Design: docs/design/radiance-kernel-composition.md section 6.

## Code Samples & Guidance



## Files

engines/radiance/components (libr3, registry, loader patch), release packaging, bc-release verify

## Validation

Loader and registry tests; refusal tests (missing family, duplicate row, incompatible ids, failed module registration); symbol-visibility audit; code-object inspection; family selftests; full qualification of a release assembled from family plugins against the single-plugin release: selected row ids, order, metadata, refusals, byte outputs, resources, cold and warm performance, mixed gfx1100 and gfx1201 starts.

## Effort & Risk

Large. Risks: engine loader change, helper contracts (remote MoE, n-gram) must not change meaning, load time with several plugins.

## Standards



## Acceptance Criteria

Each family is a self-contained loadable provider with a defined host ABI, device-linking policy, complete owned row contracts and stable dispatch identity. A sealed per-architecture manifest determines exact versions, dependencies and row order. Invalid, incomplete or incompatible compositions fail explicitly before serving. The assembled release passes equivalence qualification against the monolithic provider; no silent provider or reference fallback.

## Notes

Gated on RCS15. Not started until the baseline decides which tree survives.

The row contract, not the translation unit, is the unit of runtime composition (design review 2026-11-25). The current libr3 link is built around a complete upstream row table with generated stubs and a registry that drops stubbed rows: that suits one incomplete monolith, not independent families. Per-family rollback is deferred until compatibility and cache invalidation are proven.

Design review (2026-11-25): the engine's loader already composes several kernel libraries by hierarchy and supports override plugins (libbcgfx12 is one), so family plugins use that mechanism: RCS23's overlays are the first family plugins. What this item adds is the sealed manifest, complete callback ownership per family, stable row identities across libraries, the composition identity and the lifecycle policy; the base libraries keep their names. The row contract, not the translation unit, is the unit of runtime composition. Per-family rollback is deferred until compatibility and cache invalidation are proven.

## Change Log

- 2026-11-25T00:40:56.481749+00:00 (created-by): Created by lead
- 2026-11-25T00:41:36.975982+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T00:52:49.430011+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
- 2026-11-25T01:08:31.536022+00:00 (updated-by): Updated: section:notes
