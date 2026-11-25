---
id: RCS22
order: 0
plan: build-radiance-components
state: deprecated
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

0. Decision gate: the RCS15 baseline says which gfx11 tree survives. 1. Row ownership model: stable row ids, one owning family, complete callback set (launch, layout, relayout, scratch, init and fini, tuning) plus concurrency, collective and chaining metadata; a foreign callback is moved to an engine-owned common library or declared as a versioned dependency, never stubbed; deterministic row order independent of load order. 2. Plugin contract: existing native rad_plugin_info and row/schema exports, ABI version and layout check; library kept loaded for every plan and in-flight operation; local symbol scope, hidden visibility, explicit versioned exports, no constructor-order dependence. 3. Identity: plugin name, version and upstream release, existing native row ABI revision and layout check; compare actual ordered rows and provider versions when reusing plans. 4. Device code: first version without relocatable device code or external device symbols; per-device module registration with failure cleanup; code-object inspection at release; no unloading under GPU work or captured graphs; no live swapping. 5. Release: one plain installed plugin/version list, whole-release rollback first. 6. Stages: 22a one independent family (link, load, register, test), 22b several families, 22c installed release against the monolithic provider with mixed-architecture starts.

## Detailed Solution & Technical Design

Design: docs/design/radiance-kernel-composition.md section 6.

## Code Samples & Guidance



## Files

engines/radiance/components (libr3, registry, loader patch), release packaging, bc-release verify

## Validation

Loader and registry tests; refusal tests (missing family, duplicate plugin name, inconsistent schema, unintended shadow, missing required provider, incompatible ABI/layout, failed module registration); symbol-visibility audit; code-object inspection; family selftests; full qualification of a release assembled from family plugins against the single-plugin release: selected row ids, order, metadata, refusals, byte outputs, resources, cold and warm performance, mixed gfx1100 and gfx1201 starts.

## Effort & Risk

Large. Risks: engine loader change, helper contracts (remote MoE, n-gram) must not change meaning, load time with several plugins.

## Standards



## Acceptance Criteria

Each family is a self-contained loadable provider with a defined host ABI, device-linking policy, complete owned row contracts and stable dispatch identity. An installed per-architecture plugin/version list records the set; native YAML specifies hierarchy and the actual loader is tested for row order and dependencies. Invalid, incomplete or incompatible compositions fail explicitly before serving. The assembled release passes equivalence qualification against the monolithic provider; no silent provider or reference fallback.

Native loader actual standard ABI/layout export and plugin name/version/upstream release, correct named hierarchy/row selection and explicit required-provider checks. No blanket duplicate-op/row-name ban across intended alternatives, no second loader, no new descriptor unless a reviewed concrete upstream capability gap. Full release verify only after discrete overlay proof.

## Notes

Gated on RCS15. Not started until the baseline decides which tree survives.

The row contract, not the translation unit, is the unit of runtime composition (design review 2026-11-25). The current libr3 link is built around a complete upstream row table with generated stubs and a registry that drops stubbed rows: that suits one incomplete monolith, not independent families. Per-family rollback is deferred until compatibility and cache invalidation are proven.

Design review (2026-11-25): the engine's loader already composes several kernel libraries by hierarchy and supports override plugins (libbcgfx12 is one), so family plugins use that mechanism: RCS23's overlays are the first family plugins. What this item adds is the sealed manifest, complete callback ownership per family, stable row identities across libraries, the composition identity and the lifecycle policy; the base libraries keep their names. The row contract, not the translation unit, is the unit of runtime composition. Per-family rollback is deferred until compatibility and cache invalidation are proven.

2026-11-25 simplification (owner: no adversaries): replace the five identities with a plugin name, a version string, the upstream release it was built against and the existing row ABI revision check; no ABI or composition hashes, no code-object digests in a sealed manifest. The release manifest is a plain list of family plugins and versions installed as a set; whole-release rollback keeps the previous release root. The row-contract comparison is a test, not a hash.

2026-11-25 bump integration: docs/design/radiance-bump-integration.md and MEN24 define shared gate order. Retain RCS15 decision and RCS23 overlay-first path; no base-library split or replacement loader solely to satisfy bump checks. The plain installed plugin/version list is release provenance, not an executable serving manifest; native YAML remains the launch authority and selects the hierarchy. RSP01 verifies actual installed providers, versioned callback/row/device contracts and priority, RCS19 checks link/export, RCS23 owns origin/adoption qualification, MEN16 compares existing build records and plain version/commit fields. Prove no fallback and no unload during live work; no duplicate family build system.

2026-11-25 second integration review: docs/design/radiance-integration-review-20261125.md is the exact-source review and controls conflicting earlier wording. Shipping pin remains b6b8751 (2.2.0), upstream observed main 1c104de4 is a development candidate, 119 commits and 327 changed files; no automatic repin or backport. No new compatibility layer, launch manifest, bump driver or identity machinery. Upstream loader already exports rad_plugin_info, rad_kernel_schema_count/at, rad_kernel_count/at and requires rad_plugin_abi_version AND rad_plugin_abi_layout. Use those exports; no new descriptor entry point, new ABI or composition hash. Shared op schema and multiple candidate rows are intended; reject inconsistent schema, duplicate plugin name, accidental shadow or missing required provider, not intended overlay/base alternatives. Native hierarchy exclusive, absent named plugin warns and may fall to other named provider, so test actual winning row and expected coverage. RCS22 optional after RCS15/23, not dependency of first bump.

2026-11-25 closed as deferred: family plugins would change the engine loader for build-time and rollback gains the family compile targets and the unit harness already give (one family builds and gates in about 2.5 minutes against a 25 to 28 minute lane build; with the ccache fix a full cached build takes 5 minutes). Reopen if per-family rollback is wanted.

## Change Log

- 2026-11-25T00:40:56.481749+00:00 (created-by): Created by lead
- 2026-11-25T00:41:36.975982+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria, section:notes
- 2026-11-25T00:52:49.430011+00:00 (updated-by): Updated: section:steps, section:validation, section:acceptance_criteria, section:notes
- 2026-11-25T01:08:31.536022+00:00 (updated-by): Updated: section:notes
- 2026-11-25T02:12:49.847394+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4621
- 2026-11-25T02:12:55.849120+00:00 (updated-by): Updated: section:notes
- 2026-11-25T02:17:04.198439+00:00 (updated-by): Updated: section:steps
- RV4634
- 2026-11-25T02:41:17.067458+00:00 (updated-by): Updated: section:notes, section:acceptance_criteria
- 2026-11-25T02:52:53.904253+00:00 (updated-by): Updated: section:steps, section:validation
- 2026-11-25T09:42:30.060875+00:00 (updated-by): Updated: section:notes
- 2026-11-25T09:43:05.806498+00:00 (state-transition): State: pending → deprecated
