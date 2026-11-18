---
id: RUM10
order: 10
plan: patching-radiance-upstream
state: pending
created-at: '2026-11-18T01:28:59.863720+00:00'
breadth: ''
skill: advanced
created-by: codex
priority: P3
work: L
---

# Add multi-model fleet helper support only after the single-model migration

## Description

Optional later enhancement, not a prerequisite for the upstream bump. Add fleet-wide helper ownership, per-card power conflict handling, park/reload/reclaim and group-id reuse after stable single-model parity.

## Steps

- Do not implement during the initial bump; wait for RUM09 single-model acceptance and an actual multi-model requirement.
- Add fleet-level helper reservations and per-model extension identities using the RUM05/RUM06 ownership already present.
- Integrate lifecycle admission with Fleet::load_op, park_op, restore_op, reload and stop_all rather than constructing another scheduler.
- Extend reload identity to native effective settings plus extension generation; drain old helper work before reclaim or group-id reuse.
- Support only one named two-model case first. Add sharing/priority/park modes separately when requested and measured.

## Detailed Solution & Technical Design

This is a later enhancement, not a reason to retain a broad singleton service manager in the initial port. Upstream already owns concurrent engines, model references, card flex pools and hot reload. BigCherry should attach bounded per-engine helper sessions and external reservations to those lifetimes. Power is a physical-card policy: concurrent models cannot silently install incompatible device settings. Refuse incompatible sharing rather than introducing a power arbiter before a use case exists.

A mutable authored x-bigcherry section does not participate in upstream reload's entry.args comparison. Compare the admitted extension/native projections and replace a model generation only after draining its parent tickets. A native args match alone must not keep a stale owner map or helper namespace alive.

## Code Samples & Guidance

Initial migration uses exact original YAML byte identity and matched build/default identity (RUM04); comment-only cache misses are acceptable. For an actual later multi-model requirement, decide immutable per-model binding and semantic reuse using the existing identities. Do not require normalized_native_config or a YAML serializer.

Single-model Fleet::use_broker() is false at the pinned target. External reservations/global reclaim belong here only when a supported sharing path needs them; account at the actual allocation path, not an unused grant hook. Reuse upstream load/park/restore/reload lifetimes and existing ledgers. Avoid a second epoch service, generic fleet framework or power arbiter.

## Files

- docs/reviews/radiance-upstream-2026-11-18/MANIFEST-AND-EXTENSIONS.md
- engines/radiance/components/serving/session.cpp
- engines/radiance/components/moe-remote/engine_rank_coordinator.h

## Validation

After RUM09, start with two model identities in a card-free lifecycle fixture: independent tickets/owner maps, unload/reload and reused rank numbers. Add one two-model hardware sharing case only when that mode is enabled; do not benchmark every fleet permutation.

## Effort & Risk

L, deferred: shared-card reclaim and helper lifetime are substantially broader than a pin update. Keep this optional and unavailable until a concrete multi-model use case justifies implementation.

## Standards

Shared agent workflow; tooling ownership and evidence boundaries. No deep testing during plan authoring.

## Acceptance Criteria

Optional fleet support can be enabled for its documented case without stale helper state, cross-group barriers or overlapping resource claims. Unsupported modes remain refused. This item does not block the initial pin migration.

## Notes

Imported stable roadmap identity from the reviewed design. Prerequisites: RUM09. Review: docs/reviews/radiance-upstream-2026-11-18/README.md; proposal: MANIFEST-AND-EXTENSIONS.md. This initial item is the first-pass execution skeleton; detailed code guidance follows after the plan merge.

Prerequisite RUM09. Priority P3 deliberately excludes this from minimum bump delivery; no fleet abstraction beyond current upstream lifecycle is justified in earlier items.

Astra source review ASTRA-CONSOLIDATION.md: optional after RUM09; engine/helper cards stay disjoint initially. Hot reload, shared resource accounting and cross-engine qualification are deferred capabilities, not initial bump gates.

## Change Log

- 2026-11-18T01:28:59.863720+00:00 (created-by): Created by codex
- 2026-11-18T01:34:06.708874+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:acceptance_criteria, section:notes
- 2026-11-18T01:43:49.306824+00:00 (updated-by): Updated: section:effort_risk
- 2026-11-18T01:57:27.470953+00:00 (updated-by): Updated: section:code_samples
- 2026-11-18T02:20:26.900308+00:00 (updated-by): Updated: section:code_samples, section:notes
- 2026-11-18T02:24:22.080593+00:00 (updated-by): Updated: section:notes
- 2026-11-18T02:27:21.643448+00:00 (updated-by): Updated: section:notes
