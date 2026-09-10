---
id: TRVP01
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:00:15.865195+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# Manifest measurement database binding

## Description

Bind provider-qualified manifest measurements to canonical build/source/provider identities with fail-closed completeness.

## Steps

Define measurement DB binding using source/build/stack/provider identity; reject ambiguous/missing bindings; preserve exact artifact and manifest hashes; expose deterministic read/write and migration tests; integrate with campaign receipts.

## Detailed Solution & Technical Design

Provider qualification must not attach measurements to a merely similar build. Bind by immutable source slice, build descriptor, resolved stack and artifact identity; retain provenance for replay.

## Code Samples & Guidance



## Files

Provider measurement DB/schema/binding, manifest and receipt adapters, tests.

## Validation

Exact binding, missing/ambiguous rejection, migration/round-trip, provider stack mismatch and artifact hash tests.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Every measurement resolves to one immutable build/source/provider identity or is rejected; no heuristic binding.

## Notes

Supersedes: RO06
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro06

Supersedes: RO06
Inherited constraints: RV116 and RV130 — preserve run-ID uniqueness, next-unused migration allocation, expected/runtime separation, lineage rejection, and replay mismatch gates.
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-09T11:00:15.865195+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:05.010382+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.533033+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.420426+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T00:51:49.108839+00:00 (updated-by): Updated: section:description, section:steps, section:acceptance_criteria, section:notes
- chg_20260910_005948_legacy-planning-folders-now-co_1240
- 2026-09-10T00:59:48.923500+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T01:02:31.164200+00:00 (updated-by): Updated: section:validation
- chg_20260910_010342_successor-plans-now-have-expli_8662
- 2026-09-10T01:03:42.488874+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:39:51.923406+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_034017_repaired-six-vulkanprovider-s_6655
- 2026-09-10T03:40:17.999917+00:00 (updated-by): Updated: section:ledger-events
