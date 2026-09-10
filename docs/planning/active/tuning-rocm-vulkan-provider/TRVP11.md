---
id: TRVP11
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:00:56.194096+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# ROCm qualification matrix

## Description

Qualify ROCm/provider stacks through capability-driven matrices across exact runtime, compiler, provider-library, CK, driver, and GPU tuples; never branch on version strings.

## Steps

1. Define qualification tuples covering ROCm 7.2, 7.14, and 10 plus exact runtime/compiler/provider-library/CK revision/kernel-driver/GPU architecture combinations. 2. Capture artifact hashes, stack/provider fingerprints, capability snapshots, and expected supported/blocked results for each tuple. 3. Select validation strength and candidate set from observed capabilities, not version branches. 4. Run provider completeness, correctness, fallback, replay, and promotion checks for TRVP08-10 outputs across each tuple. 5. Record unsupported-feature cases with weaker supported validation where policy permits, and reject cross-stack evidence transfer without explicit policy. 6. Publish reproducible qualification matrix and release evidence.

## Detailed Solution & Technical Design

The matrix is tuple-based rather than a coarse “ROCm version qualified” label. Each row binds runtime/compiler/provider library/CK revision/kernel driver/GPU architecture, build artifact hashes, capability snapshot, candidate inventory, and expected supported/blocked behavior. Feature probes determine routes and validation depth; implementation does not contain if-version branches. Evidence and replay are accepted only with matching tuple identity.

## Code Samples & Guidance



## Files

qualification recipes/tools; provider capability matrix and tuple schema; BuildPlan/manifest/attestation fingerprints; release evidence reports; cross-stack compatibility/fallback/replay tests.

## Validation

Execute representative qualification rows on ROCm 7.2, 7.14, and 10 where available. Confirm each row records full tuple identity, artifact hashes, capabilities, candidate completeness, correctness, fallback, and replay behavior. Exercise missing/unsupported features and verify capability-driven weaker validation or blocked status. Attempt cross-stack replay and confirm rejection absent an explicit policy. Search implementation for forbidden version-specific route branches.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Exact stack/provider tuples and artifact hashes are recorded; capability probes, not version strings, select behavior. Every tuple has supported/blocked expectations and evidence for completeness, correctness, fallback, and replay. Cross-stack evidence is non-transferable without policy, and no if-version implementation branching remains.

## Notes

Supersedes: RO16
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro16

Supersedes RO16. TRVP09 and TRVP10 are sibling inputs after TRVP08; qualification must not serialize them incorrectly. Preserve patch 1225 and ledger/planning governance.

## Change Log

- 2026-09-09T11:00:56.194096+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:49.551989+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.579230+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.433866+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T01:03:02.193130+00:00 (updated-by): Updated: section:validation
- chg_20260910_010342_successor-plans-now-have-expli_8662
- 2026-09-10T01:03:42.570389+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:45:18.523335+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034539_repaired-trvp10-12-with-the-co_1657
- 2026-09-10T03:45:39.453859+00:00 (updated-by): Updated: section:ledger-events
