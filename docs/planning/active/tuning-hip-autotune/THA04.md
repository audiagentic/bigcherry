---
id: THA04
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:48:21.951042+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# HI121 close-out step 4: C++-authoritative canonical signature digest verification (RV84 P0-4)

## Description

Complete C++-authoritative canonical signature digest verification and live workflow validation for all audited dispatch classes.

## Steps

Retain shared verifier and schema-2 fail-closed behavior; verify canonical content and digest with real C++ for MUL_MAT, MUL_MAT_ID and routed GLU; ensure workflow tune-campaign uses dedicated verifier lane, mandatory strengthened ingest, first-device scoping and transactional winner attestation; run a real Brutus campaign and record winner_verification rows. Track unsupported ADD and other domains through HI136 quarantine work rather than weakening verification.

## Detailed Solution & Technical Design

The verifier must compare the complete observed C++ canonical JSON and digest, not reimplement serialization in Python. Use dedicated verifier builds so campaign identity is not perturbed, fail on missing/ambiguous provenance, and preserve native fallback. The active completion boundary is operational wiring plus real campaign evidence, not merely offline mocks.

## Code Samples & Guidance



## Files

signature_digest_verification.py; inventory.py; signature_capabilities.py; hi80 evidence CLI; workflow.py; schema-2 tests; real Brutus campaign artifacts.

## Validation

Poisoned canonical rejection, honest C++ matches for all three classes, offline suite, dedicated verifier lane, mandatory strengthened ingest, and real tune-campaign winner_verification rows. Unsupported domains hard-fail or use separately approved per-row quarantine; no silent unverified success.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close only when all audited classes match real C++ digests, production workflow is wired and proven by a live campaign with winner_verification rows, and unsupported operations remain fail-closed under an explicit HI136 disposition.

## Notes

Supersedes: HI125
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi125

## Change Log

- 2026-09-09T10:48:21.951042+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:24.950448+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.804519+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.286272+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:25:50.330741+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032600_repaired-two-more-tuning-succe_5113
- 2026-09-10T03:26:00.715946+00:00 (updated-by): Updated: section:ledger-events
