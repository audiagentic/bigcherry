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

Retain shared verifier and schema-2 fail-closed behavior; verify canonical content and digest with real C++ for MUL_MAT, MUL_MAT_ID and routed GLU; ensure workflow tune-campaign uses dedicated verifier lane, mandatory strengthened ingest, first-device scoping and transactional winner attestation; run a real dev campaign and record winner_verification rows. Track unsupported ADD and other domains through HI136 quarantine work rather than weakening verification.

## Detailed Solution & Technical Design

The verifier must compare the complete observed C++ canonical JSON and digest, not reimplement serialization in Python. Use dedicated verifier builds so campaign identity is not perturbed, fail on missing/ambiguous provenance, and preserve native fallback. The active completion boundary is operational wiring plus real campaign evidence, not merely offline mocks.

## Code Samples & Guidance



## Files

signature_digest_verification.py; inventory.py; signature_capabilities.py; hi80 evidence CLI; workflow.py; schema-2 tests; real dev campaign artifacts.

## Validation

Poisoned canonical rejection, honest C++ matches for all three classes, offline suite, dedicated verifier lane, mandatory strengthened ingest, and real tune-campaign winner_verification rows. Unsupported domains hard-fail or use separately approved per-row quarantine; no silent unverified success.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Close only when all audited classes match real C++ digests, production workflow is wired and proven by a live campaign with winner_verification rows, and unsupported operations remain fail-closed under an explicit HI136 disposition.

**Transferred THA01 acceptance (independent gate):** Add and verify intrinsic nested transform provenance emission at the C++ hip-autotune-tuner.cu runtime flush: schema/version, exact source revision, hardware architecture, build-descriptor hash, immutable manifest hash and evidence references. Prove runtime writer hashes equal the offline manifest algorithms; add nested fixtures through ingestion/analyse_gap provenance_complete and fail closed on mismatched or missing identity. Compile and capture a real tuning artifact on eligible hardware, ingest it with correct source/build associations, and keep legacy flat adaptation *until* real nested records are accepted. Retire the transitional flat adapter only in an independently evidenced follow-up with old-header rejection; don't silently drop/interpret prior artifacts, alter runtime tuning timing, or introduce another identity registry. THA04's original C++ canonical JSON/digest verification across MUL_MAT, MUL_MAT_ID, routed GLU and workflow winner attestations remains mandatory.

**Transferred THA12 acceptance (independent gate):** Preserve THA12's exact semantics: prove/fix ggml_hip_fill_strides nb[0] byte-count contract and comment, pin sizeof(ggml_hip_dispatch_signature_v1) with static_assert, verify architecture-code/cc injectivity, include or prove all ids type/layout/stride and dispatch/can_execute/executor input fields, test legal GGML quantized-stride normalization injectively or stop normalization, and maintain a source-reader->signature/hardware field->invariant proof table. Include exceptional experimental fusion modes, and require proof table and C++/Python canonical verifier test updates whenever selector, can_execute, executor, metadata or ABI shape changes. Fail closed on omitted/ambiguous input; no semantic widening merely to make old measurements replay.

## Notes

Supersedes: HI125
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi125

CONSOLIDATION 2026-11-22 HIP-AUTOTUNE: receiving THA01,THA12. Existing scope and acceptance are retained. The original historical source plans and evidence are archived as superseded, not completed. No new kernel, orchestrator, scheduler, patch, GPU result, runtime activation or performance qualification is implied. Changes use the existing owning implementation only.

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

## Reviews

- RV4498
- 2026-11-22T10:17:37.071815+00:00 (updated-by): Updated: section:acceptance_criteria, section:notes
