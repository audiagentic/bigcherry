---
id: THA12
order: 0
plan: tuning-hip-autotune
state: superseded
created-at: '2026-09-09T10:49:27.306659+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P1
---

# Signature identity completeness: ids fields, quantized stride normalisation, and a maintained proof table

## Description

Make dispatch signature identity complete for every factor that can change candidate legality or executor behavior, while preserving the distinction between hardware identity and operation identity.

## Steps

1. Fix ggml_hip_fill_strides comment/code mismatch and define the nb[0] byte-count invariant.
2. Pin sizeof(ggml_hip_dispatch_signature_v1) with static_assert.
3. State and enforce injective architecture-code/cc mapping.
4. Establish ids type/layout/stride invariants or add missing ids fields.
5. Prove quantized stride normalization exact over legal GGML strides, or stop normalizing.
6. Maintain a proof table mapping semantic/selection input, source reader, signature field, hardware field, and derived invariant.
7. Add coverage exceptions for experimental fusion modes and update the table whenever selector/can_execute/executor semantics change.

## Detailed Solution & Technical Design

Capability owner: tuning

Split assessment: One independent boundary; Build/Run support is a dependency.

Overlap assessment: No duplicate boundary found; related items are prerequisites or adjacent evidence.

## Code Samples & Guidance



## Files

HIP signature producer/selector/executor; proof table; signature/fusion unit tests; replay identity fixtures

## Validation

Static signature-size and architecture-map assertions; ids invariant tests; quantized-stride injectivity tests over legal layouts; proof-table review for native selector, candidate legality, executor, fusion, and transform inputs; regression tests for experimental fusion signatures.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

No replayable candidate/executor input is omitted or ambiguously normalized. Signature size cannot drift silently, ids and quantized strides are proven complete, architecture identity remains injective, and the maintained proof table is updated with every semantic change.

## Notes

Supersedes: HI161
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi161

Supersedes: HI161
Inherited semantic scope: preserve ids completeness, stride normalization proof, architecture injectivity, static-size assertion, and maintained proof-table requirements.
Migration: capability-rebaseline-v3-2026-09

CONSOLIDATION 2026-11-22 HIP-AUTOTUNE: Signature completeness, architecture/stride injectivity and C++ canonical digest verification must share one invariant and maintained proof table. Unique acceptance and stop/fallback conditions were transferred to active THA04 through the Planning API. THA04's digest is not correct unless THA12's identity fields and normalization invariants are complete; the two tickets validate the same canonical signature authority. Original code, evidence, source/negative findings and reviewed history remain unchanged in this superseded record. Supersession is about removing duplicate planning ownership, not proof of implementation, quality, hardware safety or a performance win.

## Change Log

- 2026-09-09T10:49:27.306659+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:05:35.700371+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.882720+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.410966+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:24:03.203654+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_022438_the-next-five-high-risk-tuning_6580
- 2026-09-10T02:24:38.712243+00:00 (updated-by): Updated: section:ledger-events

## Reviews

- RV4490
- 2026-11-22T10:19:22.088701+00:00 (updated-by): Updated: section:notes
- 2026-11-22T10:24:29.888132+00:00 (state-transition): State: pending → superseded
