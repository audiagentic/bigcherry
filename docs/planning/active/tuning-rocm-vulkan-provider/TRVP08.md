---
id: TRVP08
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:00:44.965045+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# CK dense GEMM provider

## Description

Integrate CK dense GEMM through the provider mechanism with source/build/provider identity and correctness-gated candidates.

## Steps

1. Start only after the hipBLASLt provider mechanism and raw-kernel audit prerequisites are complete. 2. Add CK dense GEMM through the existing BLAS family/provider route without changing native BLAS semantics. 3. Before qualification, pin CK source revision, compiler/toolchain, target architecture, templates/configuration, and all build-affecting CK options into BuildPlan, manifest, and attestation. 4. If CK is compiled, include source/template/config/compiler/options/architecture in the implementation digest; if prebuilt, record an exact provider fingerprint. 5. Expose candidates through the existing tuner and provider-qualified catalog/runtime identity, then gate promotion on correctness and independent evidence. 6. Preserve exact stack/device provenance and fallback behavior.

## Detailed Solution & Technical Design

CK dense GEMM is a provider candidate family behind the existing BLAS contract. Candidate identity is bound to the CK source revision or prebuilt fingerprint, compiler/toolchain, architecture, template/configuration/options, and runtime stack. BuildPlan/manifest/attestation carry this identity before measurement or promotion. Correctness is the baseline for the sibling split-K/pipeline and grouped/MoE work; native remains available and mathematically unchanged on unsupported or failed CK paths.

## Code Samples & Guidance



## Files

tuning/providers/ck.py; CK provider probes and build/attestation metadata; HIP provider route; catalog/runtime candidate identity; correctness, fallback, and qualification tests.

## Validation

Build and inspect CK identity metadata, then run dense GEMM representative shapes/types against the native/reference oracle. Verify candidates are measured through the existing tuner, carry exact provider/build/architecture identity, and reject missing or mismatched fingerprints. Exercise unsupported hardware/configuration fallback and ensure native BLAS semantics remain valid. Do not promote a CK winner before correctness and identity evidence passes.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

No CK candidate is qualified without pinned source/build/provider fingerprint and manifest/attestation coverage. Dense GEMM correctness, fallback, tuner measurement, and provenance gates pass. Native BLAS behavior remains unchanged, and a candidate from another CK source/compiler/architecture/configuration is rejected.

## Notes

Supersedes: RO13
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro13

Supersedes RO13. TRVP09 and TRVP10 are sibling extensions after this dense-provider baseline. Preserve patch 1225 and planning/ledger governance.

## Change Log

- 2026-09-09T11:00:44.965045+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:36.716942+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.565145+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.411873+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:43:56.529048+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034415_repaired-trvp07-09-with-the-co_7761
- 2026-09-10T03:44:15.620153+00:00 (updated-by): Updated: section:ledger-events
