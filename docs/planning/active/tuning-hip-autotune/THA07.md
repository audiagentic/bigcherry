---
id: THA07
order: 0
plan: tuning-hip-autotune
state: pending
created-at: '2026-09-09T10:48:45.986227+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: null
---

# RCCL Phase 2: topology-aware qualification/tuning for the device-3-excluded subset

## Description

Qualify RCCL for Brutus multi-GPU subsets excluding device 3 using topology and operation identities, correctness-first staged search, and fail-closed runtime selection.

## Steps

Define ReductionOperationSignatureV1 and portable TopologyIdentityV1 for {0,2},{1,2},{0,1,2}; extend rccl_qualify fixtures; staged protocol/algorithm/channel/chunk search; crash-isolated correctness against HI18; measure production reductions; promote exact winners; unqualified topology never falls through to stock RCCL; decide plugin integration only after ABI/value evidence.

## Detailed Solution & Technical Design

RCCL Phase 2 applies only cohorts that can form one comm context. Device 3 remains META-only; mixed-provider subgroup reduction is architecturally invalid without hierarchical composition. Keep topology identity free of ordinals/BDF/UUID/hostname.

## Code Samples & Guidance



## Files

hip-collectives/RCCL qualification, topology/candidate schemas, rccl_qualify.py, fixtures, runtime safety and campaign evidence.

## Validation

Fresh-process correctness then performance for all three eligible subsets, no RCCL on unqualified topology, exact topology/candidate provenance.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only exact verified RCCL winners for eligible topologies; never claim mixed RCCL/META correctness without a real hierarchical design.

## Notes

Supersedes: HI142
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-hip-autotune-hi142

## Change Log

- 2026-09-09T10:48:45.986227+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:04:50.281370+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:00.833813+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.334555+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:27:13.261402+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032737_repaired-four-major-tuning-suc_7897
- 2026-09-10T03:27:37.914893+00:00 (updated-by): Updated: section:ledger-events
