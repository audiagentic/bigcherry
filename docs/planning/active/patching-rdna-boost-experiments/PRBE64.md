---
id: PRBE64
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:58.168957+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# VK-ACO-001: Audit redundant RADV/ACO scalar waits

## Description

Audit redundant RADV/ACO scalar waits in Q4_K/Q8_0 GEMV shaders and choose shader rewrite versus Mesa issue based on ISA and runtime evidence.

## Steps

Capture RADV/ACO ISA for Q4_K and Q8_0 GEMV, compare proprietary-driver equivalents and unaffected kernels, count waits/effective bandwidth/TG, test minimal source transformations across compiler versions, and port only a rewrite that reliably changes codegen; otherwise file/track a Mesa issue.

## Detailed Solution & Technical Design

Treat this as research-oracle work. Determine whether waits arise from shader structure or ACO; do not alter source without proving compiler output changes and runtime benefit.

## Code Samples & Guidance



## Files

RADV shader capture/ISA comparison, minimal shader experiments, benchmark evidence, or Mesa issue record.

## Validation

Output parity; ISA wait count, effective bandwidth and TG versus controls.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Port only a minimal rewrite with reliable ACO codegen and runtime improvement; otherwise preserve code and record a reproducible Mesa issue.

## Notes

Supersedes: RD81
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd81

## Change Log

- 2026-09-09T10:57:58.168957+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:07.938367+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.415531+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.240506+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:17:31.880934+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031805_repaired-four-more-graph-and-v_2834
- 2026-09-10T03:18:05.321494+00:00 (updated-by): Updated: section:ledger-events
