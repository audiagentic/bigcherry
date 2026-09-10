---
id: PRBE57
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:26.943294+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# FORK-MTP-001: Independent NextN/MTP tensor placement

## Description

Qualify independent NextN/MTP tensor placement by topology transfer cost, with coupled analysis against PRBE58 before deciding patch shape.

## Steps

Register/recheck the MrLordCat source and diff commits 41a8ca78 and 1fcc05da together; map overlap before choosing one patch, ordered pair, or supersession; test NextN owner on first/last/main target GPU and device order on dual XTX/R9700; compare single GPU and MTP-off controls; measure copies/token, TG and PP cost with temp-0 identity/acceptance.

## Detailed Solution & Technical Design

Place NextN tensors according to speculative handoff cost rather than output placement. Treat RD71 and RD72 as a coupled investigation, not predetermined coupling; preserve topology-aware fallback and explicit device-order semantics.

## Code Samples & Guidance



## Files

Vulkan/HIP multi-GPU NextN placement; topology/copy-cost selector; temp-0/acceptance tests; coupled RD71/RD72 comparison and dual-GPU evidence.

## Validation

Correctness: temp-0 identity and MTP acceptance. Performance: copies/token, TG, PP cost for owner alternatives, device orders, single-GPU and MTP-off controls.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Select placement only after coupled RD71/RD72 overlap analysis and repeatable lowest total transfer cost with no acceptance regression; otherwise preserve existing placement.

## Notes

Supersedes: RD71
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd71

## Change Log

- 2026-09-09T10:57:26.943294+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:40.158372+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.384830+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.190479+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:13:19.105683+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031346_repaired-four-more-migrated-pa_4345
- 2026-09-10T03:13:46.500792+00:00 (updated-by): Updated: section:ledger-events
