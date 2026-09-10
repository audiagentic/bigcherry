---
id: PRBE61
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:45.405833+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# VK-TUNE-001: Vulkan rm_kq specialization sweep

## Description

Sweep Vulkan rm_kq specialization per captured FA/matvec signature, driver, and RDNA architecture.

## Steps

Recheck discussion #21043; benchmark rm_kq 1/2/3/4 by signature on R9700 across RADV and AMD proprietary/AMDVLK plus gfx1100 controls; inspect VGPR/registers and kernel timing, PP/TG; retain only repeatable per-driver/architecture/signature winners.

## Detailed Solution & Technical Design

Treat rm_kq as a tuning dimension, not a universal constant. Keep selection keyed to driver, architecture, kernel signature, and fallback when evidence is absent.

## Code Samples & Guidance



## Files

Vulkan kernel tuning parameter and candidate tables; per-driver signature tests; ISA/resource and benchmark evidence.

## Validation

Backend parity; VGPR/registers, kernel time, PP/TG versus controls.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Promote only a per-driver/architecture/signature rm_kq choice with repeatable runtime gain and no resource/correctness regression; no global constant.

## Notes

Supersedes: RD78
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd78

## Change Log

- 2026-09-09T10:57:45.405833+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:55.801381+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.403269+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.219232+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:16:18.025837+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031644_repaired-four-more-active-succ_8062
- 2026-09-10T03:16:44.961564+00:00 (updated-by): Updated: section:ledger-events
