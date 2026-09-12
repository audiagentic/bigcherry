---
id: PRBE19
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:54:40.683647+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Apply post-fix source-state bake-in across affected PRBE ports

## Description

Maintain the authoritative cross-cutting post-fix source-state and correctness bake-in rule for affected successor ports. PRBE19 is not a standalone implementation patch or prerequisite stage.

## Steps

1. When PRBE13 is materialized, source the shared-expert region from the reviewed post-fix image.
2. When PRBE16 is materialized, apply the corrected calc_nwarps() region.
3. When PRBE18 is materialized, source the corrected SSM fused region and launch geometry.
4. When PRBE20 Wave 2 is materialized, source affected MMVQ/SSM regions from the corrected state.
5. Never apply the raw historical RD25 diff on top of successor ports; verify semantic/content equivalence against the current v3 snapshot.

## Detailed Solution & Technical Design

PRBE19 translates the closed RD25 correctness finding into a reusable source-state constraint. Preserve immutable correctness commit 8cdf1ab081..., reviewed post-fix v2 image 9e46e1fd..., and current v3 snapshot c8af5361... as distinct provenance. The rule applies to PRBE13, PRBE16, PRBE18 and affected PRBE20 regions; it explicitly does not apply to PRBE11/RD12. “Branch tip” must not be used as if the old v2 image were current.

## Code Samples & Guidance



## Files

external source registry; successor patch source regions for PRBE13/16/18/20; semantic/content equivalence checks; provenance and bake-in validation.

## Validation

Dependent-region extraction; semantic/content equivalence against the reviewed post-fix image and current snapshot; pre-fix reachability/reproduction or proof absent; fused decode bit identity/equality; MTP batch-vs-seq controls; native/non-MTP controls; no raw standalone RD25 application.

## Effort & Risk



## Standards

Immutable reviewed post-fix source state; no standalone patch; PRBE-owner-aware sequencing; dependency-aware promotion.

## Acceptance Criteria

Every affected PRBE13/16/18/20 port uses semantically equivalent reviewed post-fix source state and passes its own correctness gate; no broken pre-fix region is materialized; PRBE19 remains a sequencing/source-state rule and RD25 remains provenance only.

## Notes

Supersedes: RD25
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd25

Supersedes: RD25 (closed historical predecessor). RD25 remains historical evidence explaining why PRBE19 exists. It must never again appear as a prerequisite task; PRBE11/RD12 is outside the affected set.

## Change Log

- 2026-09-09T10:54:40.683647+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:11:57.480362+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.214040+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.929217+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:50:36.902436+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria
- chg_20260910_025049_rdna-successors-prbe1719-now_5726
- 2026-09-10T02:50:49.233411+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T09:52:35.243207+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:notes
- chg_20260912_095506_cleaned-the-active-rdna-boost_4906
- 2026-09-12T09:55:06.248056+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T10:09:43.707400+00:00 (updated-by): Updated: section:title, section:validation, section:standards, section:acceptance_criteria
- chg_20260912_101015_fixed-the-remaining-plan-taxon_2574
- 2026-09-12T10:10:15.552804+00:00 (updated-by): Updated: section:ledger-events
