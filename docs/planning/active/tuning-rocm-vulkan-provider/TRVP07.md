---
id: TRVP07
order: 0
plan: tuning-rocm-vulkan-provider
state: pending
created-at: '2026-09-09T11:00:40.750531+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: M
priority: null
---

# Raw HIP kernel audit

## Description

Produce an exhaustive raw-HIP kernel audit and disposition artifact without creating unsafe direct-execution candidates.

## Steps

1. Run only after TRVP06 independent validation prerequisites are complete. 2. Inventory raw HIP library kernels and all relevant callsites in managed domains. 3. Classify each kernel/callsite as semantic mapping, helper, unsupported domain, unresolved, fallback/reference, or performance path; never infer ABI or launch behavior from names. 4. Produce a callsite table covering domain, owning implementation, fallback/reference/performance role, provider-stack observability, and exemption reason. 5. Report raw kernel X/mapped/unclassified counts and disposition for every unresolved item. 6. Verify no unmanaged bypass remains in TRVP03-managed domains except documented exemptions; preserve exact evidence lineage.

## Detailed Solution & Technical Design

The audit is inventory and classification only. When capability exists, inspect raw HIP library kernels and callsites through provider probes and source analysis, then map each to a semantic owner or explicit exemption. Raw kernels cannot become selectable candidates merely because a symbol is discoverable. The closure artifact records ownership, fallback/reference/performance role, observability, safety disposition, and unresolved rationale, tied to provider/device/stack evidence.

## Code Samples & Guidance



## Files

tools/provider-probes/hip_kernel_inventory.cpp; tuning/providers/hip_kernel_audit.py; exhaustive callsite/disposition report; provider-stack observability data; audit and safety tests.

## Validation

Qualification report contains complete raw-kernel inventory and callsite table with raw X/mapped/unclassified counts, every unresolved disposition, provider-stack observability, and exemption reason. Inject or discover unsupported/unresolved symbols and verify they remain visible and non-selectable. Confirm no direct unsafe launch path is added and no unmanaged TRVP03-domain bypass remains without a documented exemption.

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Raw kernel inventory is exhaustive for the qualified scope; every kernel/callsite has a semantic mapping, helper/unsupported classification, or explicit unresolved/exemption disposition. No name-based ABI inference or unsafe direct candidate is permitted. Closure report and counts are reproducible, unresolved items are visible, and all managed-domain bypasses are accounted for.

## Notes

Supersedes: RO12
Migration: capability-rebaseline-v3-2026-09
Successor key: tuning-rocm-vulkan-provider-ro12

Supersedes RO12. Depends on TRVP06. Preserve patch 1225 and fail-closed provider/stack identity rules; do not bypass ledger/planning governance.

## Change Log

- 2026-09-09T11:00:40.750531+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:17:32.074448+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.560795+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:47.404841+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:43:49.168286+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes
- chg_20260910_034415_repaired-trvp07-09-with-the-co_7761
- 2026-09-10T03:44:15.598773+00:00 (updated-by): Updated: section:ledger-events
