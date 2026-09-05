# Patch validation pointer

This page covers the patch-reference entry point only. The sole canonical
validation authority is
[testing/PATCH_VALIDATION.md](../testing/PATCH_VALIDATION.md), which owns
Experiment Contract obligations, campaign capabilities, evidence provenance,
status semantics, promotion, demotion, and re-promotion.

Do not duplicate thresholds, campaign flags, lifecycle rules, or evidence
policy here. If this pointer conflicts with the testing authority, the testing
authority and the live contract/code are authoritative.

## Choose the right page

| Need | Read |
| --- | --- |
| Create or modify a package-only patch | [PATCH_AUTHORING.md](PATCH_AUTHORING.md) |
| Understand catalog states, composition, dependencies, or apply mechanics | [PATCH_SYSTEM.md](PATCH_SYSTEM.md) |
| Validate, qualify, promote, demote, or re-promote a patch | [../testing/PATCH_VALIDATION.md](../testing/PATCH_VALIDATION.md) |
| Run repository gates or hardware procedures | [../testing/TEST.md](../testing/TEST.md) |
| Refactor an existing patch without losing identity/evidence | [PATCH_REFACTOR_RUNBOOK.md](PATCH_REFACTOR_RUNBOOK.md) |

## Minimal handoff rule

Before claiming that a patch helps, does not help, or remains unknown:

1. Confirm the patch package, source composition, bound contract IDs, target
   hardware, and current upstream pin.
2. Run the hardware-free repository/static gates and an explicitly scoped
   apply dry-run.
3. Follow the canonical testing validation workflow for the applicable
   contract-capable campaign or record the run as diagnostic/blocked.
4. Preserve control/subject identity, correctness, activation, measurements,
   provenance, and the exact evidence-backed lifecycle decision.

Patch-specific README, `validation.toml`, fixtures, and evidence belong under
`patches/<patch-id>/`. This page is navigation, not a second policy.

