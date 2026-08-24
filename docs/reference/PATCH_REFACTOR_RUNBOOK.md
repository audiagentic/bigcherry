# Patch refactor runbook

This reference implementation/migration document supersedes the older
flat-only RE41 layout decision. The normative, complete runbook is maintained
at `docs/planning/active/patch-system/PATCH_REFACTOR_RUNBOOK.md`.

The current system preserves flat patches, supports packaged patches, binds
source and validation identity immutably, and separates local CI metadata
checks from explicit hardware campaigns.
