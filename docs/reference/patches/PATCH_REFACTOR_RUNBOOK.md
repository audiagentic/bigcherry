# Patch refactor runbook

This reference implementation/migration document supersedes the older
flat-only RE41 layout decision. The normative, complete runbook is maintained
at `docs/planning/active/patch-system/PATCH_REFACTOR_RUNBOOK.md`.

The current system is package-only for production patches. The registry keeps
legacy flat discovery only for compatibility fixtures; migrated patches bind
source and validation identity immutably, and local CI metadata checks remain
separate from explicit hardware campaigns.
