# Release state

This directory is the tracked, machine-readable release-state store consumed by
BigCherry tooling. It is not a documentation archive.

- `<pin>.json` records per-upstream-revision compatibility/validation state.
- `index.json` indexes release records.
- `*-patches.md` are generated patch-composition snapshots for a release.
- `patch-rebase.json` is the current pin-bump compatibility report used by
  the rebase/apply workflow.
- `pin-transition.json`, when present, is the tracked declaration that a pin
  transition is in progress.
- `evidence/` contains small release-specific evidence that release state
  directly references.

Human-facing release notes, audit summaries, check-in guidance, and the release
ledger live under `docs/releases/`. Large build/run outputs belong under
ignored `artifacts/`, not here.

Do not move or delete historical release records merely because their pin is no
longer current: `tools/bigcherry/release/records.py` treats this directory as
the canonical release compatibility history.
