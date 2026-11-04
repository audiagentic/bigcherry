# Platform releases from explicit change entries (not adopted)

Archived from unpushed commit `8a9110f8a` (branch `fix/platform-release-ownership`, 2026-11-04). Inert: not built or tested.

Idea: the `bc-platform` release line is driven only by `platform/changes/` entries (release-state excluded), so shared
tooling, docs, tests or model/engine changes cannot enter the platform line by touching files. It changes
`release-please-config`/manifest keys (`platform/changes` instead of `.`), `release-please.yml`, `pr-title.yml`, the
release docs, and adds `test_platform_release_ownership.py`.

Not adopted: PR #564 fixed the reported 11552.0.0 platform version more simply with `exclude-paths`
(`engines`, `docs/releases/notes`) on the `.` package. Revisit this if the platform line keeps being triggered by
unrelated changes. Reapply with `git apply --3way change.patch`; the release-please manifest has moved on.
