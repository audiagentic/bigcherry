# Release documentation

Human-facing release documentation lives here:

- `CURRENT_RELEASE.md` describes the current release state.
- `CURRENT_RELEASE_LEDGER.ndjson` is the tracked release/change ledger.
- `AUDIT_SUMMARY.md`, `CHECKIN.md`, and `CHANGELOG.md` provide maintained
  release guidance and summaries.
- `notes/` contains release notes.

Machine-readable per-pin compatibility records, patch-rebase reports, and
pin-transition state live separately under root `releases/` because they are
consumed and mutated by BigCherry tooling. Do not duplicate those records here.
