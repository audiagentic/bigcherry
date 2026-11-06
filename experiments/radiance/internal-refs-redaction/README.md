# Repository internal-reference redaction

Archived from `redact-internal-refs` (tip `abfaa4e5f`, historical common ancestor `a40258234`, tip 2026-10-23). This is an archival index only: intentionally no patch, which could contain sensitive historical strings in removed diff lines.

## What and why

Replace machine-specific paths, addresses and names in examples, lab commands, development instructions and configuration with environment-driven paths/settings, and strengthen repository privacy checks. The branch made a large cross-tree edit, not a discrete runtime optimisation.

## Outcome and why stopped

**(a): general sanitisation and environment-key design already included in main's privacy/configuration work (#394 and follow-ups); (b): remaining historical path/name rewrites superseded by today's repository layout and privacy cleanup (#399, #544); (c): no safe, standalone implementation delta identified.** Main contains the current forbidden-terms workflow, example environment configuration and work-root settings. The original `BC_CACHE_ROOT` naming was specifically corrected back to `BC_WORK_ROOT` by #399. Reapplying the historical diff would overwrite newer documentation or revive removed sensitive references. The exact reason this branch was left undeleted is **unknown**; its purpose was satisfied via the other PRs. There is no performance measurement.

## Evidence and resumption

- Experiment commit `abfaa4e5f`; main PRs #394 (`892f74d9e`), #399 (`3d9430a28`), #544 (`ce045574b`).
- `.github/workflows/forbidden-terms.yml`, `config/environment.example.toml`, `docs/reference/ENVIRONMENT.md`, `.githooks/pre-commit` on current main.
- Resume by auditing **current** files and privacy rules. Do not resurrect historical raw diffs, credentials, endpoint strings or private-machine names.
