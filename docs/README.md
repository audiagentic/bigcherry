# Documentation ownership

New to the project? Start with the agent orientation guide at
[`reference/START_HERE.md`](reference/START_HERE.md).

For exploratory qualification before formal planning, use the
[`reference/experiments/EXPERIMENTAL_WORKFLOW.md`](reference/experiments/EXPERIMENTAL_WORKFLOW.md)
runbook.

Use this ownership model when adding or moving documentation:

- `docs/reference/` contains maintained, cross-cutting guidance only.
- `docs/planning/{active,completed}/<plan>/` contains plan-item design, status,
  decisions, reviews, and work history.
- `docs/evidence/<run-id>/` contains compact, tracked evidence required for
  reproducible validation.
- `artifacts/<run-id>/` contains large, transient, or machine-local campaign
  outputs and raw traces.
- `engines/llamacpp/patches/<patch-id>/` contains patch-specific rationale, validation, fixtures,
  evidence, and support files.
- `docs/archive/` contains historical or superseded prose and review snapshots;
  it is never a live authority and is not part of the maintained reference
  corpus. Historical material has one canonical path here; do not add forwarding
  copies under `docs/reference/`.
- `tools/tests/fixtures/` contains permanent deterministic test inputs.

When relocating a document, update its consumers in the same change. Preserve
historical provenance at the canonical `docs/archive/` path rather than keeping
duplicate redirect files.

Scratch, raw traces, generated corpora, and machine-local outputs are not
documentation: keep them in ignored `artifacts/`, `work/`, or host-local storage.
Deterministic inputs required by tests belong under `tools/tests/fixtures/`.

Completed lab work must be distilled before its lab directory is retired: keep only compact decision-grade evidence under `docs/evidence/` and the owning plan/patch record. Raw run trees, copied upstream source, and migration working packs belong in ignored artifacts/host storage or Git history, not the live tree.

Tracked evidence may include a raw measurement sidecar only when it is required to recompute or independently inspect a retained claim and its evidence README identifies that role. Large traces, transient logs, SQLite databases, and regenerable intermediate outputs stay out of Git. Prefer one canonical copy; duplicate payloads are acceptable only when an evidence bundle must remain independently auditable.
