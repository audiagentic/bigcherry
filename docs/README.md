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
- `patches/<patch-id>/` contains patch-specific rationale, validation, fixtures,
  evidence, and support files.
- `docs/archive/` contains historical or superseded prose and review snapshots;
  it is never a live authority and is not part of the maintained reference
  corpus.
- `tools/tests/fixtures/` contains permanent deterministic test inputs.

When relocating a document, update its consumers in the same change and retain
an explicit provenance pointer where historical references must remain valid.
