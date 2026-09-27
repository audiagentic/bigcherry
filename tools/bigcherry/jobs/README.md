# BigCherry jobs package

Durable scheduler-neutral campaign control plane.

Primary entry point:

```bash
python -m bigcherry jobs --help
```

Key modules:

```text
model.py       BatchSpec/JobSpec/GPU requirements
identity.py    frozen scientific identity
store.py       durable filesystem/inbox/events
service.py     shared application API
executor.py    scheduler-neutral protocol
slurm.py       Brutus Slurm adapter
local.py       direct LocalExecutor
remote.py      restricted remote transport
worker.py      remote target worker
workspace.py   exact-commit detached workspaces
runner.py      allocation-local campaign launch
production.py  production GPU claim/conflict/contamination policy
harvest.py     verified append-only evidence harvest
gitops.py      explicit fail-closed git transaction helpers
report.py      verified series scientific reports
status.py      status/Markdown/Prometheus projections
acceptance.py  capability-derived cutover matrix
migration.py   fail-closed legacy run_campaign.sh converter
```

Hardware discovery/binding lives in `bigcherry.hardware`.

References:

```text
docs/design/JOBS_ORCHESTRATOR.md
docs/reference/jobs/JOBS_CONTROL_PLANE.md
docs/reference/jobs/SLURM_BRUTUS.md
docs/reference/jobs/ACCEPTANCE.md
docs/reference/jobs/MIGRATION.md
```

Production cutover is still gated on the real Brutus GPU/production acceptance matrix and Windows remote hardware/staging acceptance; offline software completeness is not a substitute for those gates.
