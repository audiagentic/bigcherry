# BigCherry jobs control plane

Status: **operational implementation present and offline/real-Slurm validated; Brutus GPU/production acceptance remains hardware-gated.**

Normative architecture: `docs/design/JOBS_ORCHESTRATOR.md`. Brutus setup: `docs/reference/jobs/SLURM_BRUTUS.md`.

## Authority

`python -m bigcherry jobs ...` is the human/agent interface. The scheduler-neutral `JobService` is the application API; future HTTP/UI adapters must call the same service/schema rather than parse Slurm or CLI presentation output.

BigCherry owns durable scientific/run identity, events, evidence and review state. Slurm owns Brutus execution/resource scheduling. Rebuildable `status.json`/metrics are projections, never authority.

Default state roots:

```text
<work>/jobs/       durable batches/series/runs/attempts/events/status
<work>/hardware/   observed/accepted inventory by executor
```

## Job specification

JSON or TOML is accepted. Example:

```json
{
  "schema": "bigcherry.jobs.v1",
  "patch": "1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2",
  "architectures": ["gfx1201"],
  "model": "/models/model.gguf",
  "planned_sessions": 4,
  "producer": "1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2/producer",
  "baseline_source": "bigcherry-tuning",
  "gpu_count": 1,
  "code_ref": "patch-refactor",
  "target": {
    "executor_id": "brutus",
    "host_id": "brutus",
    "platform_family": "linux-rocm"
  }
}
```

Dry planning validates target inventory/capability and freezes scientific content without queue mutation:

```bash
python -m bigcherry jobs plan batch.json
python -m bigcherry jobs validate batch.json
```

Submission is client-idempotent:

```bash
python -m bigcherry jobs submit batch.json \
  --idempotency-key wave-2026-09-27-01 \
  --actor agent:qualification
```

Same key + same canonical request returns the existing batch. Same key + different request is a hard conflict.

## Complete operator/agent CLI

```text
jobs plan SPEC
jobs validate SPEC
jobs submit SPEC --idempotency-key KEY [--actor ACTOR]
jobs ingest --once

jobs list
jobs queue
jobs show RUN_ID
jobs status [RUN_ID]

jobs events [--after N] [--run-id RUN] [--wake-only] [--follow]
jobs logs RUN_ID [--stream stdout|stderr] [--offset N] [--limit N] [--follow]
jobs artifacts RUN_ID

jobs series list
jobs series show SERIES_ID
jobs review SERIES_ID
jobs evidence SERIES_ID
jobs harvest SERIES_ID --commit|--stage-only
jobs report SERIES_ID [--format json|markdown]

jobs disable|enable|cancel|hold|release RUN_ID
jobs retry RUN_ID --same-commit|--latest
jobs pause|resume

jobs executors list
jobs executors show EXECUTOR_ID
jobs executors doctor
jobs metrics
```

Machine-oriented commands emit canonical JSON/JSONL except explicitly requested Markdown or Prometheus text. `events --follow` resumes from durable monotonic sequence; `logs --follow` resumes by byte offset.

## Deterministic scientific identity

Series creation (`bigcherry.scientific-identity.v2`) freezes:

- focal/common/validated-enhancement patch implementation and validation digests;
- resolved experiment-contract IDs + contract hashes;
- `config/recipes.toml` and `config/experiment-contracts.toml` content identities;
- model SHA256/size;
- corpus and file-backed producer-input identities;
- producer/baseline identity;
- accepted platform environment hash;
- deterministic exact stable-GPU cohort/hash.

The attempt-start detached worktree re-resolves identity. Any mismatch blocks the attempt and requires a new series rather than silently mixing sessions across scientific definitions.

Retry creates a new attempt under the same run. `--same-commit` deliberately reuses the prior commit; `--latest` resolves the configured ref for the new attempt. Monolithic campaigns do not use Slurm native requeue.

## Durable execution/recovery

Filesystem state includes:

```text
jobs/
  requests/
  batches/<batch>/batch.json
  series/<series>/series.json
  runs/<run>/intent.json
  runs/<run>/control.json
  runs/<run>/attempts/NNN/
    attempt.json
    submission-intent.json
    submission.json
    executor-start.json
    executor-result.json
    stdout.log / stderr.log
  inbox/{pending,processing,accepted,rejected}/
  events.jsonl
  status/{status.json,status.md,bigcherry.prom}
```

Submission persists `submission-intent.json` before the executor call. Stable `execution_id` is carried into scheduler/worker correlation metadata. Recovery behavior:

```text
recorded submission        -> rebind
intent + exactly one match -> rebind
intent + proven zero match -> submit immutable request once
multiple matches           -> block/wake; never duplicate blindly
```

Receipts already in `processing/` are recovered before new pending work.

## GPU resources

Scientific `GpuRequirement` resolves against accepted stable-device inventory. Slurm sees architecture/count, not scientific physical IDs.

If a frozen cohort is a proper subset of all GPUs of one architecture, BigCherry reserves the complete accepted architecture pool and narrows only inside that already-exclusive allocation. Runtime visibility order is attested and converted to allocation-local positions; the runner never assumes the selected card is local index 0.

`Allocation.native_gpu_ids` are transient scheduler/launch identifiers. `Allocation.stable_gpu_ids` is scientific attestation. Accepted-inventory/cohort/topology drift fails closed.

Direct/local execution uses OS-held locks keyed by stable ID. Timed scientific work is whole-GPU exclusive; GPU sharding is not a v1 gating mode.

## Evidence harvest and reports

Successful attempts leave patch evidence in detached worktrees. `jobs harvest`:

1. identifies only records added relative to the attempt's frozen commit;
2. verifies record digests, patch bytes, validation bytes, contract hashes and architecture;
3. acquires HI151 maintenance fencing;
4. refuses pre-existing staged changes or dirty canonical evidence destination;
5. merges using existing append-only `patch.evidence.write_record()`;
6. stages only the exact canonical evidence path;
7. optionally commits that exact path;
8. records a verified evidence manifest + commit SHA.

No stash, reset, `git add -A`, arbitrary source copy or lifecycle promotion occurs.

`jobs report` reads only records named by committed verified evidence and uses the existing session-bootstrap estimator. Contract verdicts remain the policy authority; reports do not introduce a materiality threshold.

`review_ready` is stricter than process completion: all planned sessions must complete and committed verified evidence must exist.

## Observability / future UI API

`jobs status` emits schema `bigcherry.jobs.status.v1` from the same `JobService` state used by the CLI. A future API/UI can expose this directly plus event/log cursors without changing the domain model.

The installed observe timer atomically refreshes every 15 seconds:

```text
<jobs>/status/status.json
<jobs>/status/status.md
<jobs>/status/bigcherry.prom
```

Prometheus output currently covers pause state, event sequence, durable inbox counts, runs by normalized state and review-ready series. Provider-specific host/GPU metrics can extend this projection later.

## Executors

`config/jobs/executors.toml` defines targets; exactly one may be default.

- `slurm`: Brutus production Linux path.
- `local`: direct Linux/Windows/testing/emergency path.
- `remote`: restricted SSH JSON worker transport for future Windows/non-Slurm hosts.
- `fake`: deterministic tests.

The remote protocol/correlation semantics are mocked and implemented, but **remote scientific campaigns remain disabled until target-local source/model/artifact staging and Windows HIP stable-device discovery pass RCD11**. Controller absolute paths are not portable and are never treated as such.

## Install

Render/doctor without mutation:

```bash
python tools/admin/install_bigcherry_jobs.py \
  --project-root /srv/bigcherry \
  --work-root /mnt/data/bigcherry-jobs \
  --python /srv/bigcherry-venv/bin/python
```

Apply/enable after review:

```bash
sudo /srv/bigcherry-venv/bin/python tools/admin/install_bigcherry_jobs.py \
  --project-root /srv/bigcherry \
  --work-root /mnt/data/bigcherry-jobs \
  --python /srv/bigcherry-venv/bin/python \
  --apply --enable
```

Generated units:

```text
bigcherry-jobs-ingest.service
bigcherry-jobs-ingest.path
bigcherry-jobs-ingest.timer
bigcherry-jobs-observe.service
bigcherry-jobs-observe.timer
```

The path unit reacts to durable pending receipts; the ingest timer is only notification-loss protection. There is no custom scheduler daemon.

## Validation status

Jobs-service CI on Ubuntu 24.04 covers the production modules, including:

- submission idempotency + crash recovery;
- Slurm adapter rendering/parsing;
- hardware cohort binding/drift/runtime visibility mapping;
- production coexistence/window pure state machine;
- 16 MiB stdout + 3 MiB stderr file-backed managed streaming;
- remote protocol mock;
- systemd/Slurm config rendering;
- exact/idempotent temp-git evidence harvest;
- dirty/staged path refusal and contract-drift rejection;
- existing session-bootstrap report aggregation;
- status/Markdown/Prometheus projection.

Separate real Noble Slurm CI has run actual MUNGE/slurmctld/slurmd/sbatch/squeue and a BigCherry process inside a Slurm job.

Brutus-only gates remain real AMD stable identity/GRES, ROCm/cgroup device behavior, peer/tensor-split preflight, llama-swap production isolation, scheduler-noise qualification and one complete real planned series through submit -> execute -> harvest -> report.
