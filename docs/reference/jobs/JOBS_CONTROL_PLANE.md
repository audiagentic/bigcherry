# BigCherry jobs control plane

Status: **software/control-plane implementation is present and tested; Brutus GPU/production and Windows remote hardware acceptance remain gated.**

Normative design: `docs/design/JOBS_ORCHESTRATOR.md`. Brutus: `SLURM_BRUTUS.md`. Cutover/migration: `ACCEPTANCE.md`, `MIGRATION.md`.

## Authority

`python -m bigcherry jobs ...` is the human/agent entry point. `JobService` is the scheduler-neutral application API. BigCherry owns scientific/run identity, durable events, evidence/harvest/report and retry legality; Slurm owns Brutus execution/resources. Status/metrics are rebuildable projections.

Default roots:

```text
<work>/jobs/       batches/series/runs/attempts/events/status
<work>/hardware/   observed/accepted inventory per executor
```

## Job specification

JSON/TOML BatchSpec minimally names patch, architecture(s), model, planned sessions and target. Optional producer/common/input/corpus/toolchain/GPU capability fields are typed; arbitrary shell passthrough is not supported.

```bash
python -m bigcherry jobs plan batch.json
python -m bigcherry jobs validate batch.json
python -m bigcherry jobs submit batch.json --idempotency-key wave-01 --actor agent:qualification
```

Same idempotency key + same canonical request returns the existing batch; changed content is a hard conflict.

## Operator/agent surface

```text
jobs plan|validate SPEC
jobs submit SPEC --idempotency-key KEY [--actor ACTOR]
jobs acceptance SPEC [SPEC ...]
jobs migrate-legacy --command CMD --planned-sessions N [...]
jobs ingest --once
jobs list | queue | show RUN | status [RUN]
jobs events [--after N] [--run-id RUN] [--wake-only] [--follow]
jobs logs RUN [--stream stdout|stderr] [--offset N] [--limit N] [--follow]
jobs artifacts RUN
jobs series list | series show SERIES
jobs review|evidence SERIES
jobs harvest SERIES --commit|--stage-only
jobs report SERIES [--format json|markdown]
jobs disable|enable|cancel|hold|release RUN
jobs retry RUN --same-commit|--latest
jobs pause|resume
jobs executors list|show|doctor
jobs metrics
```

Machine output is canonical JSON/JSONL except requested Markdown/Prometheus.

## Hardware discovery

Linux AMD:

```bash
python -m bigcherry hardware discover brutus --record
python -m bigcherry hardware diff brutus
python -m bigcherry hardware accept brutus --expected-hash <reviewed-hash>
python -m bigcherry hardware render-gres brutus
```

Stable identity preference is UUID -> confirmed serial -> explicit weak hardware epoch. Architecture/model/VRAM/BDF/render/launch ordinal/driver are observations; BDF/render/ordinal are locators only. Accepted-inventory/cohort drift fails closed.

## Scientific and attempt identity

Series identity freezes focal/common/validated-enhancement implementation+validation digests, contract/registry identities, model/corpus/file-backed input hashes, producer/baseline identity, platform environment and exact stable physical GPU cohort.

Attempt start resolves one exact BigCherry commit, creates a detached workspace and re-resolves scientific identity. The runner then re-loads accepted hardware and attests the exact frozen stable cohort against actual runtime visibility before spawning the campaign. Launch-local positions are never scientific identity.

Retry always creates a new attempt/native job. Monolithic production does not use Slurm native requeue.

## Monitored execution

The attempt runner executes `validation_campaign` through bounded process monitoring after GPU attestation. It guards the **system root**, work volume and temp filesystem independently, so a detached worktree on a large data volume cannot mask a nearly-full `/` or temp filesystem.

Portable defaults:

```text
BIGCHERRY_ROOT_MIN_FREE_GIB=0
BIGCHERRY_WORK_MIN_FREE_GIB=0
BIGCHERRY_TMP_MIN_FREE_GIB=0
BIGCHERRY_ROOT_MIN_FREE_FRACTION=0.05
BIGCHERRY_WORK_MIN_FREE_FRACTION=0.02
BIGCHERRY_TMP_MIN_FREE_FRACTION=0.05
BIGCHERRY_PROJECT_MIN_FREE_GIB=0
BIGCHERRY_PROJECT_MIN_FREE_FRACTION=0
BIGCHERRY_MONITOR_POLL_SECONDS=15
BIGCHERRY_STALL_SECONDS=0
BIGCHERRY_TERM_GRACE_SECONDS=5
```

Absolute GiB floors are host policy; Brutus should set qualified reserves (candidate 50 GiB root, 200 GiB work, 20 GiB tmp). Percentage defaults remain portable to smaller LocalExecutor hosts.

Disk pressure before or during execution is retryable environment failure (75), not scientific FAIL. The liveness detector treats process-tree CPU ticks or output/artifact byte growth as progress. `BIGCHERRY_STALL_SECONDS=0` disables automatic stall termination until a host-specific threshold is qualified. Termination is process-tree TERM -> bounded grace -> KILL; launch/configuration failures are harness errors.

`experiment.bundle.run_managed()` streams stdout/stderr directly to files, preserving intent-before-spawn and terminal hashes without accumulating large logs in Python memory.

## Durable recovery

Submission intent is durable before the external executor call. Stable `execution_id` recovery is:

```text
recorded handle            -> rebind
intent + exactly one match -> rebind
intent + proven zero match -> submit immutable request once
multiple matches           -> block/wake
```

No blind duplicate submission. Attempts persist start/result/allocation-attestation records and file-backed logs.

## Evidence/report

`jobs harvest` considers only records added relative to the frozen attempt commit, verifies record/focal/validation/contract/architecture identity, takes HI151 maintenance fencing, refuses staged or dirty canonical evidence state, merges through the append-only evidence API and stages/commits exact paths only. No stash/reset/`git add -A`/auto-promotion.

`jobs report` reads committed verified records only and reuses the existing session-bootstrap estimator. `review_ready` requires planned-N completeness plus committed verified evidence.

## Observability

Durable checksummed events reconnect by sequence; logs reconnect by byte offset. `jobs status` exposes scheduler-neutral `bigcherry.jobs.status.v1`. The installed observe timer atomically refreshes `status.json`, `status.md` and Prometheus text every 15 seconds; projections are not authority. Scientific FAIL is not an operational wake by default.

## Executors

- `slurm`: Brutus Linux production path.
- `local`: direct Linux/Windows/testing/emergency path.
- `remote`: restricted SSH JSON worker for future non-Slurm/Windows hosts.
- `fake`: deterministic tests.

Remote scientific work remains disabled until target-local content staging and Windows HIP stable identity are hardware-accepted.

## Installation

```bash
python tools/admin/install_bigcherry_jobs.py \
  --project-root /srv/bigcherry \
  --work-root /mnt/data/bigcherry-jobs \
  --python /srv/bigcherry-venv/bin/python
```

Use `--apply --enable` only after reviewing rendered units. Brutus Slurm has a separate dry-run-first `tools/admin/install_brutus_slurm.py` path requiring accepted inventory and MUNGE/GRES validation.

## Remaining real-host gates

- Brutus stable identity persistence/peer topology/real generated GRES + `slurmd -G`;
- allocation -> stable-ID and ROCm cgroup/peer/tensor-split matrix;
- live llama-swap snapshot + root exclusive-window admission/recovery;
- loaded-idle/noise qualification;
- nonzero stall threshold + absolute disk reserves + representative disk/process-tree/large-log acceptance;
- one complete planned managed series through committed harvest/report;
- Windows HIP discovery + target-local remote staging before Windows scientific enablement.
