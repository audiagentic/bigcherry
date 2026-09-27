# BigCherry jobs control plane

Status: **software/control-plane implementation is present; offline jobs CI and real Noble Slurm reference are green; Brutus GPU/production and Windows remote hardware acceptance remain gated.**

Normative design: `docs/design/JOBS_ORCHESTRATOR.md`. Brutus setup: `SLURM_BRUTUS.md`. Cutover/migration: `ACCEPTANCE.md`, `MIGRATION.md`.

## Authority

`python -m bigcherry jobs ...` is the human/agent entry point. Scheduler-neutral `JobService` is the application API; future HTTP/UI adapters must use the same domain DTOs/events/log cursors rather than parse Slurm or CLI presentation output.

BigCherry owns scientific/run identity, durable events, evidence and review. Slurm owns Brutus execution/resources. Rebuildable status/metrics are projections only.

Default roots:

```text
<work>/jobs/       batches/series/runs/attempts/events/status
<work>/hardware/   observed/accepted inventory per executor
```

## Job specification

JSON/TOML BatchSpec example:

```json
{
  "schema": "bigcherry.jobs.v1",
  "patch": "1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2",
  "architectures": ["gfx1201"],
  "model": "/models/model.gguf",
  "planned_sessions": 4,
  "producer": "1265_rd30b_moe_mmq_compact_grid_rdna4_rdna2/producer",
  "hip_path": "/opt/rocm",
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

Plan/validate without queue mutation:

```bash
python -m bigcherry jobs plan batch.json
python -m bigcherry jobs validate batch.json
```

Idempotent submit:

```bash
python -m bigcherry jobs submit batch.json \
  --idempotency-key wave-2026-09-27-01 \
  --actor agent:qualification
```

Same key + same canonical request returns the existing batch; changed content is a hard conflict.

## Operator/agent CLI

```text
jobs plan SPEC
jobs validate SPEC
jobs submit SPEC --idempotency-key KEY [--actor ACTOR]
jobs acceptance SPEC [SPEC ...]
jobs migrate-legacy --command CMD --planned-sessions N [...]
jobs ingest --once

jobs list | queue
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
jobs executors list|show|doctor
jobs metrics
```

Machine commands emit canonical JSON/JSONL except requested Markdown/Prometheus. Durable events reconnect by monotonic sequence; logs reconnect by byte offset.

## Acceptance and legacy migration

`jobs acceptance` emits `bigcherry.jobs.acceptance-matrix.v1` derived from the supplied BatchSpecs plus current accepted inventories. Cases use capabilities, not card slots. Unsupported/missing hardware remains explicit and causes exit 2; it is never silently omitted.

`jobs migrate-legacy` only understands the exact historical `run_campaign.sh` shape plus options with direct JobSpec equivalents. It requires original `BC_MODEL`, `BC_HIP_PATH` and explicit planned-N. Legacy physical device/run-name are metadata only; unknown one-off flags fail closed. See `MIGRATION.md`.

## Hardware discovery/acceptance

Linux AMD:

```bash
python -m bigcherry hardware discover brutus --record
python -m bigcherry hardware diff brutus
python -m bigcherry hardware accept brutus --expected-hash <reviewed-hash>
python -m bigcherry hardware render-gres brutus
```

Discovery consumes AMD-SMI machine JSON plus sysfs. Stable identity preference is UUID -> confirmed serial -> explicit weak hardware epoch. Architecture/model/VRAM/BDF/render/HIP ordinal/driver are recorded; BDF/render/ordinal are locators only. Missing required data fails closed.

Real Brutus acceptance must still confirm the installed AMD-SMI field shapes, identity persistence and peer topology before cutover.

## Scientific identity

`bigcherry.scientific-identity.v2` freezes:

- focal/common/validated-enhancement implementation and validation digests;
- resolved contract IDs/hashes;
- recipes/experiment-contract registry content identity;
- model/corpus/file-backed producer input hashes/sizes;
- producer/baseline identity;
- accepted platform environment hash;
- exact stable physical GPU cohort/hash.

Attempt-start detached workspace re-resolves identity. Drift blocks the attempt rather than mixing definitions inside one series.

Retry always creates a new attempt/native job. `--same-commit` deliberately reuses code; `--latest` may resolve newer code. Monolithic production campaigns do not use Slurm native requeue.

## Durable execution/recovery

Important persisted state:

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

Submission intent is persisted before executor call. Stable `execution_id` recovery:

```text
recorded handle            -> rebind
intent + exactly one match -> rebind
intent + proven zero match -> submit immutable request once
multiple matches           -> block/wake
```

No blind duplicate submission.

## GPU resources

`GpuRequirement` resolves against accepted stable-device inventory. Slurm receives architecture/count, not physical identity. Proper-subset frozen cohorts reserve the complete accepted architecture pool, then narrow only inside the exclusive allocation.

`Allocation.native_gpu_ids` is transient scheduler/launch identity; `stable_gpu_ids` is scientific attestation. Inventory/cohort/topology drift fails closed. Runtime visibility is converted to allocation-local selected positions; selected card is never assumed local index 0.

Direct/local execution uses OS-held locks keyed by stable ID. Timed scientific work is whole-GPU exclusive.

## Evidence harvest/report

`jobs harvest`:

1. finds only evidence records added relative to each attempt's frozen commit;
2. verifies record digest, focal/validation bytes, contract hashes and architecture;
3. enters HI151 maintenance fencing;
4. refuses pre-existing staged changes or dirty canonical evidence destination;
5. merges through the append-only patch evidence API;
6. stages exactly the canonical destination;
7. optionally commits exactly that path;
8. records verified evidence manifest + commit SHA.

No stash/reset/`git add -A`/automatic lifecycle promotion.

`jobs report` reads committed verified records only and reuses the existing session-bootstrap estimator. `review_ready` requires planned-N completion and committed verified evidence; process completion alone is insufficient.

## Observability / future UI

`jobs status` emits scheduler-neutral schema `bigcherry.jobs.status.v1`. Installed observe timer atomically refreshes every 15 seconds:

```text
<jobs>/status/status.json
<jobs>/status/status.md
<jobs>/status/bigcherry.prom
```

Events are durable/checksummed and severity classified. Scientific FAIL is not an operational wake by default. Status/metrics remain projections over JobService/run-store/executor authority.

## Executors

`config/jobs/executors.toml` defines targets.

- `slurm`: Brutus Linux production path.
- `local`: direct Linux/Windows/testing/emergency path.
- `remote`: restricted SSH JSON worker for future non-Slurm/Windows hosts.
- `fake`: deterministic tests.

Remote protocol/correlation is implemented and mocked, but remote scientific campaigns remain disabled until target-local source/model/corpus/artifact staging and Windows HIP stable-ID discovery/re-attestation are hardware accepted.

## Installation

Job-service systemd dry render/doctor:

```bash
python tools/admin/install_bigcherry_jobs.py \
  --project-root /srv/bigcherry \
  --work-root /mnt/data/bigcherry-jobs \
  --python /srv/bigcherry-venv/bin/python
```

Apply after review with `--apply --enable`. Generated units cover durable inbox ingestion + observe projection; there is no custom scheduler daemon.

Brutus Slurm installation is separately repeatable via `tools/admin/install_brutus_slurm.py`; it is dry-run by default, requires accepted inventory and validates MUNGE + `slurmd -G` before starting services on explicit root `--apply`.

## Validation boundary

Offline jobs CI covers scheduler adapters, submission/recovery, durable events, hardware binding/drift/runtime mapping, AMD-SMI fixtures, production pure policy/window state, large file-backed streaming, remote protocol mock, installers/renderers, exact evidence harvest, contract drift, report aggregation, acceptance/migration and status/metrics.

Real Noble Slurm CI covers actual MUNGE/slurmctld/slurmd/sbatch/squeue scheduler behavior.

Still real-host gated:

- Brutus stable identity persistence/peer topology/real GRES and `slurmd -G`;
- allocation -> stable-ID attestation and ROCm device-cgroup matrix;
- live llama-swap snapshot + root exclusive-window admission/recovery;
- loaded-idle/noise qualification;
- one complete real managed series through committed harvest/report;
- Windows HIP discovery + remote staging before Windows scientific enablement.
