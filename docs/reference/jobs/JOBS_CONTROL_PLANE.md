# BigCherry jobs control plane

Status: implementation in progress; offline control-plane CI is mandatory; Brutus GPU/production acceptance remains hardware-gated.

## Authority

`bigcherry jobs` is the human/agent entry point. The scheduler-neutral `JobService` is the application API; CLI, future HTTP/UI adapters and automation must call the same domain service rather than parse Slurm output directly.

Durable state defaults to `<work>/jobs`; accepted hardware inventories default to `<work>/hardware/<executor>/accepted.json`. Filesystem records are authoritative for BigCherry scientific/run history. Slurm is execution/resource authority on Brutus.

## Batch specification

JSON or TOML is accepted. Example JSON:

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

`plan` resolves accepted hardware and freezes scientific content identities without mutating the queue:

```text
python -m bigcherry jobs plan batch.json
python -m bigcherry jobs validate batch.json
```

Submission is idempotent at the client boundary:

```text
python -m bigcherry jobs submit batch.json \
  --idempotency-key wave-2026-09-27-01 \
  --actor agent:qualification
```

Reusing the same key with identical canonical request returns the existing durable batch without adding receipts/events. Reusing it with different content is an error.

## Management

Current implemented commands:

```text
jobs plan SPEC
jobs validate SPEC
jobs submit SPEC --idempotency-key KEY [--actor ACTOR]
jobs ingest --once
jobs list
jobs show RUN_ID
jobs status [RUN_ID]
jobs events [--after N] [--run-id RUN] [--wake-only] [--follow]
jobs logs RUN_ID [--stream stdout|stderr] [--offset N] [--limit N]
jobs artifacts RUN_ID
jobs review SERIES_ID
jobs disable|enable|cancel|hold|release RUN_ID
jobs retry RUN_ID --same-commit|--latest
jobs pause|resume
jobs executors list
```

Machine output is JSON; `events --follow` is reconnectable JSONL using the durable monotonic event sequence.

`review_ready` is deliberately stricter than process completion. A series whose attempts have all exited successfully is `execution_complete`, but remains not review-ready until RCD08 verified evidence/harvest state is present.

## Deterministic identities

Series creation freezes:

- focal/common/promoted patch implementation/validation identities;
- execution-package validity;
- model and corpus content identity;
- producer inputs when file-backed;
- accepted platform environment hash;
- deterministic exact GPU stable-ID cohort and cohort hash.

At attempt start the pinned runner worktree re-resolves scientific identity and must match the frozen series identity. Drift blocks the attempt; it never silently changes an existing series.

Runs are deterministic session slots under a series. Retry creates a new attempt under the same run. `--same-commit` reuses the prior attempt commit; `--latest` resolves the configured ref for the new attempt.

## GPU resources

Scientific `GpuRequirement` is resolved before submission. Slurm receives architecture/count only.

When a frozen cohort is a proper subset of all GPUs of an architecture, BigCherry requests the complete architecture pool and later narrows within the exclusive allocation. The series never silently substitutes another same-model card.

`Allocation.native_gpu_ids` are launch/scheduler-local only. `Allocation.stable_gpu_ids` is the normalized attestation used for scientific verification. `verify_series_allocation()` fails if the accepted inventory changed, selected stable IDs are absent, topology/cohort changed, or safe over-allocation did not receive the complete architecture pool.

Local/direct execution uses OS-held device locks keyed by stable ID; process death releases locks automatically. Timed scientific measurement is whole-GPU exclusive. GPU sharding is not part of v1 scientific measurement.

## Executors

`config/jobs/executors.toml` defines targets. Exactly one target is default.

- `slurm`: Brutus; Slurm CLI/GRES/licenses exist only in `jobs/slurm.py`.
- `local`: direct Linux/Windows execution and emergency/testing path.
- `remote`: restricted SSH JSON worker protocol, intended for Windows/non-Slurm hosts.
- `fake`: tests only, instantiated directly rather than host config.

Remote execution is currently protocol/mocking scaffolding, not yet production-ready for arbitrary campaign payloads. The remote host must own target-local workspaces/artifacts; controller-local absolute paths cannot be treated as portable. Do not enable a remote target for scientific campaigns until the RCD11 remote staging/worker acceptance tests are complete.

## Install/service integration

Render or install systemd ingestion:

```text
python tools/admin/install_bigcherry_jobs.py \
  --project-root /srv/bigcherry \
  --work-root /mnt/data/bigcherry-jobs \
  --python /srv/bigcherry-venv/bin/python
```

Without `--apply` this is a dry render/doctor. Sandbox install tests use `--dest-root` and never require root/systemd.

Production install adds:

```text
/etc/bigcherry/jobs.env
/etc/systemd/system/bigcherry-jobs-ingest.service
/etc/systemd/system/bigcherry-jobs-ingest.path
/etc/systemd/system/bigcherry-jobs-ingest.timer
```

The path unit reacts to durable pending receipts; the timer is only a missed-notification safety net. There is no custom scheduler polling daemon.

Brutus Slurm installation/qualification is separately defined by `SLURM_BRUTUS.md`. v1 is MUNGE + slurmctld + slurmd, no slurmdbd/MariaDB requirement.

## Recovery model

Submission writes `submission-intent.json` before calling the executor. Stable `execution_id` is carried into scheduler/worker correlation metadata. After acceptance, `submission.json` records the native handle.

On service restart:

- receipts already under `processing/` are recovered before new pending work;
- an attempt with a recorded submission is rebound;
- an attempt with intent but no handle calls executor correlation;
- one match is rebound;
- multiple matches are recovery ambiguity and no duplicate is submitted;
- a proven zero-match can submit the same immutable request.

## Offline validation

`.github/workflows/jobs-service-validation.yml` compiles the production modules and runs `tools/tests/jobs` on Ubuntu 24.04. Tests cover scheduler rendering/parsing, idempotency, durable inbox recovery, hardware cohort binding, torn event-tail recovery, retry identity, CLI/service models, fake remote protocol, installer rendering and allocation-local device mapping.

RCD Slurm CI remains a separate real-Slurm gate. Brutus-only gates remain AMD stable identity, generated GRES `slurmd -G`, ROCm/cgroup mapping, peer/tensor split, llama-swap coexistence and real scientific evidence parity.
