# RCD durability/orchestration validation pack

Purpose: falsify `docs/design/JOBS_ORCHESTRATOR.md` assumptions before production implementation. Lab-only; production code belongs under `tools/bigcherry/jobs/` and `tools/bigcherry/hardware/`.

Canonical CI: `.github/workflows/rcd-slurm-validation.yml` on Ubuntu 24.04.

Latest green reference: GitHub Actions run `36219793101`, validation head `ffdc08d00b76f0b8775f50a2cc6e48e1a4d20e01`.

## Offline/domain falsifier

```bash
PYTHONPATH=tools python tools/lab/run-campaign-durability/mock_pipeline.py --self-test
```

Expected:

```json
{"checks": 32, "ok": true}
```

Covers capability-based GPU resolution, deterministic stable-device/cohort binding, all-of-architecture safe reservation, mixed-model ambiguity, peer/exact selection, production claim parsing/conflict policy, platform-environment separation, inventory drift and exit 75/76/77 semantics.

## Real BigCherry process/CLI smoke

```bash
PYTHONPATH=tools python tools/lab/run-campaign-durability/real_bigcherry_process_smoke.py
```

Current result:

```json
{"checks": 25, "ok": true, "scope": "real-bigcherry-modules-cli-and-child-processes"}
```

Uses current production modules and real subprocesses. Covers:

- `experiment.bundle.run_managed()` success;
- child exit 76 preservation;
- launch failure 127;
- stdout/stderr persistence + validation;
- module CLI path;
- real 8 MiB child output;
- HI48 complete/torn-tail recovery;
- real `validation_campaign.main()` parser/producer dispatch with only hardware/build producer body mocked;
- mismatch/missing scientific-input fail-closed behavior.

Known gap: `run_managed()` still buffers child output (`capture_output=True`). Streaming is required before production use with ~1.5 GB server logs.

## Managed-process failure injection

```bash
PYTHONPATH=tools python tools/lab/run-campaign-durability/real_bundle_failure_smoke.py
```

Current result:

```json
{"checks": 12, "ok": true, "scope": "real-managed-process-failure-injection"}
```

Exercises intent-before-child semantics, interrupted/failing managed execution and validation/tamper-related failure behavior against current BigCherry code.

## HI151 + git worktree recovery

```bash
PYTHONPATH=tools python tools/lab/run-campaign-durability/real_recovery_smoke.py
PYTHONPATH=tools python tools/lab/run-campaign-durability/tree_activity_race_smoke.py
```

Current results:

```json
{"checks": 18, "ok": true, "scope": "real-hi151-and-git-worktree"}
{"iterations": 50, "checks": 100, "entered": 50, "blocked": 50, "ok": true, "scope": "real-hi151-concurrent-admission-race"}
```

Covers maintenance/lease mutual exclusion, crashed stale lease cleanup, real detached attempt worktree pinning across branch advances, attempt dirtiness isolation, cleanup, and concurrent maintenance-vs-lease admission races.

## Durable submission/event recovery

```bash
PYTHONPATH=tools python tools/lab/run-campaign-durability/service_recovery_smoke.py
```

Current result:

```json
{"checks": 29, "ok": true, "scope": "durable-submission-and-event-recovery"}
```

Covers durable submission intent, crash between external acceptance and handle persistence, zero/one/multiple native-correlation outcomes, no-blind-duplicate rule, retry identity and durable event sequencing/recovery.

## Real Noble Slurm + BigCherry integration

```bash
bash tools/lab/run-campaign-durability/slurm_noble_v3_smoke.sh
```

CI installs **real Ubuntu Noble Slurm 23.11.4 + MUNGE** and launches real `slurmctld`/`slurmd`. No MariaDB/slurmdbd mocks or service dependencies.

Green reference proves:

```text
MUNGE round-trip
minimal no-db/no-account scheduler
squeue --json
real current-branch BigCherry process harness executed as a Slurm job
hold/release
running build + license-blocked measure + later build -> build1, measure, build2
afterok dependency
RequeueExit=75; same job; SLURM_RESTART_COUNT 0 -> 1
cancel
controller restart retaining queued + running ownership
jobcomp/filetxt terminal history
production cgroup plugins with ConstrainDevices=no
```

Representative successful result:

```json
{
  "ok": true,
  "scope": "real-noble-slurm-v2-no-gpu",
  "slurm_version": "23.11.4",
  "accounting": "none+jobcomp/filetxt",
  "scheduler": "backfill+bf_licenses",
  "schedule_order": "build1 measure build2",
  "restart_counts": "0 1",
  "cgroup_fallback": true
}
```

Failures discovered during CI development were retained as design findings rather than waived: Noble 23.11 has no `slurmctld -t`; exact pending reason strings are unstable diagnostics; slurmdbd/account associations are not required for the accepted v1 stack.

## Still Brutus/hardware-only

The validation pack deliberately does **not** claim:

- actual AMD UUID/RSMI stable-ID availability;
- real generated AMD `gres.conf` accepted by `slurmd -G`;
- `/dev/kfd` + render-node behavior with `ConstrainDevices=yes`;
- HIP/ROCr enumeration under cgroup device filtering for supported toolchains;
- peer access and `-sm tensor` on installed cards;
- llama-swap production claims/process attribution/window behavior;
- real build/disk/ccache pressure behavior;
- production-loaded-idle/active measurement-noise equivalence;
- full GPU `validation_campaign` evidence parity.

Initial Brutus cgroup policy is the already-proven `ConstrainDevices=no` fallback. Device filtering is enabled only after hardware qualification.
