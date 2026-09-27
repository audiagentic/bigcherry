# Brutus Slurm 23.11 install/integration runbook

Normative architecture: `docs/design/JOBS_ORCHESTRATOR.md`. Job-service operator reference: `JOBS_CONTROL_PLANE.md`.

v1 stack:

```text
munge
slurmctld
slurmd
```

No MariaDB, slurmdbd or slurmrestd. BigCherry owns scientific/domain history; Slurm owns execution/resources and keeps `jobcomp/filetxt` as a lightweight native completion trail.

## 1. Preferred repeatable installer

Dry-run/preflight first:

```bash
python tools/admin/install_brutus_slurm.py \
  --project-root /srv/bigcherry \
  --hardware-root /mnt/data/bigcherry-jobs/hardware \
  --executor-id brutus --node-name brutus \
  --cpus <N> --real-memory-mib <MiB>
```

Preflight requires an accepted hardware inventory and renders the exact current templates. It refuses production policy containing monolithic native `RequeueExit` or `ConstrainDevices=yes` before hardware qualification.

After reviewing the plan/config:

```bash
sudo /srv/bigcherry-venv/bin/python tools/admin/install_brutus_slurm.py \
  --project-root /srv/bigcherry \
  --hardware-root /mnt/data/bigcherry-jobs/hardware \
  --executor-id brutus --node-name brutus \
  --cpus <N> --real-memory-mib <MiB> --apply
```

Apply installs only the minimal package stack (`slurm-wlm`, `munge`, `jq`), provisions required directories/MUNGE, writes rendered `/etc/slurm`, validates MUNGE and `slurmd -G`, starts `slurmctld`/`slurmd`, and requires `scontrol ping` success. The action plan is offline-tested.

Manual steps below are the auditable equivalent/fallback.

## 2. Discover and accept hardware first

GPU ordinals are not hardware truth. Discover current Linux AMD state:

```bash
python -m bigcherry hardware discover brutus --record
python -m bigcherry hardware diff brutus
```

Review `observed.json`, especially identity source, architecture/model/VRAM, BDF, render node and HIP ordinal. The Linux provider prefers AMD/HIP UUID, then hardware serial, and fails closed unless an explicit weak hardware epoch is supplied when neither exists.

Accept only the reviewed hash:

```bash
python -m bigcherry hardware accept brutus \
  --expected-hash <observed-material-hash> [--allow-material-change]
python -m bigcherry hardware render-gres brutus
```

`environment.local.toml` supplies policy/toolchain/model paths, never GPU identity/architecture/VRAM overrides.

Before production cutover, capture real AMD-SMI output and prove the chosen stable identity persists across reboot/toolchain views. Linux discovery currently does not infer peer topology; populate/accept peer facts only after real HIP/topology validation.

## 3. Manual package/service baseline

```bash
sudo apt update
sudo apt install -y slurm-wlm munge jq
slurmctld -V
slurmd -V
munge --version
```

Provision MUNGE and require a round-trip before Slurm starts:

```bash
sudo install -d -o munge -g munge -m 0755 /run/munge /var/log/munge
if ! sudo test -s /etc/munge/munge.key; then
  sudo sh -c 'umask 077; dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 status=none'
fi
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 0400 /etc/munge/munge.key
sudo systemctl enable --now munge
munge -n | unmunge >/dev/null
```

Host directories:

```bash
sudo install -d -o slurm -g slurm -m 0755 /var/spool/slurmctld /var/log/slurm
sudo install -d -o root  -g root  -m 0755 /var/spool/slurmd /etc/slurm
sudo touch /var/log/slurm/bigcherry-jobcomp.log
sudo chown slurm:slurm /var/log/slurm/bigcherry-jobcomp.log
```

Capture current CPU/RAM topology (`slurmd -C`, `nproc`, `/proc/meminfo`) and choose conservative `RealMemory`; never copy another host's node line.

## 4. Render exact accepted inventory

Tracked templates:

```text
config/slurm/slurm.conf.example
config/slurm/cgroup.conf.example
```

Render using:

```bash
python tools/admin/render_bigcherry_slurm.py \
  --project-root /srv/bigcherry \
  --hardware-root /mnt/data/bigcherry-jobs/hardware \
  --executor-id brutus --node-name brutus \
  --cpus <N> --real-memory-mib <MiB> --dest-root /tmp/slurm-review
```

Generated GRES is architecture-only, one explicit render node per accepted GPU:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
```

Rules:

- never encode slot/UUID in `Type`;
- stable device ID stays in BigCherry inventory/series identity;
- BDF/render/ordinal are current locators;
- Slurm owns its allocation visibility environment;
- a frozen proper subset of one architecture reserves the complete accepted architecture pool and BigCherry narrows only inside that exclusive allocation.

## 5. cgroup baseline

Production v1 starts with:

```ini
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainDevices=no
ConstrainRAMSpace=no
ConstrainSwapSpace=no
```

`ConstrainDevices=no` is the qualified fallback, not an accidental omission. Enable device fencing only after every supported ROCm/toolchain/cohort cell proves `/dev/kfd`, allocated render nodes, unallocated-node denial, visibility mapping and llama/HIP behavior.

## 6. Scheduler/resource policy

Production template uses:

```ini
SchedulerType=sched/backfill
SchedulerParameters=bf_licenses,bf_interval=2,sched_interval=2
PriorityType=priority/basic
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

Licenses=host_activity:2,build_slot:1
AccountingStorageType=accounting_storage/none
JobCompType=jobcomp/filetxt
JobCompLoc=/var/log/slurm/bigcherry-jobcomp.log

PartitionName=bc-build   ... PriorityTier=10  State=UP
PartitionName=bc-measure ... PriorityTier=100 State=UP
```

Resource classes:

```text
build/prepare: build_slot:1 + host_activity:1
measurement:   host_activity:2 + required architecture GRES
```

**No production `RequeueExit`.** Monolithic campaign retries create `attempt+1` and a new Slurm job, optionally pinned to the same BigCherry commit. Native Slurm requeue was exercised only as a scheduler capability reference; it is not the v1 campaign retry mechanism.

## 7. Validate/start

```bash
sudo slurmd -G
sudo systemctl enable --now slurmctld slurmd
scontrol ping
sinfo -Nel
squeue --json | jq '.jobs'
scontrol show lic
scontrol show config | grep -E 'SchedulerType|SchedulerParameters|PriorityType|AccountingStorage'
```

Noble Slurm 23.11.4 has no `slurmctld -t` config-test option; do not script one. Any `slurmd -G` error blocks managed GPU work.

## 8. BigCherry submission contract

Only `tools/bigcherry/jobs/slurm.py` knows Slurm CLI syntax. A measurement request is structurally:

```text
sbatch --parsable
  --comment bigcherry:<stable-execution-id>
  --partition bc-measure
  --licenses host_activity:2
  --gres gpu:<arch>:<reserved-count>
  --time <bounded>
  --output/--error <attempt-local paths>
  <attempt launch wrapper>
```

No `--account` requirement. BigCherry persists immutable `submission-intent.json` before `sbatch` and `submission.json` after native acceptance. Recovery correlates stable execution ID:

```text
one match -> rebind
proven zero -> submit immutable request once
multiple/ambiguous -> block+wake; never duplicate blindly
```

No v1 dependency on `sacct`.

## 9. AMD allocation/cgroup falsification

For each accepted architecture/card/toolchain and relevant multi-GPU shape prove, with bounded timeouts:

- exact Slurm-visible devices map to accepted stable IDs;
- `/dev/kfd` and allocated render nodes work;
- `hipGetDeviceCount/properties` matches request;
- repeated HIP initialization/enumeration is stable;
- AMD-SMI/rocminfo inspection terminates;
- llama-bench and llama-server smoke succeed;
- peer/tensor split and required producer preflights succeed;
- 4096-context or other declared workload preflights succeed.

If any supported device-cgroup cell fails, retain `ConstrainDevices=no`, GRES scheduling, runtime visibility and BigCherry stable-ID attestation. Record that this is not a device security boundary.

## 10. Production coexistence

Slurm does not own llama-swap. After allocation and before timed measurement, BigCherry derives production potential/running/observed stable-device sets from reviewed config claims plus live runtime/process facts.

```text
target intersects potential OR ownership ambiguous -> exclusive production window
disjoint -> loaded-idle co-residency only after its isolation experiment qualifies
```

Until that isolation gate passes, quiesce production for gating measurements. Config/claim drift or production use of a target GPU during a sample contaminates/discards that sample; it is not a scientific regression result.

No static production-GPU partition exists.

## 11. Hardware drift

Material inventory/GRES change:

```bash
sudo scontrol update NodeName=brutus State=DRAIN Reason='BigCherry inventory drift'
python -m bigcherry hardware discover brutus --record
python -m bigcherry hardware diff brutus
# review + accept exact observed hash
python -m bigcherry hardware accept brutus --expected-hash <sha> --allow-material-change
# rerender /etc/slurm, then:
sudo slurmd -G
sudo systemctl restart slurmd
sudo scontrol reconfigure
python -m bigcherry jobs executors doctor
# rerun required hardware acceptance
sudo scontrol update NodeName=brutus State=RESUME
```

Same-model replacement or relevant topology change starts a new scientific cohort. Locator-only changes require scheduler reconciliation but need not change cohort when stable identity/topology are unchanged.

## 12. Queue-retirement gate

Use `docs/reference/jobs/ACCEPTANCE.md`. Required before old queue retirement:

- jobs-service CI green;
- accepted real AMD identity/GRES and `slurmd -G`;
- ROCm cgroup matrix or explicit fallback;
- production conflict/window/watchdog acceptance;
- inventory drift handling;
- direct-vs-managed evidence parity;
- forced incident/recovery matrix;
- one complete planned series through committed harvest/report with no shell watcher.

Rollback: `jobs pause`, stop managed Slurm execution before returning to direct/manual campaigns, and preserve both Slurm/BigCherry durable state. Never run two schedulers against the same GPUs concurrently.
