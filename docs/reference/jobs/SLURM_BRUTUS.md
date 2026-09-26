# Brutus Slurm 23.11 install/integration runbook

Normative architecture: `docs/design/JOBS_ORCHESTRATOR.md`. This is the concrete RCD03 Brutus procedure.

Validated v1 service stack:

```text
munge
slurmctld
slurmd
```

No MariaDB, slurmdbd or slurmrestd in v1. BigCherry owns scientific/domain history; Slurm owns execution/resource scheduling and keeps `jobcomp/filetxt` as a lightweight native completion trail.

Latest real CPU/service reference: GitHub Actions run `36219793101`, Ubuntu 24.04.5, `slurm-wlm 23.11.4-1.2ubuntu5`.

## 1. Install packages

```bash
sudo apt update
sudo apt install -y slurm-wlm munge jq

slurmctld -V
slurmd -V
munge --version
getent passwd slurm
getent passwd bigcherry
```

Expected Slurm family: 23.11.x. Record exact package versions in acceptance evidence.

## 2. MUNGE

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

MUNGE round-trip is a hard gate before Slurm services start.

## 3. Host directories

```bash
sudo install -d -o slurm -g slurm -m 0755 /var/spool/slurmctld
sudo install -d -o root  -g root  -m 0755 /var/spool/slurmd
sudo install -d -o slurm -g slurm -m 0755 /var/log/slurm
sudo install -d -o root  -g root  -m 0755 /etc/slurm
sudo touch /var/log/slurm/bigcherry-jobcomp.log
sudo chown slurm:slurm /var/log/slurm/bigcherry-jobcomp.log
```

BigCherry work/build/tmp/log roots remain under `/mnt/data`.

## 4. Discover and accept hardware before rendering Slurm GPU config

RCD12 discovered state is authoritative; do not hand-maintain GPU ordinals.

Required per GPU:

```text
stable device_id + identity_source
arch
model
VRAM
PCI BDF
render node
NUMA / peer topology
AMD driver
```

Target commands once RCD12 production code exists:

```bash
bigcherry hardware discover --json > /mnt/data/bigcherry-jobs/hardware/observed.json
bigcherry hardware diff --accepted --json
bigcherry hardware accept --inventory-hash <sha256>
```

`environment.local.toml` supplies host policy/toolchain/model paths, not GPU inventory truth.

Capture CPU topology:

```bash
slurmd -C
nproc
awk '/MemTotal:/ {print $2}' /proc/meminfo
```

Use conservative `RealMemory`; never copy another host's rendered line.

## 5. Render `/etc/slurm`

Tracked policy templates:

```text
config/slurm/slurm.conf.example
config/slurm/cgroup.conf.example
config/slurm/gres.conf.example
```

Render placeholders in `slurm.conf.example`:

```text
@NODE_NAME@        current Brutus hostname
@CPUS@             configured CPUs
@REAL_MEMORY_MIB@  conservative host RAM MiB
@NODE_GRES@        accepted architecture counts
```

Example node GRES summary:

```text
gpu:gfx1100:2,gpu:gfx1201:1,gpu:gfx1030:1
```

Generated `/etc/slurm/gres.conf` contains one explicit render-node record per accepted device:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1100 File=/dev/dri/renderD129 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
Name=gpu Type=gfx1030 File=/dev/dri/renderD131 Flags=amd_gpu_env
```

Rules:

- `Type` is architecture only; never `gfx1100_0`.
- stable UUID/device ID stays in BigCherry inventory/series identity, not GRES type.
- PCI BDF/render node/ordinal are current locators only.
- `Flags=amd_gpu_env` owns Slurm `ROCR_VISIBLE_DEVICES`; BigCherry must not replace it with host-global ordinals.
- series-bound proper subset of one architecture reserves all accepted GPUs of that architecture, verifies the bound stable IDs are present, then narrows inside the already-exclusive allocation.

## 6. cgroup baseline

Install `/etc/slurm/cgroup.conf` from the tracked template:

```ini
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainDevices=no
ConstrainRAMSpace=no
ConstrainSwapSpace=no
```

`ConstrainDevices=no` is the **qualified initial fallback**, not a temporary typo. Real Noble CI proved the cgroup process/task/accounting stack in this mode.

Only switch to `ConstrainDevices=yes` after Brutus hardware qualification in §11 passes every supported ROCm/toolchain/cohort cell.

## 7. Validate generated configuration before service cutover

Noble 23.11.4 has **no** `slurmctld -t` config-test option; do not add one to automation.

GRES validation:

```bash
sudo slurmd -G
```

Any GRES error blocks node resume/start of managed GPU work.

Controller config is validated by real daemon startup plus interrogation:

```bash
sudo slurmctld -Dvv   # foreground diagnostic when qualifying changes
# or normal systemd start after initial qualification
```

Then use `scontrol show config`; do not depend on exact human formatting beyond required settings.

## 8. Start services

```bash
sudo systemctl enable slurmctld slurmd
sudo systemctl restart slurmctld
sudo systemctl restart slurmd

scontrol ping
sinfo -Nel
squeue --json | jq '.jobs'
scontrol show lic
scontrol show config | grep -E 'SchedulerType|SchedulerParameters|PriorityType|RequeueExit|AccountingStorage'
```

Required v1 services:

```text
munge
slurmctld
slurmd
```

No `sacctmgr`, account bootstrap or `--account` argument is required.

## 9. Exact scheduler/resource policy

Core policy from `config/slurm/slurm.conf.example`:

```ini
AuthType=auth/munge
AuthInfo=cred_expire=30
CredType=cred/munge

SchedulerType=sched/backfill
SchedulerParameters=bf_licenses,bf_interval=2,sched_interval=2
PriorityType=priority/basic
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

Licenses=host_activity:2,build_slot:1
AccountingStorageType=accounting_storage/none
JobCompType=jobcomp/filetxt
JobCompLoc=/var/log/slurm/bigcherry-jobcomp.log
RequeueExit=75

ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
JobAcctGatherType=jobacct_gather/cgroup
JobAcctGatherFrequency=30

PartitionName=bc-build   Nodes=brutus PriorityTier=10  State=UP MaxTime=00:45:00
PartitionName=bc-measure Nodes=brutus PriorityTier=100 State=UP MaxTime=00:45:00
```

Resource classes:

```text
build / prepare:
  partition=bc-build
  licenses=build_slot:1,host_activity:1

monolithic v1 / timed execute:
  partition=bc-measure
  licenses=host_activity:2
  required architecture GRES
```

Real Noble CI proved a running build plus pending measure plus later build executes:

```text
build1 -> measure -> build2
```

Do not classify jobs solely from `Reason=` text. Pending reasons are diagnostic and were observed to vary while valid jobs still made correct progress.

## 10. BigCherry -> Slurm submission contract

Only `tools/bigcherry/jobs/slurm.py` may know Slurm CLI syntax.

Expected argv shape:

```bash
sbatch --parsable \
  --job-name 'bc:<run_id>:a<attempt>' \
  --comment 'bigcherry:<execution_id>' \
  --partition bc-measure \
  --licenses host_activity:2 \
  --gres 'gpu:<arch>:<reserved-count>' \
  --time '<bounded>' \
  --output '<attempt>/slurm-%j.out' \
  --error '<attempt>/slurm-%j.err' \
  --export 'ALL,BIGCHERRY_RUN_ID=...,BIGCHERRY_ATTEMPT=...,BIGCHERRY_ATTEMPT_ROOT=...' \
  '<attempt>/launch.sh'
```

No `--account` in v1.

Before `sbatch`, persist `submission-intent.json` with stable `execution_id`, exact request hash, run/attempt and pinned commit. Persist `submission.json` only after native acceptance.

Recovery when intent exists but handle is missing:

```text
one correlated native execution -> bind it
proven no execution             -> submit immutable request once
multiple/ambiguous              -> block + wake; never duplicate blindly
```

Use active Slurm state plus `jobcomp/filetxt` and attempt-local start/result sentinels because a very short terminal job may disappear from `squeue` before handle persistence.

Machine interfaces:

```bash
squeue --json
scontrol show job -o <jobid>
scontrol hold <jobid>
scontrol release <jobid>
scancel <jobid>
```

No v1 dependency on `sacct`.

## 11. AMD GRES/cgroup hardware falsification

Run for every accepted Linux ROCm toolchain and allocation shape:

```text
single GPU for each accepted architecture/card class
all relevant same-arch single-card cohorts
dual/sm-tensor cohort where applicable
```

With `ConstrainDevices=yes`, prove:

```text
/dev/kfd usable
allocated render node(s) usable
unallocated render node(s) inaccessible
ROCR_VISIBLE_DEVICES resolves exactly to allocated/bound devices
HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES not conflicting
hipGetDeviceCount == requested count
hipGetDeviceProperties succeeds for every visible device
rocminfo/AMD-SMI inspection terminates
100 timeout-bounded HIP init/enumeration cycles: no hang/error/order drift
llama-bench smoke succeeds
llama-server attestation succeeds
peer access / -sm tensor succeeds where required
4096-context producer preflight succeeds where required
```

If **any supported cell fails**, restore:

```ini
ConstrainDevices=no
```

Keep GRES scheduling, Slurm visibility and BigCherry stable-ID attestation. Record explicitly that this is not a device security boundary.

## 12. Production coexistence

Slurm does not own llama-swap.

After GPU allocation and before measurement, RCD11 derives production potential/running/observed stable-device sets from llama-swap config, `/running`, backend args/env and process/VRAM observation.

Policy:

```text
campaign target intersects production potential set
  -> privileged exclusive window; drain/stop production and block reload

disjoint
  -> production may remain loaded only under qualified idle/noise policy
```

Any production config/process/inventory drift during a measurement invalidates the sample as environment contamination.

No static production-GPU partition exists.

## 13. Hardware drift

On add/remove/replacement/BDF/render/topology change:

```bash
sudo scontrol update NodeName=brutus State=DRAIN Reason='BigCherry inventory drift'
bigcherry hardware discover --json
# inspect + explicitly accept new inventory
# regenerate slurm.conf Gres= and gres.conf
sudo slurmd -G
sudo systemctl restart slurmd   # when required by GRES change
sudo scontrol reconfigure
bigcherry jobs doctor --json
# rerun required hardware acceptance
sudo scontrol update NodeName=brutus State=RESUME
```

Same-model replacement or relevant topology change starts a new scientific hardware cohort/series. Existing series never silently rebound.

## 14. Real CPU/service smoke to reproduce on Brutus

CI implementation: `tools/lab/run-campaign-durability/slurm_noble_v3_smoke.sh`.

It currently proves on real Noble Slurm:

```text
Slurm 23.11.4
MUNGE round-trip
minimal no-db scheduler
no-account job completion
current-branch BigCherry process harness inside Slurm
hold/release
license/priority progress build1 -> measure -> build2
afterok dependency
RequeueExit=75 and SLURM_RESTART_COUNT 0 -> 1
cancel
controller restart with queued/running ownership retained
jobcomp/filetxt completion history
cgroup process/task/jobacct plugins with ConstrainDevices=no
```

Run the same logical smoke on Brutus before queue cutover.

## 15. Acceptance before queue retirement

GitHub real-service CI must remain green.

Brutus-only gates:

```text
slurmd -G with real accepted AMD inventory
real GRES -> stable-ID allocation attestation
ROCm cgroup matrix or explicit ConstrainDevices=no fallback
production conflict/non-conflict/window watchdog
inventory drift drain/reconcile
monolithic validation_campaign evidence parity on real GPUs
scheduler-isolation/noise qualification
forced incident/recovery matrix
one complete planned series with no shell watcher intervention
```

Only after RCD10 soak may `queue.sh`/`run_campaign.sh` retire.

## 16. Rollback

```bash
bigcherry jobs pause
sudo systemctl stop slurmd slurmctld
# preserve /var/spool/slurmctld, jobcomp and BigCherry run store
```

Use direct/manual campaign only after managed jobs are stopped. Never run two resource schedulers against the same GPUs concurrently.
