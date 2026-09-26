# Brutus Slurm 23.11 install/integration runbook

Normative architecture: `docs/design/JOBS_ORCHESTRATOR.md`. This is the concrete RCD03 host procedure. Slurm owns execution/resource scheduling; BigCherry owns scientific identity, commit pinning, production coexistence, retry legality, evidence and reporting.

## 1. Packages

Ubuntu 24.04/Noble target: Slurm 23.11.4.x.

```bash
sudo apt update
sudo apt install -y slurm-wlm slurmdbd munge mariadb-server jq
getent passwd slurm
getent passwd bigcherry
```

`slurmrestd` is not used in v1.

### Why slurmdbd is required

The real Noble CI (`tools/lab/run-campaign-durability/slurm_noble_smoke.sh`) installed 23.11.4, started `slurmctld/slurmd`, registered the node and ran an initial hold/release test. Subsequent jobs reproduced `PENDING Reason=InvalidAccount` with no accounting association. v1 therefore treats a minimal local `slurmdbd` association store as scheduler correctness, not optional analytics.

MariaDB is host-local only; no LAN listener is required.

## 2. MUNGE

```bash
sudo install -d -o munge -g munge -m 0755 /run/munge /var/log/munge
if ! sudo test -s /etc/munge/munge.key; then
  sudo sh -c 'umask 077; dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 status=none'
fi
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 0400 /etc/munge/munge.key
sudo systemctl enable --now munge
munge -n | unmunge
```

The round-trip must pass before `slurmdbd`/`slurmctld` start.

## 3. Local accounting database

Bind MariaDB to loopback only:

```ini
# /etc/mysql/mariadb.conf.d/99-bigcherry-slurm.cnf
[mysqld]
bind-address = 127.0.0.1
skip-name-resolve
```

```bash
sudo systemctl restart mariadb
sudo mariadb <<'SQL'
CREATE DATABASE IF NOT EXISTS slurm_acct_db;
CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY '<HOST-SECRET>';
CREATE USER IF NOT EXISTS 'slurm'@'127.0.0.1' IDENTIFIED BY '<HOST-SECRET>';
GRANT ALL PRIVILEGES ON slurm_acct_db.* TO 'slurm'@'localhost';
GRANT ALL PRIVILEGES ON slurm_acct_db.* TO 'slurm'@'127.0.0.1';
FLUSH PRIVILEGES;
SQL
```

Do not commit the DB password. Render `config/slurm/slurmdbd.conf.example` to `/etc/slurm/slurmdbd.conf`, substitute the root-owned secret, then:

```bash
sudo chown slurm:slurm /etc/slurm/slurmdbd.conf
sudo chmod 0600 /etc/slurm/slurmdbd.conf
sudo systemctl enable --now slurmdbd
sacctmgr ping
```

Create exactly the minimal scheduling associations:

```bash
sudo sacctmgr -i add cluster bigcherry
sudo sacctmgr -i add account bigcherry Cluster=bigcherry Description=BigCherry Organization=BigCherry
sudo sacctmgr -i add user bigcherry Account=bigcherry DefaultAccount=bigcherry Cluster=bigcherry
sacctmgr -nP show assoc cluster=bigcherry account=bigcherry
```

If named human Unix users submit Slurm jobs directly, associate them explicitly too; otherwise SSH operators submit through the `bigcherry` service account.

## 4. Host directories

```bash
sudo install -d -o slurm -g slurm -m 0755 /var/spool/slurmctld
sudo install -d -o root  -g root  -m 0755 /var/spool/slurmd
sudo install -d -o slurm -g slurm -m 0755 /var/log/slurm
sudo install -d -o root  -g root  -m 0755 /etc/slurm
sudo touch /var/log/slurm/bigcherry-jobcomp.log
sudo chown slurm:slurm /var/log/slurm/bigcherry-jobcomp.log
```

BigCherry work/cache/log roots remain on `/mnt/data`; controller/accounting state is small host state.

## 5. Hardware discovery before Slurm render

RCD12 discovered/accepted inventory is authoritative. Never hand-maintain slot identity.

Required per GPU: stable `device_id` (AMD UUID/RSMI unique id/serial or explicit weak epoch), arch, model, VRAM, PCI BDF, render node, NUMA/topology, driver.

Target commands:

```bash
bigcherry hardware discover --json > /mnt/data/bigcherry-jobs/hardware/observed.json
bigcherry hardware diff --accepted --json
bigcherry hardware accept --inventory-hash <sha256>
```

Until RCD12 implements those commands, its generated accepted inventory is required. `environment.local.toml` remains host policy/toolchain/model configuration, not GPU truth.

CPU topology baseline:

```bash
slurmd -C
```

Render current CPU count and conservative `RealMemory`; never copy another host's line.

## 6. Render `/etc/slurm`

Templates:

- `config/slurm/slurm.conf.example`
- `config/slurm/slurmdbd.conf.example`
- `config/slurm/cgroup.conf.example`
- `config/slurm/gres.conf.example`

`slurm.conf` placeholders:

```text
@NODE_NAME@        current Brutus hostname
@CPUS@             configured CPUs
@REAL_MEMORY_MIB@  conservative host RAM
@NODE_GRES@        accepted architecture counts, e.g. gpu:gfx1100:2,gpu:gfx1201:1,gpu:gfx1030:1
```

Generated `/etc/slurm/gres.conf`:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1100 File=/dev/dri/renderD129 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
Name=gpu Type=gfx1030 File=/dev/dri/renderD131 Flags=amd_gpu_env
```

Rules:

- `Type` is architecture only; never `gfx1100_0`.
- BDF/render node/ordinal are locators, not scientific identity.
- Slurm `Flags=amd_gpu_env` owns `ROCR_VISIBLE_DEVICES`; BigCherry must not overwrite it with host-global ordinals.
- if a series is bound to a proper subset of same-arch cards, v1 requests all cards of that architecture, verifies the allocation contains the pre-bound stable IDs, then narrows visibility inside the already-exclusive allocation.

Before applying GPU config:

```bash
sudo slurmd -G
```

Any error blocks node resume.

## 7. Start order and service health

```bash
sudo systemctl enable slurmdbd slurmctld slurmd
sudo systemctl restart slurmdbd
sacctmgr ping
sudo systemctl restart slurmctld
sudo systemctl restart slurmd

scontrol ping
sinfo -Nel
scontrol show config | grep -E 'SchedulerType|SchedulerParameters|Licenses|RequeueExit|AccountingStorage'
scontrol show lic
squeue --json | jq '.jobs'
sacctmgr -nP show assoc cluster=bigcherry account=bigcherry
```

Required v1 services: `mariadb`, `munge`, `slurmdbd`, `slurmctld`, `slurmd`.

## 8. Resource model

```text
Licenses=host_activity:2,build_slot:1
bc-build   PriorityTier=10
bc-measure PriorityTier=100
SchedulerType=sched/backfill
SchedulerParameters=bf_licenses
AccountingStorageEnforce=associations
```

All BigCherry submissions explicitly pass `--account=bigcherry`.

```text
monolithic v1:
  partition=bc-measure
  account=bigcherry
  licenses=host_activity:2
  finite --time
  architecture GRES/all-of-arch reservation

v1.5 prepare:
  bc-build + build_slot:1,host_activity:1

v1.5 execute:
  bc-measure + host_activity:2 + --dependency=afterok:<prepare>
```

The GitHub Noble smoke proves the configured license/priority behavior before Brutus cutover.

## 9. BigCherry -> Slurm submission contract

Only `tools/bigcherry/jobs/slurm.py` may know Slurm syntax.

```bash
sbatch --parsable \
  --account bigcherry \
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

`launch.sh` runs the exact pinned attempt checkout/canonical campaign argv; it never resolves a branch. `submission-intent.json` is durable before `sbatch`; returned native ID is persisted after acceptance. Recovery correlates by `--comment=bigcherry:<execution_id>` and refuses duplicate submit when acceptance is ambiguous.

Machine interfaces:

```bash
squeue --json
scontrol show job -o <jobid>
scontrol hold <jobid>
scontrol release <jobid>
scancel <jobid>
sacct -j <jobid>
```

Never parse human `squeue` columns in production adapter code.

## 10. Retry semantics

```text
0   execution completed; BigCherry result may be scientific PASS or FAIL
75  same-commit transient; Slurm RequeueExit=75; same native job/attempt
76  harness/code fix required; jobs retry --latest => attempt+1/new native job
77  invalid input or contract/composition drift; block
```

Record `SLURM_RESTART_COUNT` on every exit-75 restart. Scientific results never trigger automatic requeue.

## 11. Production coexistence

Slurm does not own llama-swap. After allocation and before measurement, RCD11 derives `ProductionSnapshot` from configured potential GPU claims plus live process/runtime observation.

- target intersects any potential production GPU -> privileged exclusive window; production reload is blocked while measurement runs.
- disjoint -> production may remain loaded but must satisfy the currently qualified idle/noise policy.
- production config/process/inventory drift during a sample -> terminate/contamination; no scientific result.

No static production-GPU partition exists.

## 12. cgroup qualification/fallback

Candidate production mode:

```text
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainDevices=yes
```

For every accepted ROCm/toolchain/cohort prove `/dev/kfd`, allocated render access, blocked unallocated render nodes, exact visibility/count/order, 100 bounded HIP init cycles, llama-bench/server, and peer/`-sm tensor` paths.

On any supported-cell failure:

```text
ConstrainDevices=no
```

Keep GRES scheduling + Slurm visibility + BigCherry attestation; record that GRES is not a security boundary.

## 13. Hardware drift

On add/remove/replacement/BDF/render/topology change:

```bash
sudo scontrol update NodeName=brutus State=DRAIN Reason='BigCherry inventory drift'
bigcherry hardware discover --json
# operator accepts new inventory/cohort after inspection
# regenerate slurm.conf Gres + gres.conf
sudo slurmd -G
sudo systemctl restart slurmd   # when required for GRES change
sudo scontrol reconfigure
bigcherry jobs doctor --json
# run hardware acceptance
sudo scontrol update NodeName=brutus State=RESUME
```

Same-model replacement or relevant topology change starts a new hardware cohort/series; existing series never silently rebound.

## 14. Acceptance before queue retirement

GitHub Noble real-service CI must pass:

```text
exact Slurm 23.11.4 package
MUNGE round-trip
loopback MariaDB + slurmdbd
cluster/account/user association
slurmctld/slurmd node registration
squeue --json
hold/release/cancel
afterok dependency
license/priority progress
RequeueExit=75 + SLURM_RESTART_COUNT
jobcomp/filetxt + sacct record
real BigCherry run_managed/CLI/HI48/validation_campaign dispatch smoke
```

Brutus-only gates:

```text
generated slurmd -G with real AMD inventory
real AMD GRES -> stable-ID attestation
ROCm cgroup matrix or explicit fallback
production conflict/non-conflict/window watchdog
inventory drift drain/reconcile
monolithic validation_campaign evidence parity
scheduler-isolation measurements
```

Only after RCD10 soak may `queue.sh`/`run_campaign.sh` retire.

## 15. Rollback

```bash
bigcherry jobs pause
sudo systemctl stop slurmd slurmctld slurmdbd
# preserve controller/accounting state, jobcomp, and BigCherry run store
```

Use direct/manual campaign only after managed jobs stop; never run two resource schedulers against the same GPUs concurrently.
