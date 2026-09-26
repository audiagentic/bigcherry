# Brutus Slurm 23.11 install/integration runbook

Normative architecture: `docs/design/JOBS_ORCHESTRATOR.md`. This document is the concrete host procedure for RCD03. Slurm owns execution/resource scheduling only; BigCherry owns scientific/domain identity, commit pinning, production coexistence, retry legality, evidence and reporting.

## 1. Packages and accounts

Ubuntu 24.04/Noble target: `slurm-wlm 23.11.4-*`, `munge`.

```bash
sudo apt update
sudo apt install -y slurm-wlm munge
scontrol -V                       # after /etc/slurm/slurm.conf exists
getent passwd slurm
getent passwd bigcherry           # create dedicated service/operator account if absent
```

Do not install MariaDB/slurmdbd/slurmrestd for v1.

## 2. MUNGE

Single-host key only; never commit it.

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

MUNGE round-trip must succeed before Slurm services are started.

## 3. Host directories

```bash
sudo install -d -o slurm -g slurm -m 0755 /var/spool/slurmctld
sudo install -d -o root  -g root  -m 0755 /var/spool/slurmd
sudo install -d -o slurm -g slurm -m 0755 /var/log/slurm
sudo install -d -o root  -g root  -m 0755 /etc/slurm
sudo touch /var/log/slurm/bigcherry-jobcomp.log
sudo chown slurm:slurm /var/log/slurm/bigcherry-jobcomp.log
```

BigCherry work/cache/log roots remain on `/mnt/data`; Slurm controller state stays small under `/var/spool/slurmctld`.

## 4. Hardware discovery before config

RCD12 discovered/accepted inventory is authoritative. Never type GPU slot numbers into the policy by hand.

Required accepted record per GPU: stable `device_id` (AMD UUID/RSMI unique id/serial or explicit weak epoch), arch, model, VRAM, PCI BDF, render node, NUMA/topology, driver.

Before first render and after any card/slot change:

```bash
# implementation target
bigcherry hardware discover --json > /mnt/data/bigcherry-jobs/hardware/observed.json
bigcherry hardware diff --accepted --json
bigcherry hardware accept --inventory-hash <sha256>    # explicit operator action only
```

Until those commands are implemented, RCD12 must provide an equivalent generated `accepted.json`; `environment.local.toml` is policy/toolchain/model configuration, not GPU truth.

CPU topology baseline:

```bash
slurmd -C
```

Use discovered CPU count and a conservative `RealMemory` below actual host RAM. Do not copy another host's NodeName/CPU/RAM line.

## 5. Render `/etc/slurm`

Templates:

- `config/slurm/slurm.conf.example`
- `config/slurm/cgroup.conf.example`
- `config/slurm/gres.conf.example`

Render placeholders:

```text
@NODE_NAME@        = current Brutus hostname
@CPUS@             = current configured CPUs
@REAL_MEMORY_MIB@  = conservative current host RAM
@NODE_GRES@        = architecture counts from accepted inventory, e.g.
                     gpu:gfx1100:2,gpu:gfx1201:1,gpu:gfx1030:1
```

Generated `/etc/slurm/gres.conf` has one line per current render node:

```text
Name=gpu Type=gfx1100 File=/dev/dri/renderD128 Flags=amd_gpu_env
Name=gpu Type=gfx1100 File=/dev/dri/renderD129 Flags=amd_gpu_env
Name=gpu Type=gfx1201 File=/dev/dri/renderD130 Flags=amd_gpu_env
Name=gpu Type=gfx1030 File=/dev/dri/renderD131 Flags=amd_gpu_env
```

Rules:

- `Type` is architecture only; never `gfx1100_0`.
- PCI BDF/render node/ordinal are current locators, not scientific identity.
- `Flags=amd_gpu_env` lets Slurm emit `ROCR_VISIBLE_DEVICES`.
- BigCherry preserves Slurm visibility; it must not overwrite it with host-global ordinals.
- proper-subset scientific cohorts reserve all GPUs of that architecture in v1, then BigCherry narrows to pre-bound stable IDs inside the exclusive allocation.

Before applying any GPU config:

```bash
sudo slurmd -G
```

Any error blocks service resume.

## 6. Start services

```bash
sudo systemctl enable slurmctld slurmd
sudo systemctl restart slurmctld
sudo systemctl restart slurmd

scontrol ping
sinfo -Nel
scontrol show config | grep -E 'SchedulerType|SchedulerParameters|Licenses|RequeueExit'
scontrol show lic
squeue --json | jq '.jobs'
```

Required v1 services: `munge`, `slurmctld`, `slurmd`. `slurmdbd` is intentionally absent.

## 7. Resource model

`slurm.conf` defines:

```text
Licenses=host_activity:2,build_slot:1
bc-build   PriorityTier=10
bc-measure PriorityTier=100
SchedulerType=sched/backfill
SchedulerParameters=bf_licenses
```

Requests:

```text
monolithic v1 campaign:
  partition=bc-measure
  licenses=host_activity:2
  finite --time
  architecture GRES/all-of-arch reservation from SeriesGpuBinding

v1.5 prepare:
  partition=bc-build
  licenses=build_slot:1,host_activity:1

v1.5 execute:
  partition=bc-measure
  licenses=host_activity:2
  dependency=afterok:<prepare-job-id>
```

`bf_licenses` + the higher measurement `PriorityTier` are acceptance-tested by `tools/lab/run-campaign-durability/slurm_noble_smoke.sh`: a queued measurement must run before later build work once the current build releases `host_activity`.

## 8. BigCherry -> Slurm submission contract

Only `tools/bigcherry/jobs/slurm.py` may know Slurm syntax.

Exact v1 shape:

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

The generated `launch.sh` runs the exact pinned-attempt checkout and canonical campaign argv; it does not resolve a branch. `submission-intent.json` is durable before `sbatch`; returned native job ID is persisted after acceptance. Recovery correlates by `--comment=bigcherry:<execution_id>` and refuses duplicate submit if acceptance is ambiguous.

Machine interfaces:

```bash
squeue --json
scontrol show job -o <jobid>
scontrol hold <jobid>
scontrol release <jobid>
scancel <jobid>
```

Never parse human `squeue` columns in production code.

## 9. Retry semantics

```text
exit 0   execution completed; BigCherry result may be scientific PASS or FAIL
exit 75  same-commit transient; Slurm RequeueExit=75; same Slurm job ID/attempt
exit 76  harness/code fix required; terminate; `jobs retry --latest` creates attempt+1/new Slurm job
exit 77  invalid input / contract/composition drift; block
```

`SLURM_RESTART_COUNT` is recorded on every exit-75 restart. Scientific outcomes never trigger automatic requeue.

## 10. Production coexistence

Slurm does not own llama-swap. Every measurement wrapper obtains an RCD11 `ProductionSnapshot` after allocation and before campaign start.

- allocated stable IDs intersect any model's potential production set -> privileged exclusive production window; llama-swap cannot reload while measurement runs.
- no intersection -> production remains up but must satisfy configured idle/noise attestation until active coexistence is empirically qualified.
- production config/process/inventory drift during a sample -> terminate sample, classify environment contamination, no scientific result.

No static production GPU partition exists.

## 11. cgroup qualification/fallback

Default candidate:

```text
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainDevices=yes
```

For each accepted ROCm/toolchain/device cohort prove `/dev/kfd`, allocated render access, blocked unallocated render nodes, exact visible count/order, 100 bounded HIP init loops, llama-bench/server, and peer/`-sm tensor` paths.

If any supported cell fails:

```text
ConstrainDevices=no
```

Keep GRES scheduling + Slurm `ROCR_VISIBLE_DEVICES` + BigCherry attestation. Record the fallback in status/acceptance; do not claim cgroup isolation.

## 12. Hardware drift/change procedure

On add/remove/replacement/BDF/render/topology change:

```bash
sudo scontrol update NodeName=brutus State=DRAIN Reason='BigCherry inventory drift'
# wait/cancel jobs according to contamination policy
bigcherry hardware discover --json
# operator reviews/accepts new inventory epoch/cohort
# regenerate slurm.conf node Gres and gres.conf
sudo slurmd -G
sudo systemctl restart slurmd       # required if live reconfigure cannot safely absorb GRES change
sudo scontrol reconfigure
bigcherry jobs doctor --json
# hardware acceptance smoke
sudo scontrol update NodeName=brutus State=RESUME
```

A same-model replacement card or relevant topology change starts a new hardware cohort/series. Never silently rebind existing series sessions.

## 13. Acceptance before queue retirement

Required:

```text
GitHub Noble real-Slurm CI passes:
  MUNGE
  slurmctld/slurmd registration
  squeue --json
  hold/release/cancel
  afterok dependency
  license/priority progress
  RequeueExit=75 + SLURM_RESTART_COUNT
  jobcomp/filetxt

Brutus-only passes:
  generated slurmd -G
  real AMD GRES allocation -> stable IDs
  cgroup matrix or explicit fallback
  production conflict/non-conflict paths
  dynamic inventory drift drain/reconcile
  monolithic validation_campaign evidence parity
```

Only after RCD10 soak may `queue.sh`/`run_campaign.sh` be retired.

## 14. Rollback

```bash
bigcherry jobs pause
sudo systemctl stop slurmd slurmctld
# retain /var/spool/slurmctld, jobcomp, and BigCherry run store read-only
```

Use direct/manual campaign path only after Slurm jobs are stopped; never run both resource schedulers against the same GPUs concurrently.
