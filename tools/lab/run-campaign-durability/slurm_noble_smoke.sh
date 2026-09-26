#!/usr/bin/env bash
set -euo pipefail

# Real Slurm/MUNGE smoke for an Ubuntu 24.04 disposable host (GitHub Actions).
# This validates scheduler/service semantics only. It intentionally does not
# claim AMD GRES/cgroup/ROCm hardware qualification.

log() { printf '[rcd-slurm-smoke] %s\n' "$*" >&2; }
die() { log "FAIL: $*"; exit 1; }

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends slurm-wlm munge jq

SLURM_VERSION="$(scontrol --version | awk '{print $2}')"
log "installed Slurm ${SLURM_VERSION}"
case "$SLURM_VERSION" in
  23.11.*) ;;
  *) die "expected Ubuntu Noble Slurm 23.11.x, got ${SLURM_VERSION}" ;;
esac

HOST="$(hostname -s)"
CPUS="$(nproc)"
REALMEM="$(awk '/MemTotal:/ {m=int($2/1024)-1024; if (m < 1024) m=int($2/1024*0.8); print m}' /proc/meminfo)"

cleanup() {
  set +e
  sudo pkill -TERM -x slurmd >/dev/null 2>&1 || true
  sudo pkill -TERM -x slurmctld >/dev/null 2>&1 || true
  sudo pkill -TERM -x munged >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

# MUNGE, started manually because GitHub-hosted runners do not rely on systemd.
sudo install -d -o munge -g munge -m 0755 /run/munge /var/log/munge
if [[ ! -s /etc/munge/munge.key ]]; then
  sudo sh -c 'umask 077; dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 status=none'
fi
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 0400 /etc/munge/munge.key
sudo -u munge /usr/sbin/munged --force
munge -n | unmunge >/dev/null
log "MUNGE round-trip ok"

sudo install -d -o slurm -g slurm -m 0755 /tmp/bc-slurm-state
sudo install -d -o root -g root -m 0755 /tmp/bc-slurmd-spool
sudo install -d -o root -g root -m 0755 /etc/slurm

sudo tee /etc/slurm/slurm.conf >/dev/null <<EOF
ClusterName=bigcherry-ci
SlurmctldHost=${HOST}(127.0.0.1)
SlurmUser=slurm
SlurmdUser=root
AuthType=auth/munge
CredType=cred/munge
StateSaveLocation=/tmp/bc-slurm-state
SlurmdSpoolDir=/tmp/bc-slurmd-spool
SlurmctldLogFile=/tmp/bc-slurmctld.log
SlurmdLogFile=/tmp/bc-slurmd.log
SlurmctldDebug=info
SlurmdDebug=info
SwitchType=switch/none
MpiDefault=none
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
ReturnToService=2
SchedulerType=sched/backfill
SchedulerParameters=bf_licenses
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU
PriorityType=priority/multifactor
Licenses=host_activity:2,build_slot:1
JobCompType=jobcomp/filetxt
JobCompLoc=/tmp/bc-slurm-jobcomp.log
RequeueExit=75
MinJobAge=300
NodeName=${HOST} NodeAddr=127.0.0.1 CPUs=${CPUS} RealMemory=${REALMEM} State=UNKNOWN
PartitionName=bc-build Nodes=${HOST} PriorityTier=10 Default=NO State=UP MaxTime=00:10:00
PartitionName=bc-measure Nodes=${HOST} PriorityTier=100 Default=YES State=UP MaxTime=00:10:00
EOF

# Validate controller config before daemon start.
sudo slurmctld -t
log "slurmctld config test ok"

sudo slurmctld -Dvv > /tmp/bc-slurmctld.stdout 2>&1 &
CTLD_PID=$!
sleep 1
sudo slurmd -Dvv > /tmp/bc-slurmd.stdout 2>&1 &
SLURMD_PID=$!

for _ in $(seq 1 40); do
  if scontrol ping 2>/dev/null | grep -q 'UP'; then
    if sinfo -h -N -o '%T' 2>/dev/null | grep -Eq 'idle|mix|alloc'; then
      break
    fi
  fi
  sleep 0.5
done
scontrol ping | grep -q 'UP' || { cat /tmp/bc-slurmctld.stdout >&2; die "slurmctld not UP"; }
sinfo -N -l
log "single-node Slurm service is live"

# Machine-readable status is a hard adapter contract.
squeue --json >/tmp/bc-squeue.json
jq -e '.jobs | type == "array"' /tmp/bc-squeue.json >/dev/null
log "squeue --json contract ok"

wait_state() {
  local job="$1" regex="$2" timeout="${3:-60}" state
  for _ in $(seq 1 $((timeout * 2))); do
    state="$(scontrol show job -o "$job" 2>/dev/null | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')"
    if [[ "$state" =~ $regex ]]; then
      printf '%s\n' "$state"
      return 0
    fi
    sleep 0.5
  done
  scontrol show job "$job" >&2 || true
  return 1
}

# Hold/release maps cleanly to Executor.control().
HOLD_JOB="$(sbatch --parsable --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='sleep 1')"
scontrol hold "$HOLD_JOB"
wait_state "$HOLD_JOB" 'PENDING' 10 >/dev/null
scontrol show job -o "$HOLD_JOB" | grep -q 'Reason=JobHeldUser' || die "hold reason not visible"
scontrol release "$HOLD_JOB"
wait_state "$HOLD_JOB" 'COMPLETED' 30 >/dev/null
log "hold/release contract ok"

# PriorityTier + bf_licenses must prevent queued build stream from jumping a
# pending host_activity:2 measurement once the currently-running build exits.
rm -f /tmp/bc-order
BUILD1="$(sbatch --parsable --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='echo build1 >> /tmp/bc-order; sleep 4')"
wait_state "$BUILD1" 'RUNNING' 20 >/dev/null
MEASURE="$(sbatch --parsable --partition=bc-measure --licenses=host_activity:2 --time=00:01:00 --wrap='echo measure >> /tmp/bc-order; sleep 2')"
BUILD2="$(sbatch --parsable --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='echo build2 >> /tmp/bc-order; sleep 1')"
wait_state "$MEASURE" 'PENDING' 10 >/dev/null
PENDING_REASON="$(squeue -h -j "$MEASURE" -o '%R')"
[[ "$PENDING_REASON" == *License* || "$PENDING_REASON" == *Priority* || "$PENDING_REASON" == *Resources* ]] || die "unexpected measurement pending reason: ${PENDING_REASON}"
wait_state "$MEASURE" 'COMPLETED' 60 >/dev/null
wait_state "$BUILD2" 'COMPLETED' 60 >/dev/null
mapfile -t ORDER </tmp/bc-order
printf 'schedule-order=%s\n' "${ORDER[*]}"
MEASURE_IDX=-1; BUILD2_IDX=-1
for i in "${!ORDER[@]}"; do
  [[ "${ORDER[$i]}" == measure ]] && MEASURE_IDX="$i"
  [[ "${ORDER[$i]}" == build2 ]] && BUILD2_IDX="$i"
done
(( MEASURE_IDX >= 0 && BUILD2_IDX >= 0 && MEASURE_IDX < BUILD2_IDX )) || die "measurement did not run before later build"
log "license/priority progress contract ok"

# afterok dependency is the v1.5 prepare -> execute mechanism.
rm -f /tmp/bc-prepare /tmp/bc-execute
PREP="$(sbatch --parsable --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='echo prepared > /tmp/bc-prepare')"
EXEC="$(sbatch --parsable --partition=bc-measure --licenses=host_activity:2 --dependency=afterok:${PREP} --time=00:01:00 --wrap='test -s /tmp/bc-prepare && echo executed > /tmp/bc-execute')"
wait_state "$EXEC" 'COMPLETED' 60 >/dev/null
test -s /tmp/bc-execute || die "afterok dependent did not execute"
log "dependency contract ok"

# RequeueExit=75 must restart the same native job and increment restart count.
cat >/tmp/bc-requeue.sh <<'EOS'
#!/usr/bin/env bash
set -eu
printf '%s\n' "${SLURM_RESTART_COUNT:-0}" >> /tmp/bc-restarts
if (( ${SLURM_RESTART_COUNT:-0} < 1 )); then
  exit 75
fi
exit 0
EOS
chmod +x /tmp/bc-requeue.sh
rm -f /tmp/bc-restarts
REQUEUE="$(sbatch --parsable --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 /tmp/bc-requeue.sh)"
wait_state "$REQUEUE" 'COMPLETED' 60 >/dev/null
mapfile -t RESTARTS </tmp/bc-restarts
printf 'restart-counts=%s\n' "${RESTARTS[*]}"
[[ "${RESTARTS[*]}" == "0 1" ]] || die "expected restart counts '0 1'"
log "RequeueExit/SLURM_RESTART_COUNT contract ok"

# Cancellation semantics.
CANCEL="$(sbatch --parsable --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:02:00 --wrap='sleep 60')"
wait_state "$CANCEL" 'RUNNING' 20 >/dev/null
scancel "$CANCEL"
wait_state "$CANCEL" 'CANCELLED' 30 >/dev/null
log "cancel contract ok"

# Basic completion history without slurmdbd.
for _ in $(seq 1 20); do
  [[ -s /tmp/bc-slurm-jobcomp.log ]] && break
  sleep 0.5
done
test -s /tmp/bc-slurm-jobcomp.log || die "jobcomp/filetxt stayed empty"
log "jobcomp/filetxt contract ok"

jq -n \
  --arg slurm_version "$SLURM_VERSION" \
  --arg host "$HOST" \
  --arg pending_reason "$PENDING_REASON" \
  --arg schedule_order "${ORDER[*]}" \
  --arg restart_counts "${RESTARTS[*]}" \
  '{ok:true, scope:"real-slurm-services-no-gpu", slurm_version:$slurm_version, host:$host, pending_reason:$pending_reason, schedule_order:$schedule_order, restart_counts:$restart_counts}'
