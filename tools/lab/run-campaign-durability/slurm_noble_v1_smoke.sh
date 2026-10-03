#!/usr/bin/env bash
set -euo pipefail

log() { printf '[rcd-slurm-v1] %s\n' "$*" >&2; }
die() { log "FAIL: $*"; exit 1; }

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends slurm-wlm munge jq

SLURM_VERSION="$(dpkg-query -W -f='${Version}' slurm-wlm | cut -d- -f1)"
case "$SLURM_VERSION" in 23.11.*) ;; *) die "expected Noble 23.11.x, got $SLURM_VERSION";; esac
HOST="$(hostname -s)"
CPUS="$(nproc)"
REALMEM="$(awk '/MemTotal:/ {m=int($2/1024)-1024; if (m<1024) m=int($2/1024*0.8); print m}' /proc/meminfo)"
REPO="${GITHUB_WORKSPACE:-$PWD}"
log "Slurm=$SLURM_VERSION host=$HOST cpus=$CPUS memory_mib=$REALMEM"

cleanup() {
  set +e
  sudo pkill -TERM -x slurmd >/dev/null 2>&1 || true
  sudo pkill -TERM -x slurmctld >/dev/null 2>&1 || true
  sudo pkill -TERM -x munged >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

sudo install -d -o munge -g munge -m 0755 /run/munge /var/log/munge
if [[ ! -s /etc/munge/munge.key ]]; then
  sudo sh -c 'umask 077; dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 status=none'
fi
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 0400 /etc/munge/munge.key
sudo -u munge /usr/sbin/munged --force
munge -n | unmunge >/dev/null
log "MUNGE round-trip ok"

sudo rm -rf /tmp/bc-slurm-state /tmp/bc-slurmd-spool
sudo install -d -o slurm -g slurm -m 0755 /tmp/bc-slurm-state
sudo install -d -o root -g root -m 0755 /tmp/bc-slurmd-spool /etc/slurm
sudo rm -f /tmp/bc-slurmctld.log /tmp/bc-slurmd.log /tmp/bc-slurmctld.stdout \
  /tmp/bc-slurmd.stdout /tmp/bc-slurm-jobcomp.log /tmp/bc-squeue.json \
  /tmp/bc-scontrol-config.txt /tmp/bc-order /tmp/bc-restarts /tmp/bc-bigcherry-slurm.json

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
AccountingStorageType=accounting_storage/none
JobCompType=jobcomp/filetxt
JobCompLoc=/tmp/bc-slurm-jobcomp.log
RequeueExit=75
MinJobAge=300
NodeName=${HOST} NodeAddr=127.0.0.1 CPUs=${CPUS} RealMemory=${REALMEM} State=UNKNOWN
PartitionName=bc-build Nodes=${HOST} PriorityTier=10 Default=NO State=UP MaxTime=00:10:00
PartitionName=bc-measure Nodes=${HOST} PriorityTier=100 Default=YES State=UP MaxTime=00:10:00
EOF
export SLURM_CONF=/etc/slurm/slurm.conf

# Slurm 23.11 has no slurmctld config-check-only option. Startup is the parse
# gate; GPU config is separately checked on Brutus with slurmd -G.
sudo slurmctld -Dvv >/tmp/bc-slurmctld.stdout 2>&1 &
sleep 1
sudo slurmd -Dvv >/tmp/bc-slurmd.stdout 2>&1 &

for _ in $(seq 1 60); do
  if scontrol ping 2>/dev/null | grep -q UP && sinfo -h -N -o '%T' 2>/dev/null | grep -Eq 'idle|mix|alloc'; then
    break
  fi
  sleep 0.5
done
scontrol ping | grep -q UP || { cat /tmp/bc-slurmctld.stdout >&2; die "controller unavailable"; }
sinfo -h -N -o '%T' | grep -Eq 'idle|mix|alloc' || {
  cat /tmp/bc-slurmd.stdout >&2 || true
  scontrol show node >&2 || true
  die "node unavailable"
}
scontrol show config >/tmp/bc-scontrol-config.txt
cat /tmp/bc-scontrol-config.txt | grep -E 'SchedulerType|SchedulerParameters|AccountingStorageType|Licenses' >&2 || true
scontrol show lic | grep -q host_activity || die "host_activity license missing"
squeue --json >/tmp/bc-squeue.json
jq -e '.jobs | type == "array"' /tmp/bc-squeue.json >/dev/null
log "controller/node/licenses/squeue-json live"

wait_state() {
  local job="$1" regex="$2" timeout="${3:-60}" state
  for _ in $(seq 1 $((timeout*2))); do
    state="$(scontrol show job -o "$job" 2>/dev/null | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')"
    [[ "$state" =~ $regex ]] && { printf '%s\n' "$state"; return 0; }
    sleep 0.5
  done
  scontrol show job "$job" >&2 || true
  return 1
}

SBATCH=(--parsable --output=/tmp/bc-%j.out --error=/tmp/bc-%j.err)

# No accounting database/account is required in v1: prove behavior, not config spelling.
BASIC="$(sbatch "${SBATCH[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='true')"
wait_state "$BASIC" COMPLETED 30 >/dev/null
log "no-account submission completed"

# Run current BigCherry production modules/CLI child-process harness *inside Slurm*.
printf -v BC_WRAP 'cd %q && PYTHONPATH=tools python tools/lab/run-campaign-durability/real_bigcherry_process_smoke.py > /tmp/bc-bigcherry-slurm.json' "$REPO"
BC_JOB="$(sbatch "${SBATCH[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:03:00 --wrap="$BC_WRAP")"
wait_state "$BC_JOB" COMPLETED 120 >/dev/null
jq -e '.ok == true and .scope == "real-bigcherry-modules-cli-and-child-processes"' /tmp/bc-bigcherry-slurm.json >/dev/null
log "real BigCherry process harness completed under Slurm job=$BC_JOB"

# Hold/release semantics used by jobs hold/release.
HOLD="$(sbatch "${SBATCH[@]}" --hold --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='sleep 1')"
wait_state "$HOLD" PENDING 10 >/dev/null
scontrol show job -o "$HOLD" | grep -q 'Reason=JobHeldUser' || die "hold reason missing"
scontrol release "$HOLD"
wait_state "$HOLD" COMPLETED 30 >/dev/null
log "hold/release ok"

# A higher-tier measurement blocked on both host_activity tokens must execute
# before a later low-tier build once the current build frees the token.
rm -f /tmp/bc-order
B1="$(sbatch "${SBATCH[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='echo build1 >> /tmp/bc-order; sleep 4')"
wait_state "$B1" RUNNING 20 >/dev/null
M="$(sbatch "${SBATCH[@]}" --partition=bc-measure --licenses=host_activity:2 --time=00:01:00 --wrap='echo measure >> /tmp/bc-order; sleep 2')"
B2="$(sbatch "${SBATCH[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='echo build2 >> /tmp/bc-order; sleep 1')"
wait_state "$M" PENDING 10 >/dev/null
PENDING_REASON="$(squeue -h -j "$M" -o '%R')"
wait_state "$M" COMPLETED 60 >/dev/null
wait_state "$B2" COMPLETED 60 >/dev/null
mapfile -t ORDER </tmp/bc-order
MI=-1; BI=-1
for i in "${!ORDER[@]}"; do
  [[ "${ORDER[$i]}" == measure ]] && MI="$i"
  [[ "${ORDER[$i]}" == build2 ]] && BI="$i"
done
(( MI >= 0 && BI >= 0 && MI < BI )) || die "measurement starved/reordered: ${ORDER[*]}"
log "license/priority progress ok: ${ORDER[*]} (initial reason=$PENDING_REASON)"

# v1.5 dependency contract.
rm -f /tmp/bc-prepare /tmp/bc-execute
P="$(sbatch "${SBATCH[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='echo prepared > /tmp/bc-prepare')"
E="$(sbatch "${SBATCH[@]}" --partition=bc-measure --licenses=host_activity:2 --dependency=afterok:${P} --time=00:01:00 --wrap='test -s /tmp/bc-prepare && echo executed > /tmp/bc-execute')"
wait_state "$E" COMPLETED 60 >/dev/null
test -s /tmp/bc-execute || die "afterok dependent did not execute"
log "afterok dependency ok"

# Exit 75 must requeue the same native job and increment restart count.
cat >/tmp/bc-requeue.sh <<'EOS'
#!/usr/bin/env bash
set -eu
printf '%s\n' "${SLURM_RESTART_COUNT:-0}" >> /tmp/bc-restarts
(( ${SLURM_RESTART_COUNT:-0} < 1 )) && exit 75
exit 0
EOS
chmod +x /tmp/bc-requeue.sh
rm -f /tmp/bc-restarts
R="$(sbatch "${SBATCH[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 /tmp/bc-requeue.sh)"
wait_state "$R" COMPLETED 60 >/dev/null
mapfile -t RESTARTS </tmp/bc-restarts
[[ "${RESTARTS[*]}" == '0 1' ]] || die "restart counts=${RESTARTS[*]}"
log "RequeueExit=75 ok: ${RESTARTS[*]}"

C="$(sbatch "${SBATCH[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:02:00 --wrap='sleep 60')"
wait_state "$C" RUNNING 20 >/dev/null
scancel "$C"
wait_state "$C" CANCELLED 30 >/dev/null
log "cancel ok"

# Controller checkpoint/recovery must retain queued and running ownership.
Q="$(sbatch "${SBATCH[@]}" --hold --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='true')"
LONG="$(sbatch "${SBATCH[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='sleep 5')"
wait_state "$LONG" RUNNING 20 >/dev/null
sudo pkill -TERM -x slurmctld
for _ in $(seq 1 20); do ! pgrep -x slurmctld >/dev/null && break; sleep 0.25; done
sudo slurmctld -Dvv >>/tmp/bc-slurmctld.stdout 2>&1 &
for _ in $(seq 1 40); do scontrol ping 2>/dev/null | grep -q UP && break; sleep 0.5; done
scontrol ping | grep -q UP || die "controller did not recover"
wait_state "$Q" PENDING 10 >/dev/null
scontrol release "$Q"
wait_state "$Q" COMPLETED 30 >/dev/null
wait_state "$LONG" COMPLETED 30 >/dev/null
log "controller restart retained queued/running ownership"

for _ in $(seq 1 20); do [[ -s /tmp/bc-slurm-jobcomp.log ]] && break; sleep 0.5; done
test -s /tmp/bc-slurm-jobcomp.log || die "jobcomp/filetxt empty"
log "jobcomp/filetxt ok"

jq -n \
  --arg slurm_version "$SLURM_VERSION" \
  --arg bigcherry_job "$BC_JOB" \
  --arg pending_reason "$PENDING_REASON" \
  --arg schedule_order "${ORDER[*]}" \
  --arg restart_counts "${RESTARTS[*]}" \
  '{ok:true,scope:"real-noble-slurm-minimal-v1-no-gpu",slurm_version:$slurm_version,bigcherry_slurm_job:$bigcherry_job,accounting:"none+jobcomp/filetxt",pending_reason:$pending_reason,schedule_order:$schedule_order,restart_counts:$restart_counts}'
