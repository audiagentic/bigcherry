#!/usr/bin/env bash
set -euo pipefail

log() { printf '[rcd-slurm-v2] %s\n' "$*" >&2; }
die() { log "FAIL: $*"; exit 1; }

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends slurm-wlm munge jq
SLURM_VERSION="$(dpkg-query -W -f='${Version}' slurm-wlm | cut -d- -f1)"
case "$SLURM_VERSION" in 23.11.*) ;; *) die "expected Noble Slurm 23.11.x, got $SLURM_VERSION";; esac
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
sudo rm -f /tmp/bc-* /tmp/slurm-* 2>/dev/null || true

write_slurm_conf() {
  local proctrack="$1" taskplugin="$2" jobacct="$3"
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
ProctrackType=${proctrack}
TaskPlugin=${taskplugin}
ReturnToService=2
SchedulerType=sched/backfill
SchedulerParameters=bf_licenses,bf_interval=2,sched_interval=2,requeue_delay=2
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU
PriorityType=priority/basic
Licenses=host_activity:2,build_slot:1
AccountingStorageType=accounting_storage/none
JobCompType=jobcomp/filetxt
JobCompLoc=/tmp/bc-slurm-jobcomp.log
RequeueExit=75
MinJobAge=300
${jobacct}
NodeName=${HOST} NodeAddr=127.0.0.1 CPUs=${CPUS} RealMemory=${REALMEM} State=UNKNOWN
PartitionName=bc-build Nodes=${HOST} PriorityTier=10 Default=NO State=UP MaxTime=00:10:00
PartitionName=bc-measure Nodes=${HOST} PriorityTier=100 Default=YES State=UP MaxTime=00:10:00
EOF
  export SLURM_CONF=/etc/slurm/slurm.conf
}

start_slurm() {
  sudo slurmctld -Dvv >>/tmp/bc-slurmctld.stdout 2>&1 &
  sleep 0.5
  sudo slurmd -Dvv >>/tmp/bc-slurmd.stdout 2>&1 &
  for _ in $(seq 1 60); do
    if scontrol ping 2>/dev/null | grep -q UP && sinfo -h -N -o '%T' 2>/dev/null | grep -Eq 'idle|mix|alloc'; then
      return 0
    fi
    sleep 0.25
  done
  cat /tmp/bc-slurmctld.stdout >&2 || true
  cat /tmp/bc-slurmd.stdout >&2 || true
  return 1
}

stop_slurm() {
  sudo pkill -TERM -x slurmd >/dev/null 2>&1 || true
  sudo pkill -TERM -x slurmctld >/dev/null 2>&1 || true
  for _ in $(seq 1 40); do
    if ! pgrep -x slurmctld >/dev/null && ! pgrep -x slurmd >/dev/null; then return 0; fi
    sleep 0.25
  done
  return 1
}

wait_state() {
  local job="$1" regex="$2" timeout="${3:-60}" state
  for _ in $(seq 1 $((timeout*4))); do
    state="$(scontrol show job -o "$job" 2>/dev/null | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')"
    [[ "$state" =~ $regex ]] && { printf '%s\n' "$state"; return 0; }
    sleep 0.25
  done
  scontrol show job "$job" >&2 || true
  return 1
}

wait_file() {
  local path="$1" timeout="${2:-30}"
  for _ in $(seq 1 $((timeout*4))); do [[ -s "$path" ]] && return 0; sleep 0.25; done
  return 1
}

write_slurm_conf proctrack/linuxproc task/none ""
start_slurm || die "base controller/node did not become ready"
scontrol show config >/tmp/bc-scontrol-config.txt
scontrol show lic >/tmp/bc-licenses.txt
squeue --json >/tmp/bc-squeue.json
jq -e '.jobs | type == "array"' /tmp/bc-squeue.json >/dev/null
for expected in 'PriorityType.*priority/basic' 'SchedulerParameters.*bf_licenses.*bf_interval=2.*sched_interval=2.*requeue_delay=2' 'Licenses.*host_activity:2,build_slot:1'; do
  grep -Eq "$expected" /tmp/bc-scontrol-config.txt || die "active config missing: $expected"
done
log "minimal no-db scheduler live"

SBATCH=(--parsable --output=/tmp/bc-%j.out --error=/tmp/bc-%j.err)

BASIC="$(sbatch "${SBATCH[@]}" -p bc-build --licenses=build_slot:1,host_activity:1 -t 00:01:00 --wrap='true')"
wait_state "$BASIC" COMPLETED 20 >/dev/null || die "no-account basic job failed"
log "no-account job completed"

printf -v BC_WRAP 'cd %q && PYTHONPATH=tools python tools/lab/run-campaign-durability/real_bigcherry_process_smoke.py > /tmp/bc-bigcherry-slurm.json' "$REPO"
BC_JOB="$(sbatch "${SBATCH[@]}" -p bc-build --licenses=build_slot:1,host_activity:1 -t 00:03:00 --wrap="$BC_WRAP")"
wait_state "$BC_JOB" COMPLETED 60 >/dev/null || die "BigCherry Slurm job failed"
jq -e '.ok == true and .scope == "real-bigcherry-modules-cli-and-child-processes"' /tmp/bc-bigcherry-slurm.json >/dev/null
log "real BigCherry current-branch harness ran under Slurm job=$BC_JOB"

HOLD="$(sbatch "${SBATCH[@]}" --hold -p bc-build --licenses=build_slot:1,host_activity:1 -t 00:01:00 --wrap='sleep 1')"
wait_state "$HOLD" PENDING 10 >/dev/null || die "held job not pending"
scontrol show job -o "$HOLD" | grep -q 'Reason=JobHeldUser' || die "hold reason missing"
scontrol release "$HOLD"
wait_state "$HOLD" COMPLETED 20 >/dev/null || die "released job failed"
log "hold/release ok"

# Starvation/progress: build1 really owns one host_activity token + build_slot.
# Measurement needs both host_activity tokens; build2 is lower tier and cannot
# take them when build1 releases.
rm -f /tmp/bc-order /tmp/bc-build1-started
B1="$(sbatch "${SBATCH[@]}" -p bc-build --licenses=build_slot:1,host_activity:1 -t 00:01:00 --wrap='echo build1 >> /tmp/bc-order; echo started > /tmp/bc-build1-started; sleep 6')"
wait_file /tmp/bc-build1-started 20 || die "build1 never actually started"
wait_state "$B1" RUNNING 5 >/dev/null || die "build1 not RUNNING after marker"
M="$(sbatch "${SBATCH[@]}" -p bc-measure --licenses=host_activity:2 -t 00:01:00 --wrap='echo measure >> /tmp/bc-order; sleep 2')"
B2="$(sbatch "${SBATCH[@]}" -p bc-build --licenses=build_slot:1,host_activity:1 -t 00:01:00 --wrap='echo build2 >> /tmp/bc-order; sleep 1')"
sleep 0.5
M_REASON="$(squeue -h -j "$M" -o '%R' || true)"
[[ "$M_REASON" =~ Licenses|Priority|Resources ]] || die "measurement pending reason unexpected: $M_REASON"
wait_state "$M" COMPLETED 30 >/dev/null || die "measurement did not progress"
wait_state "$B2" COMPLETED 30 >/dev/null || die "build2 did not complete"
mapfile -t ORDER </tmp/bc-order
[[ "${ORDER[*]}" == 'build1 measure build2' ]] || die "starvation/order failure: ${ORDER[*]}"
log "license-aware priority progress ok: ${ORDER[*]} reason=$M_REASON"

rm -f /tmp/bc-prepare /tmp/bc-execute
P="$(sbatch "${SBATCH[@]}" -p bc-build --licenses=build_slot:1,host_activity:1 -t 00:01:00 --wrap='echo prepared > /tmp/bc-prepare')"
E="$(sbatch "${SBATCH[@]}" -p bc-measure --licenses=host_activity:2 --dependency=afterok:${P} -t 00:01:00 --wrap='test -s /tmp/bc-prepare && echo executed > /tmp/bc-execute')"
wait_state "$E" COMPLETED 30 >/dev/null || die "afterok execute failed"
test -s /tmp/bc-execute || die "dependency output missing"
log "afterok dependency ok"

# RequeueExit: force exactly one exit75 with a marker. Separately prove Slurm's
# restart counter is injected on the second run. requeue_delay=2 keeps CI and
# operational transient recovery bounded instead of the default 120s.
cat >/tmp/bc-requeue.sh <<'EOS'
#!/usr/bin/env bash
set -eu
printf '%s\n' "${SLURM_RESTART_COUNT:-0}" >> /tmp/bc-restarts
if [[ ! -e /tmp/bc-requeue-once ]]; then
  : > /tmp/bc-requeue-once
  exit 75
fi
exit 0
EOS
chmod +x /tmp/bc-requeue.sh
rm -f /tmp/bc-restarts /tmp/bc-requeue-once
R="$(sbatch "${SBATCH[@]}" -p bc-build --licenses=build_slot:1,host_activity:1 -t 00:01:00 /tmp/bc-requeue.sh)"
for _ in $(seq 1 80); do
  info="$(scontrol show job -o "$R" 2>/dev/null || true)"
  [[ "$info" == *'Restarts=1'* ]] && break
  sleep 0.25
done
scontrol show job -o "$R" | grep -q 'Restarts=1' || die "automatic requeue did not register Restarts=1"
wait_state "$R" COMPLETED 30 >/dev/null || die "requeued job did not complete"
mapfile -t RESTARTS </tmp/bc-restarts
[[ "${RESTARTS[*]}" == '0 1' ]] || die "SLURM_RESTART_COUNT sequence=${RESTARTS[*]}"
log "RequeueExit=75 same job + restart counter ok: ${RESTARTS[*]}"

C="$(sbatch "${SBATCH[@]}" -p bc-build --licenses=build_slot:1,host_activity:1 -t 00:02:00 --wrap='sleep 60')"
wait_state "$C" RUNNING 20 >/dev/null || die "cancel fixture never ran"
scancel "$C"
wait_state "$C" CANCELLED 20 >/dev/null || die "cancel state missing"
log "cancel ok"

# Controller restart retains a held queued job and a running job.
Q="$(sbatch "${SBATCH[@]}" --hold -p bc-build --licenses=build_slot:1,host_activity:1 -t 00:01:00 --wrap='true')"
LONG="$(sbatch "${SBATCH[@]}" -p bc-build --licenses=build_slot:1,host_activity:1 -t 00:01:00 --wrap='sleep 8')"
wait_state "$LONG" RUNNING 20 >/dev/null || die "restart running fixture never ran"
sudo pkill -TERM -x slurmctld
for _ in $(seq 1 40); do ! pgrep -x slurmctld >/dev/null && break; sleep 0.25; done
sudo slurmctld -Dvv >>/tmp/bc-slurmctld.stdout 2>&1 &
for _ in $(seq 1 40); do scontrol ping 2>/dev/null | grep -q UP && break; sleep 0.25; done
scontrol ping | grep -q UP || die "controller restart failed"
wait_state "$Q" PENDING 10 >/dev/null || die "queued job lost across controller restart"
scontrol release "$Q"
wait_state "$Q" COMPLETED 20 >/dev/null || die "queued job failed after controller restart"
wait_state "$LONG" COMPLETED 20 >/dev/null || die "running job lost across controller restart"
log "controller state recovery ok"

for _ in $(seq 1 40); do [[ -s /tmp/bc-slurm-jobcomp.log ]] && break; sleep 0.25; done
test -s /tmp/bc-slurm-jobcomp.log || die "jobcomp/filetxt empty"
grep -q "JobId=${BC_JOB}" /tmp/bc-slurm-jobcomp.log || die "BigCherry Slurm job absent from jobcomp"
log "jobcomp/filetxt completion history ok"

# Restart with the production process/accounting cgroup plugins but no device
# filtering. This proves the non-GPU fallback stack on Noble; only
# ConstrainDevices=yes + AMD /dev/kfd/renderD behavior stays Brutus-only.
stop_slurm || die "failed to stop base Slurm"
sudo tee /etc/slurm/cgroup.conf >/dev/null <<'EOF'
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainDevices=no
ConstrainRAMSpace=no
ConstrainSwapSpace=no
EOF
write_slurm_conf proctrack/cgroup task/cgroup,task/affinity 'JobAcctGatherType=jobacct_gather/cgroup
JobAcctGatherFrequency=30'
start_slurm || die "cgroup plugin stack failed to start"
CG="$(sbatch "${SBATCH[@]}" -p bc-build --licenses=build_slot:1,host_activity:1 -t 00:01:00 --wrap='cat /proc/self/cgroup > /tmp/bc-cgroup-membership')"
wait_state "$CG" COMPLETED 20 >/dev/null || die "cgroup-mode job failed"
test -s /tmp/bc-cgroup-membership || die "cgroup membership missing"
grep -Eq 'slurm|job_' /tmp/bc-cgroup-membership || {
  cat /tmp/bc-cgroup-membership >&2
  die "job not visibly placed in Slurm cgroup"
}
log "production cgroup plugins work with ConstrainDevices=no fallback"

jq -n \
  --arg slurm_version "$SLURM_VERSION" \
  --arg bigcherry_job "$BC_JOB" \
  --arg pending_reason "$M_REASON" \
  --arg schedule_order "${ORDER[*]}" \
  --arg restart_counts "${RESTARTS[*]}" \
  '{ok:true,scope:"real-noble-slurm-v2-no-gpu",slurm_version:$slurm_version,bigcherry_slurm_job:$bigcherry_job,accounting:"none+jobcomp/filetxt",priority:"basic+PriorityTier",scheduler:"backfill+bf_licenses",schedule_order:$schedule_order,pending_reason:$pending_reason,restart_counts:$restart_counts,cgroup_fallback:true}'
