#!/usr/bin/env bash
set -euo pipefail

# Real Slurm/MUNGE/slurmdbd smoke for an Ubuntu 24.04 disposable host.
# Validates real Noble 23.11 service/scheduler semantics without claiming AMD
# GRES/cgroup/ROCm hardware qualification.

log() { printf '[rcd-slurm-smoke] %s\n' "$*" >&2; }
die() { log "FAIL: $*"; exit 1; }

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  slurm-wlm slurmdbd munge mariadb-server jq

SLURM_VERSION="$(dpkg-query -W -f='${Version}' slurm-wlm | cut -d- -f1)"
log "installed Slurm ${SLURM_VERSION}"
case "$SLURM_VERSION" in
  23.11.*) ;;
  *) die "expected Ubuntu Noble Slurm 23.11.x, got ${SLURM_VERSION}" ;;
esac

HOST="$(hostname -s)"
CPUS="$(nproc)"
REALMEM="$(awk '/MemTotal:/ {m=int($2/1024)-1024; if (m < 1024) m=int($2/1024*0.8); print m}' /proc/meminfo)"
USER_NAME="$(id -un)"

cleanup() {
  set +e
  sudo pkill -TERM -x slurmd >/dev/null 2>&1 || true
  sudo pkill -TERM -x slurmctld >/dev/null 2>&1 || true
  sudo pkill -TERM -x slurmdbd >/dev/null 2>&1 || true
  sudo pkill -TERM -x munged >/dev/null 2>&1 || true
  sudo pkill -TERM -x mariadbd >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

# MUNGE
sudo install -d -o munge -g munge -m 0755 /run/munge /var/log/munge
if [[ ! -s /etc/munge/munge.key ]]; then
  sudo sh -c 'umask 077; dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 status=none'
fi
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 0400 /etc/munge/munge.key
sudo -u munge /usr/sbin/munged --force
munge -n | unmunge >/dev/null
log "MUNGE round-trip ok"

# Minimal local accounting database. GitHub runners have no usable systemd, so
# start MariaDB manually, bound only to loopback. Brutus uses the packaged
# systemd service with the same loopback-only policy.
sudo rm -rf /tmp/bc-mysql-data
sudo install -d -o mysql -g mysql -m 0750 /tmp/bc-mysql-data /run/mysqld
sudo mariadb-install-db \
  --user=mysql \
  --datadir=/tmp/bc-mysql-data \
  --auth-root-authentication-method=normal \
  >/tmp/bc-mariadb-init.log 2>&1
sudo -u mysql mariadbd \
  --datadir=/tmp/bc-mysql-data \
  --socket=/run/mysqld/mysqld.sock \
  --pid-file=/run/mysqld/mysqld.pid \
  --bind-address=127.0.0.1 \
  --port=3306 \
  --skip-name-resolve \
  --log-error=/tmp/bc-mariadb.log \
  >/tmp/bc-mariadb.stdout 2>&1 &
for _ in $(seq 1 40); do
  mariadb-admin --protocol=socket --socket=/run/mysqld/mysqld.sock ping >/dev/null 2>&1 && break
  sleep 0.5
done
mariadb-admin --protocol=socket --socket=/run/mysqld/mysqld.sock ping >/dev/null || {
  cat /tmp/bc-mariadb.log >&2 || true
  die "MariaDB did not start"
}
sudo mariadb --protocol=socket --socket=/run/mysqld/mysqld.sock <<'SQL'
CREATE DATABASE IF NOT EXISTS slurm_acct_db;
CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY 'slurm-ci';
CREATE USER IF NOT EXISTS 'slurm'@'127.0.0.1' IDENTIFIED BY 'slurm-ci';
GRANT ALL PRIVILEGES ON slurm_acct_db.* TO 'slurm'@'localhost';
GRANT ALL PRIVILEGES ON slurm_acct_db.* TO 'slurm'@'127.0.0.1';
FLUSH PRIVILEGES;
SQL
log "loopback MariaDB accounting database ok"

sudo install -d -o slurm -g slurm -m 0755 /tmp/bc-slurm-state /tmp/bc-slurmdbd-state
sudo install -d -o root -g root -m 0755 /tmp/bc-slurmd-spool
sudo install -d -o root -g root -m 0755 /etc/slurm

sudo tee /etc/slurm/slurmdbd.conf >/dev/null <<EOF
AuthType=auth/munge
DbdHost=${HOST}
DbdAddr=127.0.0.1
DbdPort=6819
SlurmUser=slurm
DebugLevel=info
LogFile=/tmp/bc-slurmdbd.log
PidFile=/tmp/bc-slurmdbd.pid
StorageType=accounting_storage/mysql
StorageHost=127.0.0.1
StoragePort=3306
StorageLoc=slurm_acct_db
StorageUser=slurm
StoragePass=slurm-ci
EOF
sudo chown slurm:slurm /etc/slurm/slurmdbd.conf
sudo chmod 0600 /etc/slurm/slurmdbd.conf

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
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=127.0.0.1
AccountingStoragePort=6819
AccountingStorageEnforce=associations
JobCompType=jobcomp/filetxt
JobCompLoc=/tmp/bc-slurm-jobcomp.log
RequeueExit=75
MinJobAge=300
NodeName=${HOST} NodeAddr=127.0.0.1 CPUs=${CPUS} RealMemory=${REALMEM} State=UNKNOWN
PartitionName=bc-build Nodes=${HOST} PriorityTier=10 Default=NO State=UP MaxTime=00:10:00
PartitionName=bc-measure Nodes=${HOST} PriorityTier=100 Default=YES State=UP MaxTime=00:10:00
EOF
export SLURM_CONF=/etc/slurm/slurm.conf

# slurmdbd must precede slurmctld. Register the cluster/account/user association
# before accepting jobs; this is required for reliable 23.11 scheduling.
sudo slurmdbd -Dvv >/tmp/bc-slurmdbd.stdout 2>&1 &
for _ in $(seq 1 40); do
  sacctmgr ping 2>/dev/null | grep -qi 'UP' && break
  sleep 0.5
done
sacctmgr ping | grep -qi 'UP' || {
  cat /tmp/bc-slurmdbd.stdout >&2 || true
  cat /tmp/bc-slurmdbd.log >&2 || true
  die "slurmdbd not UP"
}
sacctmgr -i add cluster bigcherry-ci >/dev/null
sacctmgr -i add account bigcherry Cluster=bigcherry-ci Description=BigCherry Organization=BigCherry >/dev/null
sacctmgr -i add user "$USER_NAME" Account=bigcherry DefaultAccount=bigcherry Cluster=bigcherry-ci >/dev/null
sacctmgr -i add user root Account=bigcherry DefaultAccount=bigcherry Cluster=bigcherry-ci >/dev/null
sacctmgr -nP show assoc cluster=bigcherry-ci account=bigcherry | grep -q '^bigcherry-ci|bigcherry|' || die "BigCherry association missing"
log "slurmdbd account/association ok"

sudo slurmctld -Dvv >/tmp/bc-slurmctld.stdout 2>&1 &
sleep 1
sudo slurmd -Dvv >/tmp/bc-slurmd.stdout 2>&1 &

for _ in $(seq 1 40); do
  if scontrol ping 2>/dev/null | grep -q 'UP'; then
    if sinfo -h -N -o '%T' 2>/dev/null | grep -Eq 'idle|mix|alloc'; then
      break
    fi
  fi
  sleep 0.5
done
scontrol ping | grep -q 'UP' || { cat /tmp/bc-slurmctld.stdout >&2; die "slurmctld not UP"; }
sinfo -h -N -o '%T' | grep -Eq 'idle|mix|alloc' || {
  cat /tmp/bc-slurmd.stdout >&2 || true
  scontrol show node >&2 || true
  die "slurmd did not register usable node"
}
sinfo -N -l
log "single-node Slurm service is live"

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

SBATCH_COMMON=(--parsable --account=bigcherry)

HOLD_JOB="$(sbatch "${SBATCH_COMMON[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='sleep 1')"
scontrol hold "$HOLD_JOB"
wait_state "$HOLD_JOB" 'PENDING' 10 >/dev/null
scontrol show job -o "$HOLD_JOB" | grep -q 'Reason=JobHeldUser' || die "hold reason not visible"
scontrol release "$HOLD_JOB"
wait_state "$HOLD_JOB" 'COMPLETED' 30 >/dev/null
log "hold/release contract ok"

rm -f /tmp/bc-order
BUILD1="$(sbatch "${SBATCH_COMMON[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='echo build1 >> /tmp/bc-order; sleep 4')"
wait_state "$BUILD1" 'RUNNING' 20 >/dev/null
MEASURE="$(sbatch "${SBATCH_COMMON[@]}" --partition=bc-measure --licenses=host_activity:2 --time=00:01:00 --wrap='echo measure >> /tmp/bc-order; sleep 2')"
BUILD2="$(sbatch "${SBATCH_COMMON[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='echo build2 >> /tmp/bc-order; sleep 1')"
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

rm -f /tmp/bc-prepare /tmp/bc-execute
PREP="$(sbatch "${SBATCH_COMMON[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 --wrap='echo prepared > /tmp/bc-prepare')"
EXEC="$(sbatch "${SBATCH_COMMON[@]}" --partition=bc-measure --licenses=host_activity:2 --dependency=afterok:${PREP} --time=00:01:00 --wrap='test -s /tmp/bc-prepare && echo executed > /tmp/bc-execute')"
wait_state "$EXEC" 'COMPLETED' 60 >/dev/null
test -s /tmp/bc-execute || die "afterok dependent did not execute"
log "dependency contract ok"

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
REQUEUE="$(sbatch "${SBATCH_COMMON[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:01:00 /tmp/bc-requeue.sh)"
wait_state "$REQUEUE" 'COMPLETED' 60 >/dev/null
mapfile -t RESTARTS </tmp/bc-restarts
printf 'restart-counts=%s\n' "${RESTARTS[*]}"
[[ "${RESTARTS[*]}" == "0 1" ]] || die "expected restart counts '0 1'"
log "RequeueExit/SLURM_RESTART_COUNT contract ok"

CANCEL="$(sbatch "${SBATCH_COMMON[@]}" --partition=bc-build --licenses=build_slot:1,host_activity:1 --time=00:02:00 --wrap='sleep 60')"
wait_state "$CANCEL" 'RUNNING' 20 >/dev/null
scancel "$CANCEL"
wait_state "$CANCEL" 'CANCELLED' 30 >/dev/null
log "cancel contract ok"

# Verify both lightweight completion log and slurmdbd accounting history.
for _ in $(seq 1 20); do
  [[ -s /tmp/bc-slurm-jobcomp.log ]] && break
  sleep 0.5
done
test -s /tmp/bc-slurm-jobcomp.log || die "jobcomp/filetxt stayed empty"
for _ in $(seq 1 20); do
  sacct -nX -j "$REQUEUE" -o JobIDRaw,State >/tmp/bc-sacct.txt 2>/dev/null || true
  grep -q "$REQUEUE" /tmp/bc-sacct.txt && break
  sleep 0.5
done
grep -q "$REQUEUE" /tmp/bc-sacct.txt || die "slurmdbd accounting record missing"
log "jobcomp + slurmdbd accounting contracts ok"

jq -n \
  --arg slurm_version "$SLURM_VERSION" \
  --arg host "$HOST" \
  --arg pending_reason "$PENDING_REASON" \
  --arg schedule_order "${ORDER[*]}" \
  --arg restart_counts "${RESTARTS[*]}" \
  '{ok:true, scope:"real-noble-slurm-services-no-gpu", slurm_version:$slurm_version, host:$host, accounting:"slurmdbd+mariadb-loopback", pending_reason:$pending_reason, schedule_order:$schedule_order, restart_counts:$restart_counts}'
