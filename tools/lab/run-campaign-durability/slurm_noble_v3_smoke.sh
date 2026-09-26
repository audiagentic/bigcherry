#!/usr/bin/env bash
set -euo pipefail

# Iteration wrapper over v2. Noble 23.11 facts proven/checked here:
# - no association DB is required for jobs to execute, but pending Reason can
#   transiently report InvalidAccount; state/resource/order are authoritative.
# - automatic requeue delay is controlled by AuthInfo=cred_expire on Noble;
#   SchedulerParameters=requeue_delay is not available in 23.11.
src="$(dirname "$0")/slurm_noble_v2_smoke.sh"
tmp="$(mktemp /tmp/bc-slurm-v3.XXXXXX.sh)"
cp "$src" "$tmp"
python3 - "$tmp" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
text = text.replace(
    "AuthType=auth/munge\nCredType=cred/munge",
    "AuthType=auth/munge\nAuthInfo=cred_expire=30\nCredType=cred/munge",
)
text = text.replace(
    "SchedulerParameters=bf_licenses,bf_interval=2,sched_interval=2,requeue_delay=2",
    "SchedulerParameters=bf_licenses,bf_interval=2,sched_interval=2",
)
text = text.replace(
    "'SchedulerParameters.*bf_licenses.*bf_interval=2.*sched_interval=2.*requeue_delay=2'",
    "'SchedulerParameters.*bf_licenses.*bf_interval=2.*sched_interval=2'",
)
old = '''M_REASON="$(squeue -h -j "$M" -o '%R' || true)"
[[ "$M_REASON" =~ Licenses|Priority|Resources ]] || die "measurement pending reason unexpected: $M_REASON"
wait_state "$M" COMPLETED 30 >/dev/null || die "measurement did not progress"'''
new = '''M_REASON="$(squeue -h -j "$M" -o '%R' || true)"
wait_state "$M" PENDING 5 >/dev/null || die "measurement was not pending while build1 held host_activity"
wait_state "$B2" PENDING 5 >/dev/null || die "build2 unexpectedly started while build1 held build_slot"
log "pending reasons are diagnostic only: measure=$M_REASON build2=$(squeue -h -j "$B2" -o '%R' || true)"
wait_state "$M" COMPLETED 30 >/dev/null || die "measurement did not progress"'''
if old not in text:
    raise SystemExit("expected v2 pending-reason block not found")
text = text.replace(old, new)
text = text.replace(
    "# operational transient recovery bounded instead of the default 120s.",
    "# operational transient recovery uses AuthInfo=cred_expire=30 instead of the default 120s.",
)
text = text.replace(
    'wait_state "$R" COMPLETED 30 >/dev/null || die "requeued job did not complete"',
    'wait_state "$R" COMPLETED 50 >/dev/null || die "requeued job did not complete"',
)
path.write_text(text)
PY
chmod +x "$tmp"
exec bash "$tmp"
