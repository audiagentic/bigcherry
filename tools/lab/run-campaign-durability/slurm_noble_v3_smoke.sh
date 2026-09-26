#!/usr/bin/env bash
set -euo pipefail

# Iteration wrapper over v2: Slurm without an association DB can report
# transient Reason=InvalidAccount while jobs remain runnable and later execute.
# Pending-reason text is therefore diagnostic only; state/order/resource
# behavior is the acceptance contract.
src="$(dirname "$0")/slurm_noble_v2_smoke.sh"
tmp="$(mktemp /tmp/bc-slurm-v3.XXXXXX.sh)"
cp "$src" "$tmp"
python3 - "$tmp" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
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
path.write_text(text.replace(old, new))
PY
chmod +x "$tmp"
exec bash "$tmp"
