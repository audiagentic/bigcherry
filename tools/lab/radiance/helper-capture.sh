#!/bin/bash
# RAD06: record real helper requests for `r3_moe_helper --replay` / `--replay-sweep`. Builds the branch, runs ONE
# server cell of helper-ab.json with R3_MOE_HELPER_CAPTURE set, and leaves the captures (capture format 2: layer
# slot map, routes, the engine's own quantised E4M3 activation planes) in the capture directory. The server's
# load stage is paid here once; every later kernel experiment replays these files without a server.
# Usage: helper-capture.sh <branch> [capture-dir=/mnt/data/bigcherry-work/captures] [every=11] [max=24]
set -u
BR=$1; CAP=${2:-/mnt/data/bigcherry-work/captures}; EVERY=${3:-11}; MAX=${4:-24}
W=/mnt/data/bigcherry-work
cd $W/wt-pr8 && git fetch -q origin && git checkout -q --detach $BR && git log --oneline -1
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH BIGCHERRY_ENVIRONMENT=/mnt/vault/development/projects/bigcherry/workspaces/main/config/environment.local.toml
PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance --experiment moe-remote-handoff > $W/capture-build.log 2>&1
grep -E 'ok build|FAILED' $W/capture-build.log | tail -1; grep -E ' error:|r3_rewrite.py failed' $W/capture-build.log | sort -u | head
B=$(ls -td /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/*/*/ | head -1)
mkdir -p $CAP
python3 - tools/lab/radiance/helper-ab.json $W/helper-capture.json $CAP $EVERY $MAX <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
a = d["arms"][0]
a["name"] = "capture-a"
a["settings"]["R3_MOE_HELPER_ROWS"] = "0"
a.setdefault("environment", {}).update({"R3_MOE_HELPER_CAPTURE": sys.argv[3],
                                        "R3_MOE_HELPER_CAPTURE_EVERY": sys.argv[4],
                                        "R3_MOE_HELPER_CAPTURE_MAX": sys.argv[5]})
c = json.loads(json.dumps(a)); c["name"] = "capture-b"   # the harness wants two arms; both record
d["arms"] = [a, c]
json.dump(d, open(sys.argv[2], "w"), indent=1)
PY
export RAD_BIN=$B/bin/radiance RAD_HOME=$B/radiance_home R9700_BDF=0000:09:00.0
PYTHONPATH=tools timeout 5400 python3 -m bigcherry ab-benchmark --server-config $W/helper-capture.json \
  --output $W/runs/rad06-capture-$(date +%m%d%H%M) --pairs 2 --settle-seconds 10 2>&1 | tail -2
ls -S $CAP | head -8; echo "captured $(ls $CAP | wc -l)"
echo CAPTURE_DONE
