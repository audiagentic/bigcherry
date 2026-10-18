#!/bin/bash
# RAD06: promotion evidence for libr3's settings. Builds the branch once, then runs each single-setting A/B (the
# setting is the only variable between the two arms; see core/components.py _evidence_problem) in turn and stages each
# capture's run.json for docs/evidence/RAD06/. Waits for an idle host first: two server runs on one GPU set corrupt each
# other. `bigcherry patch-lint` then checks the staged run.json against the setting's activation marker.
# Usage: helper-evidence.sh <branch> <name>=<config.json> [<name>=<config.json> ...]
#   e.g. helper-evidence.sh origin/main R3_MOE_FP16=tools/lab/radiance/helper-fp16-ab.json R3_MOE_AHALF=tools/lab/radiance/helper-ahalf-ab.json
set -u
BR=$1; shift
W=/mnt/data/bigcherry-work
STAGE=$W/evidence-RAD06
mkdir -p $STAGE
# Another evidence batch, or any server run, holds the GPUs: wait for both (a batch has gaps between its A/Bs).
while pgrep -f '[h]elper-evidence.sh' | grep -vx $$ > /dev/null || ps -eo cmd | grep -E '[a]b-benchmark|[r]adiance/bin|[r]3_moe_helper' > /dev/null; do sleep 20; done
cd $W/wt-pr8 && git fetch -q origin && git checkout -q --detach $BR && git log --oneline -1
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH BIGCHERRY_ENVIRONMENT=/mnt/vault/development/projects/bigcherry/workspaces/main/config/environment.local.toml
PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance --experiment moe-remote-handoff > $W/evidence-build.log 2>&1
grep -E 'ok build|FAILED' $W/evidence-build.log | tail -1; grep -E ' error:|r3_rewrite.py failed' $W/evidence-build.log | sort -u | head
B=$(ls -td /mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/*/*/ | head -1)
export RAD_BIN=$B/bin/radiance RAD_HOME=$B/radiance_home R9700_BDF=0000:09:00.0
for spec in "$@"; do
  name=${spec%%=*}; cfg=${spec#*=}
  out=$W/runs/rad06-evidence-$name-$(date +%m%d%H%M%S)
  echo "== $name  ($cfg)"
  PYTHONPATH=tools timeout 5400 python3 -m bigcherry ab-benchmark --server-config $cfg --output $out --pairs 2 --settle-seconds 20 2>&1 | tail -2
  cp $out/run.json $STAGE/$name.run.json && echo "staged $STAGE/$name.run.json"
  for d in $out/pair-*; do echo "  $(basename $d) $(grep -oE '"(pp4096|tg512|tg2048)_tps": *[0-9.]+' $d/cell.json | tr '\n' ' ')"; done
  python3 - $STAGE/$name.run.json $name <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print('  activation:', json.dumps(r.get('setting_activation', {})))
print('  failed cells:', [x.get('mode') for x in r.get('runs', []) if x.get('returncode') != 0])
PY
done
echo EVIDENCE_DONE
