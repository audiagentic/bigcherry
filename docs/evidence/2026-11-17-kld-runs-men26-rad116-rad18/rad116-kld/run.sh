#!/bin/bash
# radiance v1.16.0 bump (PR 555): KLD of Flash-Next helper offload vs a freshly recorded engine-local reference on the bump build, same protocol and limits as MEN26 (kld.json from helper-kld.json).
# the two XTX), on origin/main AFTER #542 merges (rad0002's implementation changes there, and the evidence must bind to
# the implementation being promoted). Config: kld.json (from tools/lab/radiance/helper-kld.json, MEN11).
# no set -u: tools/env/bigcherry-env.sh reads unset variables
D=<work-root>/runs/rad116-kld
P=<dev-root>/projects/bigcherry/workspaces/main
VAR=<dev-root>/tmp/bigcherry/rad116-kld
rm -f $D/done $D/status; echo $$ > $D/run.pid
LEASED=0
finish() { [ $LEASED = 1 ] && agent-lease release gpu; [ -d $VAR/main ] && git -C $P worktree remove --force $VAR/main; touch $D/done; }
trap finish EXIT
trap 'exit 130' INT TERM HUP
fail() { echo "$1" > $D/status; exit 1; }
while [ ! -e <work-root>/runs/rad116-ab1/done ]; do sleep 60; done   # after the speed A/B (no build during a measurement)
git -C $P fetch -q origin || fail fetch-failed
SHA=b1bd5f005568ea1425f307d874e587bea9a5056e; echo "$SHA" > $D/main.sha
mkdir -p $VAR; [ -d $VAR/main ] && git -C $P worktree remove --force $VAR/main
git -C $P worktree add -q --detach $VAR/main $SHA || fail worktree-failed
(cd $VAR/main && set -- && source tools/env/bigcherry-env.sh >/dev/null 2>&1 && source tools/lab/radiance/lab-build.sh \
   && rad_build "$D/main.build.log") > $D/main.build.dir || fail build-failed
B=$(tail -1 $D/main.build.dir); [ -x $B/bin/radiance ] || fail "no radiance in $B"
(cd $VAR/main && PYTHONPATH=tools python3 -c "
from bigcherry.patch import evidence; from pathlib import Path
print(evidence.patch_validation_subject_digest(Path('engines/radiance/patches/rad0002_moe_remote_handoff/patch.py')))") > $D/rad0002.digest
echo builds-ready > $D/status
cd $VAR/main; set --; source tools/env/bigcherry-env.sh >/dev/null 2>&1
export RAD_BIN=$B/bin/radiance RAD_HOME=$B/radiance_home R9700_BDF=0000:09:00.0 KLD_REF_DIR=$D/reference
until agent-lease acquire gpu --ttl 6h --purpose "radiance v1.16.0 bump KLD helper offload vs engine-local reference"; do sleep 60; done
LEASED=1
PYTHONPATH=tools timeout 14400 python3 -m bigcherry.campaign.benchmark --server-config $D/kld.json --output $D/capture --pairs 1 \
  > $D/capture.log 2>&1 || echo "capture exit $?" >> $D/status
PYTHONPATH=tools timeout 7200 python3 -m bigcherry.campaign.benchmark --server-config $D/kld-isolate.json --output $D/capture-isolate --pairs 1 \
  > $D/capture-isolate.log 2>&1 || echo "isolate exit $?" >> $D/status   # two-XTX arm (v1.14: p99.9 +0.41, passes)
agent-lease release gpu; LEASED=0
grep -qE "capture exit|isolate exit" $D/status || echo ok > $D/status
