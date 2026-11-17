#!/bin/bash
# MEN26: the repeat captures are bit-identical (the p99.9 gap is deterministic, not noise). Isolate the tail helper:
# helper-off vs helper-2xtx (two XTX helpers, no RDNA2 tail) vs helper-on (production three-helper tail layout).
D=<work-root>/runs/men26-kld
VAR=<dev-root>/tmp/bigcherry/men26-kld
P=<dev-root>/projects/bigcherry/workspaces/main
rm -f $D/isolate.done; LEASED=0
finish() { [ $LEASED = 1 ] && agent-lease release gpu; [ -d $VAR/main ] && git -C $P worktree remove --force $VAR/main; touch $D/isolate.done; }
trap finish EXIT; trap 'exit 130' INT TERM HUP
while [ ! -e $D/repeat.done ]; do sleep 30; done
B=$(tail -1 $D/main.build.dir)
[ -d $VAR/main ] || git -C $P worktree add -q --detach $VAR/main $(cat $D/main.sha)
cd $VAR/main; set --; source tools/env/bigcherry-env.sh >/dev/null 2>&1
export RAD_BIN=$B/bin/radiance RAD_HOME=$B/radiance_home R9700_BDF=0000:09:00.0 KLD_REF_DIR=$D/reference
until agent-lease acquire gpu --ttl 2h --purpose "MEN26: KLD isolate tail helper"; do sleep 60; done
LEASED=1
PYTHONPATH=tools timeout 7200 python3 -m bigcherry.campaign.benchmark --server-config $D/kld-isolate.json --output $D/capture-isolate --pairs 1 > $D/capture-isolate.log 2>&1
