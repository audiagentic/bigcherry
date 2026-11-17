#!/bin/bash
# MEN26: two repeat captures of helper-off vs helper-on against the same recorded reference, to measure the run-to-run
# spread of KLD p99.9 (the only metric the first capture breached: +1.58 vs a 0.5 limit).
D=<work-root>/runs/men26-kld
VAR=<dev-root>/tmp/bigcherry/men26-kld
P=<dev-root>/projects/bigcherry/workspaces/main
rm -f $D/repeat.done; LEASED=0
finish() { [ $LEASED = 1 ] && agent-lease release gpu; touch $D/repeat.done; }
trap finish EXIT; trap 'exit 130' INT TERM HUP
B=$(tail -1 $D/main.build.dir)
[ -d $VAR/main ] || git -C $P worktree add -q --detach $VAR/main $(cat $D/main.sha)
cd $VAR/main; set --; source tools/env/bigcherry-env.sh >/dev/null 2>&1
export RAD_BIN=$B/bin/radiance RAD_HOME=$B/radiance_home R9700_BDF=0000:09:00.0 KLD_REF_DIR=$D/reference
until agent-lease acquire gpu --ttl 2h --purpose "MEN26: KLD repeat (tail-metric spread)"; do sleep 60; done
LEASED=1
for i in 2 3; do
  PYTHONPATH=tools timeout 7200 python3 -m bigcherry.campaign.benchmark --server-config $D/kld-repeat.json --output $D/capture$i --pairs 1 > $D/capture$i.log 2>&1
done
