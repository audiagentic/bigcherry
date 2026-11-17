#!/bin/bash
# RHP06 mid-scoring forced-churn KL on flashnext-4gpu, main after #733 (pure manifest).
D=<work-root>/runs/rhp06-churn-paced; P=<repo>
echo $$ > $D/run.pid; rm -f $D/done $D/status; LEASED=0
finish() { [ $LEASED = 1 ] && agent-lease release gpu; touch $D/done; }
trap finish EXIT; trap 'exit 130' INT TERM HUP

git -C $P pull -q --ff-only origin main || { echo "primary pull failed" >> $D/status; exit 1; }
cd $P; set --; source tools/env/bigcherry-env.sh >/dev/null 2>&1; WORK=$BC_WORK_ROOT
# Build only while holding the lease: a lane build beside a running engine (pinned host memory) OOM-killed one.
until agent-lease acquire gpu --ttl 2h --purpose "RHP06 mid-scoring forced-churn KL (main after #733)"; do sleep 60; done; LEASED=1
(nice -n 15 env PYTHONPATH=tools python3 -m bigcherry build --lane radiance:stock:linux-radiance > $D/build.log 2>&1)
NB=$(grep -oE 'ok build_plan_id=[0-9a-f]+' $D/build.log | tail -1 | cut -d= -f2); NB=$(ls -d $WORK/builds/*/$NB 2>/dev/null | head -1)
[ -x "$NB/bin/radiance" ] || { echo "no build ($NB)" >> $D/status; exit 1; }
echo "build $NB ($(git -C $P rev-parse --short HEAD))" >> $D/status
export RAD_BIN=$NB/bin/radiance RAD_HOME=$NB/radiance_home R9700_BDF=0000:09:00.0 KLD_REF_DIR=<work-root>/runs/rad116-kld/reference
PYTHONPATH=tools timeout 5400 python3 -m bigcherry.campaign.benchmark --server-config $D/kld.json --output $D/out --pairs 1 > $D/kld.log 2>&1
echo "kld exit=$?" >> $D/status
