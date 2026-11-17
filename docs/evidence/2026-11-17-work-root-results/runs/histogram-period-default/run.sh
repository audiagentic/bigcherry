#!/bin/bash
# Is the live policy effective on a DEFAULT start (histogram period 60 s from the schema default and the launcher), or inert?
D=<work-root>/runs/histogram-period-default; P=<repo>
echo $$ > $D/run.pid; rm -f $D/done $D/status; LEASED=0; STARTED=0
cd $P; set --; source tools/env/bigcherry-env.sh >/dev/null 2>&1; WORK=$BC_WORK_ROOT
L="bash $P/tools/lab/radiance/launch-flashnext-best.sh"; NB=<work-root>/builds/f17f21fda44979e9abe84f13e96b8896/08b5dd81b4874390bf0cbf489db0d7df
finish() { [ $STARTED = 1 ] && $L stop > $D/stop.log 2>&1; [ $LEASED = 1 ] && agent-lease release gpu; touch $D/done; }
trap finish EXIT; trap 'exit 130' INT TERM HUP
until agent-lease acquire gpu --ttl 1h --purpose "default-period policy check (flashnext-4gpu, no --record-period)"; do sleep 30; done; LEASED=1
STARTED=1; $L start --build $NB --port 18450 > $D/launch.log 2>&1 || { echo "start failed" >> $D/status; exit 1; }
M=$(ls -t $WORK/flashnext/manifest-1*.json | head -1)
python3 -c "import json,sys;m=json.load(open(sys.argv[1]))['spec']['handoff'];print('period_s',m['telemetry']['period_s'],'live_policy' in m['placement'], 'cadence_ns',m['placement'].get('live_policy',{}).get('cadence_ns'),'max_heat_age_ns',m['placement'].get('live_policy',{}).get('max_heat_age_ns'))" $M >> $D/status
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do PYTHONPATH=$P/tools timeout 900 python3 -m bigcherry stream-bench --url http://127.0.0.1:18450 --streams 1 --prompt-tokens 27000 --max-tokens 8 --reps 1 --temperature 0 --seed $((900+i)) --output $D/p$i.json > $D/p$i.log 2>&1; python3 <work-root>/runs/policy-live-5/doc_prefill.py 18450 44000 $((20+i)) > $D/d$i.log 2>&1; done
echo "traffic done" >> $D/status
$L stop > $D/stop.log 2>&1; STARTED=0
H=$(ls -t $WORK/flashnext/helper0-*.log | head -1); cp $H $D/helper0.log
echo "no_live_feed lines: $(grep -c no_live_feed $D/helper0.log); evaluated: $(grep -o 'evaluated=[0-9]*' $D/helper0.log | tail -1); committed: $(grep -o 'committed=[0-9]*' $D/helper0.log | tail -1); epochs: $(grep -c 'placement cycle' $WORK/flashnext/engine-*.log 2>/dev/null | tail -1)" >> $D/status
