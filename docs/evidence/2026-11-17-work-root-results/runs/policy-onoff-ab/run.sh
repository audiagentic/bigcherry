#!/bin/bash
# Live placement policy ON vs OFF on the current build, flashnext-4gpu defaults (heat period 1 s), ABBA = on1 off1 off2 on2.
# ON = the calibrated store present (the manifest must carry live_policy); OFF = store moved aside (the manifest must not). The state is
# checked from each arm's manifest and an arm whose state differs from its label is marked INVALID. Both arms get the same warm-up traffic.
D=<work-root>/runs/policy-onoff-ab; P=<repo>
echo $$ > $D/run.pid; rm -f $D/done $D/status; LEASED=0; STARTED=0
cd $P; set --; source tools/env/bigcherry-env.sh >/dev/null 2>&1; WORK=$BC_WORK_ROOT
L="bash $P/tools/lab/radiance/launch-flashnext-best.sh"; STORE=$WORK/live-policy-calibration.toml; PORT=18450
NB=<work-root>/builds/f17f21fda44979e9abe84f13e96b8896/b43f943466729a2eb484e082dc45069f
finish() { [ $STARTED = 1 ] && $L stop > $D/final-stop.log 2>&1; [ -f $STORE.aboff ] && mv $STORE.aboff $STORE; [ $LEASED = 1 ] && agent-lease release gpu; touch $D/done; }
trap finish EXIT; trap 'exit 130' INT TERM HUP
until agent-lease acquire gpu --ttl 4h --purpose "live placement policy on/off A/B (flashnext-4gpu, current build)"; do sleep 60; done; LEASED=1
cp $STORE $D/store.toml
SB="env PYTHONPATH=$P/tools timeout 2400 python3 -m bigcherry stream-bench --url http://127.0.0.1:$PORT --temperature 0"
agg() { tail -n1 $D/$1.log | grep -oE '[0-9.]+ tok/s aggregate' | grep -oE '^[0-9.]+'; }
stp() { tail -n1 $D/$1.log | grep -oE 'step +[0-9.]+ ms' | grep -oE '[0-9.]+'; }
eng() { python3 -c "
import json,statistics as s,sys
d=json.load(open(sys.argv[1])); print('%.0f'%s.mean(r['engine']['decode_tps_engine'] for r in d['rounds']))" $D/$1.json 2>/dev/null; }
WARM_S=0
arm() { local a=$1 want=$2 t=$1
  if [ $want = off ]; then mv $STORE $STORE.aboff; fi
  STARTED=1; $L start --build $NB --port $PORT > $D/$t.launch.log 2>&1 || { echo "$t start failed" >> $D/status; [ -f $STORE.aboff ] && mv $STORE.aboff $STORE; $L stop >> $D/$t.stop.log 2>&1; STARTED=0; return; }
  M=$(ls -t $WORK/flashnext/manifest-1*.json | head -1)
  have=$(python3 -c "import json,sys;p=json.load(open(sys.argv[1]))['spec']['handoff']['placement'];print('on' if 'live_policy' in p else 'off')" $M)
  if [ $want = on ] && [ "$WARM_S" = 0 ]; then WARM_S=$(python3 -c "
import json,sys,math
p=json.load(open(sys.argv[1]))['spec']['handoff']['placement']['live_policy']
print(math.ceil(((p['min_samples']+1)*p['cadence_ns']+p['min_residency_ns'])/1e9))" $M); echo "warm_s derived (min_samples+1)*cadence + min_residency = $WARM_S" >> $D/status; fi
  echo "$t label=$want manifest=$have $([ $want = $have ] && echo VALID || echo INVALID)" >> $D/status
  local t0=$SECONDS i=0; while [ $((SECONDS-t0)) -lt $WARM_S ]; do i=$((i+1)); $SB --streams 1 --prompt-tokens 27000 --max-tokens 8 --reps 1 --seed $((900+i)) --output $D/$t.warm$i.json > $D/$t.warm$i.log 2>&1; python3 $D/doc_prefill.py $PORT 44000 $((20+i)) > $D/$t.warmdoc$i.log 2>&1; done
  for i in 1 2 3 4 5 6 7 8; do $SB --streams 1 --prompt-tokens 27000 --max-tokens 8 --reps 1 --seed $((500+i)) --output $D/$t.cold$i.json > $D/$t.cold$i.log 2>&1; done
  for k in 1 2 3 4 5 6 7 8 9 10; do python3 $D/doc_prefill.py $PORT 44000 $k > $D/$t.doc$k.log 2>&1; done
  for n in 1 6; do $SB --streams $n --prompt-tokens 900 --max-tokens 256 --reps 2 --seed 7 --output $D/$t.dec$n.json > $D/$t.dec$n.log 2>&1; done
  $SB --streams 6 --prompt-tokens 8192 --max-tokens 32 --reps 2 --seed 11 --output $D/$t.sess6.json > $D/$t.sess6.log 2>&1
  echo "$t cold-ttft $(for i in 1 2 3 4 5 6 7 8; do tail -n1 $D/$t.cold$i.log | grep -oE 'ttft max +[0-9.]+' | grep -oE '[0-9.]+$'; done | tr '\n' ' ')| doc $(for k in 1 2 3 4 5 6 7 8 9 10; do grep -oE '[0-9]+ tok/s' $D/$t.doc$k.log | tail -1 | grep -oE '^[0-9]+'; done | tr '\n' ' ')| dec1 agg $(agg $t.dec1) eng $(eng $t.dec1) ms $(stp $t.dec1) | dec6 agg $(agg $t.dec6) eng $(eng $t.dec6) ms $(stp $t.dec6)" >> $D/status
  E=$(ls -t $WORK/flashnext/engine-*.log | head -1); cp $E $D/$t.engine.log; echo "$t placement cycles: $(grep -c 'placement cycle .* reopened epoch' $D/$t.engine.log)" >> $D/status
  $L stop > $D/$t.stop.log 2>&1; STARTED=0; [ -f $STORE.aboff ] && mv $STORE.aboff $STORE; }
arm on1 on; arm off1 off; arm off2 off; arm on2 on
