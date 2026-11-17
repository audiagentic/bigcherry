#!/bin/bash
D=<work-root>/runs/host-pool
until [ -f <work-root>/runs/kv-tier/done ]; do sleep 20; done
cd <repo>; set --; source tools/env/bigcherry-env.sh >/dev/null 2>&1
agent-lease acquire gpu --ttl 3h --purpose "host pool sizing: 9 GiB + KV host tier from freed RAM" || exit 1
export W4H2=1 GEMV_LARGE=32
L=<work-root>/runs/w4h2/lab/launch-flashnext-best.sh
arm() { n=$1; shift
  bash $L start --build <work-root>/builds/3a4f76bf9f7cf4039f9e52607bb02d59/762ce878d98f02c217a27b336735be07 --port 18410 --record-heat off "$@" > $D/$n.launch.log 2>&1
  grep -o "<work-root>/flashnext/engine-[0-9-]*.log" $D/$n.launch.log | head -1 > $D/$n.enginelog
  (cd $D && timeout 1800 python3 ctxsweep2.py 43008 > $n.probe-fresh.log 2>&1)
  (cd $D && timeout 1800 python3 multi.py > $n.multi.log 2>&1)
  (cd $D && timeout 1800 python3 ctxsweep2.py 43008 > $n.probe.log 2>&1)
  PYTHONPATH=tools timeout 1800 python3 -m bigcherry stream-bench --url http://127.0.0.1:18410 --streams 1,6 --prompt-tokens 512 --max-tokens 512 --reps 2 --temperature 0 --output $D/$n.decode.json > $D/$n.decode.log 2>&1
  free -g > $D/$n.free.txt
  bash $L stop > $D/$n.stop.log 2>&1; }
arm pool9 --host-pool-mib 9216
arm pool9tier --host-pool-mib 9216 --engine-arg --prefix-cache-mib --engine-arg 12288
agent-lease release gpu
touch $D/done
