#!/bin/bash
D=<work-root>/runs/pwr-stock; P=<repo>; cd $P; set --; source tools/env/bigcherry-env.sh >/dev/null 2>&1
B=<work-root>/builds/daf793f13be8d5762deecb73e3372641/a29e62ada0fe691746707b2b87c3e9f3
export RADIANCE_HOME=$B/radiance_home BC_SERVE_BINARY=$B/radiance_home/bin/bc-serve
trap 'bash tools/lab/radiance/launch-flashnext-best.sh stop > $D/docpf.stop.log 2>&1; echo done > $D/docpf.status' EXIT
bash tools/lab/radiance/launch-flashnext-best.sh start --build $B --port 18452 > $D/docpf.launch.log 2>&1 || exit 1
for k in 1 2 3 4 5 6; do python3 <work-root>/runs/policy-onoff-ab/doc_prefill.py 18452 44000 $k; done > $D/docpf.out 2>&1
