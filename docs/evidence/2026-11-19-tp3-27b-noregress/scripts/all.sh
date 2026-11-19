#!/bin/bash
# ABBA over two configs, one fresh launch each (each config twice); a short pause between launches
D=<dev-root>/tmp/bigcherry/tp3-check
B=<work-root>/builds/36aef34754713df1487f4e97febd74bb/6cdd2755227c7f95996c02ad8f0ed8be
n=0
for cfg in qwen27b-tp3-int8-556-long-bf16 qwen27b-tp3-int8-556-long qwen27b-tp3-int8-556-long qwen27b-tp3-int8-556-long-bf16; do
  n=$((n+1)); bash $D/one.sh $B $D/l$n-$cfg $cfg
  echo "launch $n $cfg exit=$? at $(date -Is)" >> $D/all.log
  sleep 20
done
