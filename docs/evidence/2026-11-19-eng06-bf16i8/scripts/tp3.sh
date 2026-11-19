#!/bin/bash
# TP3 27B -556-long: bf16_i8 vs fp8 vs bf16, ABBA-style, one fresh start each; unique run dir per launch (no replay)
D=<dev-root>/tmp/bigcherry/eng06-hw
B=<work-root>/builds/36aef34754713df1487f4e97febd74bb/6cdd2755227c7f95996c02ad8f0ed8be
n=0
for spec in "i8 qwen27b-tp3-int8-556-long engine.kv_cache_dtype=\"bf16_i8\"" "fp8 qwen27b-tp3-int8-556-long" "bf16 qwen27b-tp3-int8-556-long-bf16" "bf16 qwen27b-tp3-int8-556-long-bf16" "fp8 qwen27b-tp3-int8-556-long" "i8 qwen27b-tp3-int8-556-long engine.kv_cache_dtype=\"bf16_i8\""; do
  set -- $spec; arm=$1; cfg=$2; shift 2; n=$((n+1))
  EXTRA_SETS="$*" bash $D/one.sh $B $D/t$n-$arm $cfg
  echo "tp3 launch $n $arm $cfg exit=$? at $(date -Is)" >> $D/tp3.log
  sleep 20
done
