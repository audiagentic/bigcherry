#!/bin/bash
# ENG06 step 3: TP3 27B decode at ~250K (1 stream) and ~200K (4 streams), bf16 vs bf16_i8 vs fp8, one fresh start each, ABBA-ish
D=<dev-root>/tmp/bigcherry/eng06-hw
B=<work-root>/builds/36aef34754713df1487f4e97febd74bb/6cdd2755227c7f95996c02ad8f0ed8be
n=0
for spec in "i8 qwen27b-tp3-int8-556-long engine.kv_cache_dtype=\"bf16_i8\"" "bf16 qwen27b-tp3-int8-556-long-bf16" "fp8 qwen27b-tp3-int8-556-long" "fp8 qwen27b-tp3-int8-556-long" "bf16 qwen27b-tp3-int8-556-long-bf16" "i8 qwen27b-tp3-int8-556-long engine.kv_cache_dtype=\"bf16_i8\""; do
  set -- $spec; arm=$1; cfg=$2; shift 2; n=$((n+1))
  LONGBENCH=1 EXTRA_SETS="$*" bash $D/one.sh $B $D/L$n-$arm $cfg
  echo "long launch $n $arm $cfg exit=$? at $(date -Is)" >> $D/long.log
  sleep 20
done
