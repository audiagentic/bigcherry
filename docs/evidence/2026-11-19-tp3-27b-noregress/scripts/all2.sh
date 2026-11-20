#!/bin/bash
# control arm: same engine bits, bc-serve from before today's serving fixes (4fabff91f), sandwiched by new-arm launches
D=<dev-root>/tmp/bigcherry/tp3-check
NEW=<work-root>/builds/36aef34754713df1487f4e97febd74bb/6cdd2755227c7f95996c02ad8f0ed8be
OLD=$D/oldbuild
n=4
for spec in "$OLD qwen27b-tp3-int8-556-long-bf16 old" "$OLD qwen27b-tp3-int8-556-long old" "$NEW qwen27b-tp3-int8-556-long new" "$NEW qwen27b-tp3-int8-556-long-bf16 new"; do
  set -- $spec; n=$((n+1))
  bash $D/one.sh $1 $D/l$n-$3-$2 $2
  echo "launch $n $3 $2 exit=$? at $(date -Is)" >> $D/all.log
  sleep 20
done
