#!/bin/bash
# the SHIPPED configs from the slice branch: TP3 -long-bf16i8 sweep (TTFT + decode), Flash-Next 4-session variant needle (policy off: no calibration for its name)
D=<dev-root>/tmp/bigcherry/eng06-hw
B=<work-root>/builds/36aef34754713df1487f4e97febd74bb/6cdd2755227c7f95996c02ad8f0ed8be
export REPO_CFG=<dev-root>/projects/bigcherry/workspaces/main.worktrees/eng06-bf16i8-ship
bash $D/one.sh $B $D/S1-tp3-shipped qwen27b-tp3-int8-556-long-bf16i8
echo "shipped 1 tp3 exit=$? at $(date -Is)" >> $D/shipped.log
sleep 20
NEEDLE=1 bash $D/one.sh $B $D/S2-fn-shipped flashnext-4gpu-bf16i8
echo "shipped 2 fn exit=$? at $(date -Is)" >> $D/shipped.log
