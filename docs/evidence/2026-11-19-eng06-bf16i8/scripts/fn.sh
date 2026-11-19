#!/bin/bash
cd <dev-root>/projects/bigcherry/workspaces/main.worktrees/eng06-hw
export FLASHNEXT_PARITY_SETS='engine.kv_cache_dtype="bf16_i8" service.max_sessions=4' FLASHNEXT_PARITY_WARMUP_S=451
bash tools/lab/radiance/flashnext-parity.sh measure --build <work-root>/builds/36aef34754713df1487f4e97febd74bb/6cdd2755227c7f95996c02ad8f0ed8be --out <dev-root>/tmp/bigcherry/eng06-hw/fn-i8 --launches 2 > <dev-root>/tmp/bigcherry/eng06-hw/fn-i8.log 2>&1
echo "exit=$? at $(date -Is)" >> <dev-root>/tmp/bigcherry/eng06-hw/fn-i8.log
