#!/bin/bash
# RCCL 2-rank AllReduce bandwidth on the two 7900 XTX (PCIe 4.0 x8 each, no P2P) at prefill sizes:
# 21 and 42 MiB (ubatch 2048 bf16 / f32), Simple protocol, P2P disabled, NCCL_BUFFSIZE 1/4/8/16 MiB.
# Question (GPT 2026-10-01): can RCCL overlap D2H/H2D beyond the ~6.5 GB/s serial host-stage ceiling?
# Usage: rccl-bw-sweep.sh <out-dir>
set -u
out=$1; mkdir -p "$out"
bin=/home/audumla/rccl-heterogeneous-src/rccl-tests/build/all_reduce_perf
export HIP_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_PROTO=Simple
for bs in 1048576 4194304 8388608 16777216; do
  for dt in bfloat16 float; do
    NCCL_BUFFSIZE=$bs "$bin" -g 2 -b 21M -e 42M -f 2 -d $dt -n 20 -w 5 > "$out/bs$bs-$dt.txt" 2>&1
    echo "BUFFSIZE=$bs dtype=$dt"; grep -E "^\s+[0-9]+ " "$out/bs$bs-$dt.txt" | head -4
  done
done
echo SWEEP_DONE
