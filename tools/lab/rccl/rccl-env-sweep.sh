#!/bin/bash
# RCCL tuning sweep for the dual-XTX SHM transport (stock kernel: XTX<->XTX P2P is off, RCCL goes via host
# shared memory). llama-bench pp512/pp4096 + tg128 on Qwen3.8-27B Q8_0 -sm tensor for each env setting,
# 3 reps each, two passes in reversed order to expose drift. Usage: rccl-env-sweep.sh <llama-server> <out-dir>
set -u
failed=0
srv=$1 out=$2; mkdir -p "$out"
bench=$(dirname "$srv")/llama-bench
model=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
declare -A cfg=(
  [default]=""
  [shm-ce]="NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3"
  [proto-simple]="NCCL_PROTO=Simple"
  [proto-ll128]="NCCL_PROTO=LL128"
  [ch4]="NCCL_MIN_NCHANNELS=4"
  [buf16m]="NCCL_BUFFSIZE=16777216"
  [shm-ce-ch4]="NCCL_SHM_USE_CUDA_MEMCPY=1 NCCL_SHM_MEMCPY_MODE=3 NCCL_MIN_NCHANNELS=4"
)
order=(default shm-ce proto-simple proto-ll128 ch4 buf16m shm-ce-ch4)
rev=(); for ((i=${#order[@]}-1; i>=0; i--)); do rev+=("${order[$i]}"); done
printf 'pass\tarm\tbench_exit\tcsv_exit\n' > "$out/status.tsv"
for pass in a b; do
  [ $pass = a ] && list=("${order[@]}") || list=("${rev[@]}")
  for name in "${list[@]}"; do
    echo "== $pass $name: ${cfg[$name]}"
    # 300 s cap: NCCL_SHM_USE_CUDA_MEMCPY hung llama-bench indefinitely on 2026-10-02.
    timeout -k 10 300 env ${cfg[$name]} HIP_VISIBLE_DEVICES=0,1 ROCR_VISIBLE_DEVICES=0,1 "$bench" -m "$model" -sm tensor -ngl 99 \
      -fa 1 -p 512,4096 -n 128 -ub 512 -b 2048 -r 3 -o csv 2> "$out/$pass-$name.stderr" > "$out/$pass-$name.csv"
    bench_rc=$?
    python3 - "$out/$pass-$name.csv" "$pass" "$name" <<'PY'
import csv, math, sys
try:
    with open(sys.argv[1], newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty CSV")
    for r in rows:
        rate = float(r["avg_ts"])
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("invalid throughput")
        if "n_prompt" not in r or "n_gen" not in r:
            raise ValueError("missing shape fields")
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(sys.argv[2], sys.argv[3], "INVALID_RESULT", str(exc))
    raise SystemExit(1)
print(sys.argv[2], sys.argv[3], " ".join(
    f"{('pp'+r['n_prompt']) if r['n_prompt']!='0' else ('tg'+r['n_gen'])}={float(r['avg_ts']):.1f}" for r in rows))

PY
    csv_rc=$?
    printf '%s\t%s\t%s\t%s\n' "$pass" "$name" "$bench_rc" "$csv_rc" >> "$out/status.tsv"
    if (( bench_rc != 0 || csv_rc != 0 )); then failed=1; fi
  done
done
if (( failed != 0 )); then
  echo SWEEP_INCOMPLETE
  exit 1
fi
echo SWEEP_DONE
