#!/bin/bash
# Output quality of the RAD02 layout: the R9700 runs Flash-Next alone (tp 1) and the XTX cards are expert helpers.
# Scored against an exact reference the two XTX record (helper-off, tp 2) -- the same reference the RAD03 noise floor
# was measured on (XTX re-chunking: KLD ~0.06 on the 4-doc corpus), so the bar is that floor, not bit-exactness.
# Usage: B=<lane build dir> MODEL=<.rad> tools/lab/radiance/fn-r9700-kld.sh <out-dir>
#   HELPER_DEVS ("0 1"), HELPER_MIB (20480), HOST_POOL_MIB (30720), SMALL_ROWS (8), REF_DIR (reuse a reference)
set -u
out=${1:?out dir}
B=${B:?set B to the lane build directory}
model=${MODEL:?MODEL}
corpus=${CORPUS:-/mnt/data/bigcherry-work/corpus/rad-kld-corpus-4.jsonl}
home=$B/radiance_home
helper=$home/helpers/r3_moe_helper
ref=${REF_DIR:-$out/reference}
mkdir -p "$out"
[ -s "$corpus" ] || { echo "NO_CORPUS $corpus"; exit 1; }
common=(--kv-cache-dtype fp8 --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 64
        --radiance-home "$home" --model "$model")

stop_helpers() {
    pgrep -x r3_moe_helper >/dev/null && { pkill -TERM -x r3_moe_helper; sleep 3; pkill -KILL -x r3_moe_helper; }
    rm -f /dev/shm/r3-moe-helper
}
report() {  # <name> <rc>
    echo "== $1: exit $2"
    grep -E "KLD mean|top-1 agreement|recorded|^E " "$out/$1.log" | head -4 | cut -c1-200
}

if [ -z "${REF_DIR:-}" ]; then
    stop_helpers
    HIP_VISIBLE_DEVICES=0,1 R3_MOE_HELPER=0 timeout 5400 "$B/bin/radiance" "${common[@]}" --tp 2 --tp-wire exact \
        --placement expert_tiered --host-pool-mib 14336 --gpu-headroom-mib 320 --kernels libr3,libr3moe,libref \
        --kld-record "$ref" --kld-corpus "$corpus" > "$out/ref.log" 2>&1
    report ref $?
fi

stop_helpers
read -r -a devs <<< "${HELPER_DEVS:-0 1}"
pids=()
for i in "${!devs[@]}"; do
    env HIP_VISIBLE_DEVICES=${devs[$i]} R3_MOE_HELPER_ENGINE_RANKS=1 R3_MOE_HELPER_COUNT=${#devs[@]} R3_MOE_HELPER_INDEX=$i \
        R3_MOE_HELPER_MIB=${HELPER_MIB:-20480} R3_MOE_HELPER_MAX_ROWS=64 R3_MOE_HELPER_SMALL_ROWS=${SMALL_ROWS:-8} \
        RADIANCE_HOME="$home" "$helper" "$model" > "$out/helper$i.log" 2>&1 &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    until grep -qE '^r3_moe_helper: (ready |waiting for engine runtime expert maps)' "$out/helper$i.log"; do
        kill -0 "${pids[$i]}" 2>/dev/null || { echo "HELPER_EXITED $i"; tail -5 "$out/helper$i.log"; exit 1; }
        sleep 2
    done
done
HIP_VISIBLE_DEVICES=2 R3_MOE_HELPER=1 R3_MOE_HELPER_SKIP_OWNED=1 R3_MOE_HELPER_MAX_ROWS=64 \
    timeout 5400 "$B/bin/radiance" "${common[@]}" --tp 1 --placement expert_tiered \
    --host-pool-mib "${HOST_POOL_MIB:-30720}" --kernels libr4d,libr3moe,libref \
    --kld-ref "$ref" --kld-out "$out/r9700.json" > "$out/r9700.log" 2>&1
rc=$?
for p in "${pids[@]}"; do kill -TERM "$p"; wait "$p"; done
report r9700 $rc
[ -n "${REF_DIR:-}" ] || [ "${KEEP_REF:-0}" = 1 ] || rm -rf "$ref"
echo FN_R9700_KLD_DONE
