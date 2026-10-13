#!/bin/bash
# Flash-Next output quality with the R9700 expert helper (RAD03). Greedy text cannot be byte-identical across gfx1100
# and gfx1201 kernels, so quality is scored as distribution distance: radiance KL mode records the helper-off logits
# over a corpus, then scores a helper-off rerun (the control: run-to-run noise) and a helper-on run against them.
# KL mode is prefill-only, so steps are capped at 64 tokens (the helper lane limit) and R3_MOE_HELPER_MAX_ROWS=64
# sends every step's owned routes through the R9700.
# Usage: B=<lane build dir> MODEL=<.rad> tools/lab/radiance/fn-helper-kld.sh <out-dir>
set -u
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?out dir}
B=${B:?set B to the lane build directory}
model=${MODEL:?MODEL}
corpus=${CORPUS:-/mnt/data/bigcherry-work/corpus/rad-kld-corpus.jsonl}
home=$B/radiance_home
helper=$home/helpers/r3_moe_helper
ref=$out/reference
mkdir -p "$out"
[ -s "$corpus" ] || { echo "NO_CORPUS $corpus"; exit 1; }
args=(--tp 2 --tp-wire exact --placement expert_tiered --host-pool-mib 14336 --gpu-headroom-mib 320
      --kv-cache-dtype fp8 --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 64
      --radiance-home "$home" --kernels libr3,libref --model "$model")

run() {  # <name> <helper 0|1> <mode args...>
    local name=$1 h=$2; shift 2
    local hp=
    rm -f /dev/shm/r3-moe-helper
    if [ "$h" = 1 ]; then
        HIP_VISIBLE_DEVICES=2 RADIANCE_HOME="$home" R3_MOE_HELPER_MAX_ROWS=64 "$helper" "$model" > "$out/$name.helper.log" 2>&1 &
        hp=$!
        until grep -q '^r3_moe_helper: ready ' "$out/$name.helper.log"; do
            kill -0 "$hp" 2>/dev/null || { echo "HELPER_EXITED"; tail -5 "$out/$name.helper.log"; return 1; }
            sleep 2
        done
    fi
    HIP_VISIBLE_DEVICES=0,1 R3_MOE_HELPER=$h R3_MOE_HELPER_MAX_ROWS=64 timeout 5400 "$B/bin/radiance" "${args[@]}" "$@" \
        > "$out/$name.log" 2>&1
    local rc=$?
    [ -n "$hp" ] && { kill -TERM "$hp"; wait "$hp"; grep served "$out/$name.helper.log"; }
    echo "== $name (helper $h): exit $rc"
    grep -E "KLD mean|top-1 agreement|recorded|^E " "$out/$name.log" | head -4 | cut -c1-200
}

run ref 0 --kld-record "$ref" --kld-corpus "$corpus" || exit 1
run control 0 --kld-ref "$ref" --kld-out "$out/control.json"
run helper 1 --kld-ref "$ref" --kld-out "$out/helper.json"
[ "${KEEP_REF:-0}" = 1 ] || rm -rf "$ref"
echo FN_HELPER_KLD_DONE
