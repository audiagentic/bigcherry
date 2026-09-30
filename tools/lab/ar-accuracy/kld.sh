#!/bin/bash
# Logit KL-divergence of one AllReduce configuration against a reference, on a fixed corpus.
# Usage: kld.sh <perplexity-binary> <model> <corpus> <reference.kld> <base|compare> [KEY=VAL...] [-- extra args]
#   base:    writes <reference.kld> (run this for the reference configuration first)
#   compare: reads it and prints llama-perplexity's KLD / top-token statistics
# KEY=VAL pairs are exported for this run only (e.g. GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_WIRE=bf16).
# Output lands in $BC_RUN_DIR (set by queue.sh SCRIPT rows).
set -eu
bin=$1 model=$2 corpus=$3 ref=$4 mode=$5
shift 5
extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        --) shift; extra=("$@"); break ;;
        *=*) export "$1"; shift ;;
        *) echo "unexpected argument $1" >&2; exit 2 ;;
    esac
done
export HIP_VISIBLE_DEVICES=${BC_GPUS:-0,1} ROCR_VISIBLE_DEVICES=${BC_GPUS:-0,1}
common=(-m "$model" -f "$corpus" -c 2048 --chunks 32 -ngl 99 -sm tensor --flash-attn on "${extra[@]}")
out=${BC_RUN_DIR:-.}
env | grep -E '^(GGML_CUDA_ALLREDUCE|GGML_CUDA_AR_)' | sort > "$out/env.txt" || true
if [ "$mode" = compare ] && [ ! -s "$ref" ]; then echo "reference $ref missing; run the base configuration first" >&2; exit 3; fi
case "$mode" in
    base) "$bin" "${common[@]}" --kl-divergence-base "$ref" 2>&1 | tee "$out/perplexity.log" | tail -n 20 ;;
    compare) "$bin" "${common[@]}" --kl-divergence-base "$ref" --kl-divergence 2>&1 | tee "$out/perplexity.log" | tail -n 40 ;;
    *) echo "mode must be base or compare" >&2; exit 2 ;;
esac
