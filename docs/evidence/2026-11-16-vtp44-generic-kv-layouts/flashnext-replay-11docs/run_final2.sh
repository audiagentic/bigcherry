#!/bin/bash
# VTP44: separating run-to-run nondeterminism from a cross-build numeric difference. 1) the earlier build's 11-document route (its header given the
# document digests, the documents being the same) replayed with fp8 on the review-fix build: the per-position KLD rows must equal the earlier build's
# replayed fp8 (the replay path did not change). 2) the route recorded a second time on the review-fix build: must equal its first recording.
set -u
F=<dev-root>/tmp/bigcherry/vtp44-fn
B=<work-root>/builds/266a4b73e1d9555ea3fa58acde819c36/b54290b18d5c4fe2caeb70088d73a203
M=<models-root>/radiance/bigcherry/qwen3.8-next-flash-hcbf16/qwen3.8-next-flash-hcbf16.rad
O=$F/final; REF3=$F/kld3/ref3
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:/opt/ffmpeg6/lib HIP_PATH=/opt/rocm-7.2.4 ROCM_PATH=/opt/rocm-7.2.4
export HIP_VISIBLE_DEVICES=0,1 ROCR_VISIBLE_DEVICES=0,1
run() { local tag=$1 mbt=$2 kv=$3 t0=$(date +%s); shift 3
  timeout 3600 $B/bin/radiance --model "$M" --radiance-home $B/radiance_home --kernels libr3,libr3moe,libref --placement expert_tiered --tp 2 --tp-wire exact \
    --host-pool-mib 14336 --gpu-headroom-mib 320 --max-num-seqs 1 --max-num-batched-tokens "$mbt" --num-speculative-tokens 0 --max-model-len 20480 \
    --kv-cache-dtype "$kv" --deterministic "$@" > "$O/$tag.log" 2>&1
  echo "$(date -u +%T) $tag exit $? $(( $(date +%s) - t0 ))s" | tee -a "$O/receipt.txt"; }
score() { echo "--kld-ref $REF3 --kld-out $O/$1.kld.json --expert-stats $O/$1.stats.json"; }
run replay3-oldroute-fp8 64 fp8 $(score replay3-oldroute-fp8) --expert-route-replay "$O/route3-old-with-digests.route"
run route3b 64 bf16 $(score route3b) --expert-route-record "$O/route3b.route"
echo "end2_utc $(date -u +%FT%TZ)" >> "$O/receipt.txt"
