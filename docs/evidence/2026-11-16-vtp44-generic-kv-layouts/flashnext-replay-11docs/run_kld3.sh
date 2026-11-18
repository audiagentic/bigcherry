#!/bin/bash
# VTP44 closing KLD set: 11 independent documents (8 x ~4k tokens, 3 x ~16k tokens; corpus3.jsonl) on Flash-Next (hcbf16), two 7900 XTX TP2
# exact wire, no helper, deterministic, one sequence, no speculation. 1) bf16 chunk-64 reference logits (top-64); 2) the reference configuration
# scored against them while recording its expert route (must give KLD 0); 3) that route replayed in the self check, a chunk-128 run and the three
# KV types. Run under the gpu lease.
set -u
F=<dev-root>/tmp/bigcherry/vtp44-fn
B=<work-root>/builds/2a71a0404317ae66cb97299a1cbc8939/44a32c5d01d41bb0ccd49d061e85d822
M=<models-root>/radiance/bigcherry/qwen3.8-next-flash-hcbf16/qwen3.8-next-flash-hcbf16.rad
O=$F/kld3; mkdir -p "$O"
REF=$O/ref3; ROUTE=$O/route3.route; CORPUS=$F/corpus3.jsonl
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:/opt/ffmpeg6/lib HIP_PATH=/opt/rocm-7.2.4 ROCM_PATH=/opt/rocm-7.2.4
export HIP_VISIBLE_DEVICES=0,1 ROCR_VISIBLE_DEVICES=0,1
{ echo "start_utc $(date -u +%FT%TZ)"; echo "radiance_sha256 $(sha256sum $B/bin/radiance | cut -d' ' -f1)"; echo "build $B"; } >> "$O/receipt.txt"
run() { # tag mbt kv extra-args...
  local tag=$1 mbt=$2 kv=$3 t0=$(date +%s); shift 3
  timeout 3600 $B/bin/radiance --model "$M" --radiance-home $B/radiance_home --kernels libr3,libr3moe,libref \
    --placement expert_tiered --tp 2 --tp-wire exact --host-pool-mib 14336 --gpu-headroom-mib 320 \
    --max-num-seqs 1 --max-num-batched-tokens "$mbt" --num-speculative-tokens 0 \
    --max-model-len 20480 --kv-cache-dtype "$kv" --deterministic "$@" > "$O/$tag.log" 2>&1
  local rc=$?
  echo "$(date -u +%T) $tag exit $rc $(( $(date +%s) - t0 ))s" | tee -a "$O/receipt.txt"
  return $rc
}
score() { echo "--kld-ref $REF --kld-out $O/$1.kld.json --expert-stats $O/$1.stats.json"; }
run ref-record 64 bf16 --kld-record "$REF" --kld-corpus "$CORPUS" --kld-top 64 || { echo "STOP: reference not recorded" | tee -a "$O/receipt.txt"; exit 1; }
run ref-route 64 bf16 $(score ref-route) --expert-route-record "$ROUTE" || { echo "STOP: route not recorded" | tee -a "$O/receipt.txt"; exit 1; }
for arm in "replay-self 64 bf16" "replay-mbt128 128 bf16" "replay-bf16_i8 64 bf16_i8" "replay-bf16_fp8 64 bf16_fp8" "replay-fp8 64 fp8"; do
  set -- $arm
  run $1 $2 $3 $(score $1) --expert-route-replay "$ROUTE"
done
echo "end_utc $(date -u +%FT%TZ)" >> "$O/receipt.txt"
