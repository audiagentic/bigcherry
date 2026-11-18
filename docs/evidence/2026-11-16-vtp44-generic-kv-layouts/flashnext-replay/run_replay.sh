#!/bin/bash
# VTP44 routing-flip measurement, step 1: per-layer routed-expert tallies (--expert-stats) of the same two documents under
# different KV caches, to see whether the cache changes which experts tokens are sent to. Flash-Next (qwen3.8-next-flash-hcbf16),
# two 7900 XTX TP2 exact wire, no helper, deterministic, 1 sequence, no speculation: the recipe of the KLD calibration runs.
#   bf16-a / bf16-b   same config twice (control: must match exactly)
#   bf16-mbt128       only the chunk size changes (arithmetic-noise reference; KLD band RAD39)
#   bf16_i8, bf16_fp8, fp8   cache type changes
# Run under the gpu lease. Inputs live in the task folder (copied there so a workspace cleanup cannot remove them).
set -u
F=<dev-root>/tmp/bigcherry/vtp44-fn
B=<work-root>/builds/2a71a0404317ae66cb97299a1cbc8939/44a32c5d01d41bb0ccd49d061e85d822
M=<models-root>/radiance/bigcherry/qwen3.8-next-flash-hcbf16/qwen3.8-next-flash-hcbf16.rad
O=$F/replay
REF=$F/routing-in/ref
mkdir -p "$O"
[ -d "$REF" ] || { echo "reference missing: $REF" >&2; exit 2; }
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:/opt/ffmpeg6/lib HIP_PATH=/opt/rocm-7.2.4 ROCM_PATH=/opt/rocm-7.2.4
export HIP_VISIBLE_DEVICES=0,1 ROCR_VISIBLE_DEVICES=0,1
{ echo "start_utc $(date -u +%FT%TZ)"; echo "radiance_sha256 $(sha256sum $B/bin/radiance | cut -d' ' -f1)"; echo "build $B"; } >> "$O/receipt.txt"
run() { # tag mbt kvdtype
  local tag=$1 mbt=$2 kv=$3 t0=$(date +%s)
  timeout 1800 $B/bin/radiance --model "$M" --radiance-home $B/radiance_home --kernels libr3,libr3moe,libref \
    --placement expert_tiered --tp 2 --tp-wire exact --host-pool-mib 14336 --gpu-headroom-mib 320 \
    --max-num-seqs 1 --max-num-batched-tokens "$mbt" --num-speculative-tokens 0 \
    --max-model-len 8192 --kv-cache-dtype "$kv" --deterministic \
    --kld-ref "$REF" --kld-out "$O/$tag.kld.json" --expert-stats "$O/$tag.stats.json" --expert-route-replay "$F/route-ref.route" > "$O/$tag.log" 2>&1
  echo "$(date -u +%T) $tag exit $? $(( $(date +%s) - t0 ))s" | tee -a "$O/receipt.txt"
}
run replay-self 64 bf16
run replay-mbt128 128 bf16
run replay-bf16_i8 64 bf16_i8
run replay-bf16_fp8 64 bf16_fp8
run replay-fp8 64 fp8
echo "end_utc $(date -u +%FT%TZ)" >> "$O/receipt.txt"
