#!/bin/bash
# VTP44 final-build confirmation (build with the GPT review fixes): 1) 2-document record + self-replay (KLD must be 0; the route's ids must equal the
# earlier build's); 2) negative runs, each of which must fail with its message; 3) the 11-document route re-recorded (ids equal the earlier build's) and one
# replayed arm whose per-position KLD rows must be byte-identical to the earlier build's. Run under the gpu lease, after run_kld3.sh.
set -u
F=<dev-root>/tmp/bigcherry/vtp44-fn
B=<work-root>/builds/266a4b73e1d9555ea3fa58acde819c36/b54290b18d5c4fe2caeb70088d73a203
M=<models-root>/radiance/bigcherry/qwen3.8-next-flash-hcbf16/qwen3.8-next-flash-hcbf16.rad
O=$F/final; mkdir -p "$O"
REF2=$F/routing-in/ref; REF3=$F/kld3/ref3
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:/opt/ffmpeg6/lib HIP_PATH=/opt/rocm-7.2.4 ROCM_PATH=/opt/rocm-7.2.4
export HIP_VISIBLE_DEVICES=0,1 ROCR_VISIBLE_DEVICES=0,1
{ echo "start_utc $(date -u +%FT%TZ)"; echo "radiance_sha256 $(sha256sum $B/bin/radiance | cut -d' ' -f1)"; echo "build $B"; } >> "$O/receipt.txt"
run() { # tag mbt kv maxlen extra...
  local tag=$1 mbt=$2 kv=$3 ml=$4 t0=$(date +%s); shift 4
  timeout 3600 $B/bin/radiance --model "$M" --radiance-home $B/radiance_home --kernels libr3,libr3moe,libref \
    --placement expert_tiered --tp 2 --tp-wire exact --host-pool-mib 14336 --gpu-headroom-mib 320 \
    --max-num-seqs 1 --max-num-batched-tokens "$mbt" --num-speculative-tokens 0 \
    --max-model-len $ml --kv-cache-dtype "$kv" --deterministic "$@" > "$O/$tag.log" 2>&1
  local rc=$?
  echo "$(date -u +%T) $tag exit $rc $(( $(date +%s) - t0 ))s" | tee -a "$O/receipt.txt"
  return $rc
}
score() { echo "--kld-ref $1 --kld-out $O/$2.kld.json --expert-stats $O/$2.stats.json"; }
neg() { # tag pattern: the last run must have failed and said $pattern
  local tag=$1 pat=$2 rc=$3
  if [ "$rc" != 0 ] && grep -qiE "$pat" "$O/$tag.log"; then echo "NEG $tag PASS (exit $rc, message matched)" | tee -a "$O/receipt.txt"
  else echo "NEG $tag FAIL (exit $rc, pattern '$pat': $(grep -ciE "$pat" "$O/$tag.log") matches)" | tee -a "$O/receipt.txt"; fi
}
# 1) the 2-document set
run rec2 64 bf16 8192 $(score $REF2 rec2) --expert-route-record "$O/rec2.route"
run self2 64 bf16 8192 $(score $REF2 self2) --expert-route-replay "$O/rec2.route"
# 2) negatives
python3 - <<PY
import re
d=open("$O/rec2.route","rb").read()
open("$O/bad-digest.route","wb").write(re.sub(rb'("doc_hashes":\["[0-9a-f]{15})([0-9a-f])', lambda m: m.group(1)+(b"0" if m.group(2)!=b"0" else b"1"), d, count=1))
open("$O/truncated.route","wb").write(d[:-1000])
open("$O/incomplete.route","wb").write(d.replace(b'"complete":true', b'"complete":fals', 1))
PY
run neg-olddigest 64 bf16 8192 $(score $REF2 neg-olddigest) --expert-route-replay "$F/route-ref.route"; neg neg-olddigest "no document digests" $?
run neg-digest 64 bf16 8192 $(score $REF2 neg-digest) --expert-route-replay "$O/bad-digest.route"; neg neg-digest "token digests differ" $?
run neg-trunc 64 bf16 8192 $(score $REF2 neg-trunc) --expert-route-replay "$O/truncated.route"; neg neg-trunc "size is not its header" $?
run neg-incomplete 64 bf16 8192 $(score $REF2 neg-incomplete) --expert-route-replay "$O/incomplete.route"; neg neg-incomplete "incomplete" $?
run neg-wrongref 64 bf16 20480 $(score $REF3 neg-wrongref) --expert-route-replay "$O/rec2.route"; neg neg-wrongref "longer than the record|more documents|replayed .* rows|digests differ" $?
run neg-both 64 bf16 8192 $(score $REF2 neg-both) --expert-route-replay "$O/rec2.route" --expert-route-record "$O/neg-both.route"; neg neg-both "exclusive" $?
run neg-seqs 64 bf16 8192 $(score $REF2 neg-seqs) --expert-route-replay "$O/rec2.route" --max-num-seqs 2; neg neg-seqs "max-num-seqs 1" $?
# 3) the 11-document equivalence
run route3 64 bf16 20480 $(score $REF3 route3) --expert-route-record "$O/route3.route"
run replay3-fp8 64 fp8 20480 $(score $REF3 replay3-fp8) --expert-route-replay "$O/route3.route"
echo "end_utc $(date -u +%FT%TZ)" >> "$O/receipt.txt"
