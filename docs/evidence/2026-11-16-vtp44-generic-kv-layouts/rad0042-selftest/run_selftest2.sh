#!/bin/bash
# VTP44: the rad0042 selftest overlay with the inventory fix and the gqa6/block-4 fold cases, on one gfx1100 card, one case at a time.
# A case the plugin has no smoke test for prints "skip ... no smoke case" (rc 2): reported as SKIP, not FAIL. Run under the gpu lease.
set -u
D=<dev-root>/tmp/bigcherry/rad0042
SELF=$D/libr3/r3_selftest
# the kernels are the replay build's libr3 plugin (the overlay changes only the selftest binary)
LIB=<work-root>/builds/2a71a0404317ae66cb97299a1cbc8939/44a32c5d01d41bb0ccd49d061e85d822/radiance_home/kernels/libr3.so
[ -f "$LIB" ] || { echo "no libr3.so" >&2; exit 2; }
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:/opt/ffmpeg6/lib HIP_PATH=/opt/rocm-7.2.4 ROCM_PATH=/opt/rocm-7.2.4
unset HIP_VISIBLE_DEVICES; export ROCR_VISIBLE_DEVICES=0
O=$D/run; mkdir -p $O; : > $O/summary.txt
echo "start_utc $(date -u +%FT%TZ) selftest $SELF lib $LIB" >> $O/summary.txt
{ echo bf16_i8_rows
  for t in b16f8 b16i8; do
    for r in attn_prefill_h256_gqa6_$t attn_decode_h256_gqa6_$t attn_prefill_h256_gqa12_$t attn_decode_h256_gqa12_$t \
             attn_prefill_h256_gqa6_bs4_$t attn_decode_h256_gqa6_bs4_$t attn_decode_gq_h256_gqa12_$t attn_decode_gq_h256_gqa6_bs4_$t kv_store_$t qk_norm_rope_gate_kv_store_$t; do echo $r; done
  done; } > $O/candidates.txt
while read -r c; do
  t0=$(date +%s)
  timeout 1200 $SELF $LIB --case $c > $O/$c.txt 2>&1; rc=$?
  if grep -q "no smoke case" $O/$c.txt; then v=SKIP; elif [ $rc = 0 ] && ! grep -q "^  FAIL" $O/$c.txt; then v=pass; else v="FAIL rc=$rc"; fi
  echo "$c $v $(grep -c '^  ok ' $O/$c.txt) ok $(( $(date +%s) - t0 ))s" | tee -a $O/summary.txt
done < $O/candidates.txt
echo "end_utc $(date -u +%FT%TZ)" >> $O/summary.txt
