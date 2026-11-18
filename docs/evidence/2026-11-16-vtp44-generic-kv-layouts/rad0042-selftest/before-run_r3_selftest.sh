#!/bin/bash
# VTP44: libr3's gfx1100 device selftest for the packed-KV rows (bf16_fp8 b16f8, bf16_i8 b16i8) at gqa6 and gqa12, one case at a time on one
# 7900 XTX. A case name the plugin does not have prints an error and is reported as unknown. Run under the gpu lease.
set -u
D=<dev-root>/tmp/bigcherry/rad0045
SELF=$D/r3selftest/libr3/r3_selftest
B=<work-root>/builds/2a71a0404317ae66cb97299a1cbc8939/44a32c5d01d41bb0ccd49d061e85d822
LIB=$B/radiance_home/kernels/libr3.so
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:/opt/ffmpeg6/lib HIP_PATH=/opt/rocm-7.2.4 ROCM_PATH=/opt/rocm-7.2.4
unset HIP_VISIBLE_DEVICES; export ROCR_VISIBLE_DEVICES=0
O=$D/r3selftest-run; mkdir -p $O; : > $O/summary.txt
echo "start_utc $(date -u +%FT%TZ) selftest $SELF lib $LIB" >> $O/summary.txt
for pre in attn_prefill_h256 attn_decode_h256 attn_decode_gq_h256; do
 for g in "" _gqa6 _gqa12; do
  for t in b16f8 b16i8; do
   for c in ${pre}${g}_${t}; do echo $c; done
  done
 done
done | sort -u > $O/candidates.txt
for c in bf16_i8_rows kv_store_b16f8 kv_store_b16i8 qk_norm_rope_gate_kv_store_b16f8 qk_norm_rope_gate_kv_store_b16i8; do echo $c >> $O/candidates.txt; done
while read -r c; do
  t0=$(date +%s)
  timeout 900 $SELF $LIB --case $c > $O/$c.txt 2>&1; rc=$?
  verdict=$(grep -qiE "no such|unknown|not found|no row" $O/$c.txt && echo unknown || { [ $rc = 0 ] && echo pass || echo "FAIL rc=$rc"; })
  echo "$c $verdict $(( $(date +%s) - t0 ))s" | tee -a $O/summary.txt
done < $O/candidates.txt
echo "end_utc $(date -u +%FT%TZ)" >> $O/summary.txt
