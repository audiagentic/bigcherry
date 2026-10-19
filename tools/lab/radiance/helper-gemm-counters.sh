#!/bin/bash
# RAD06: hardware counters for the helper grouped MoE GEMM on replayed real requests (gfx1100).
# Usage: helper-gemm-counters.sh <build-dir> <model.rad> <capture.bin> <out-dir>
set -u
B=$1; M=$2; F=$3; OUT=$4
export ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:$PATH
rm -rf "$OUT"
for set in "SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS" "SQ_WAIT_INST_LDS SQ_WAIT_ANY SQ_BUSY_CYCLES GRBM_GUI_ACTIVE" "SQ_INSTS_VMEM SQ_INSTS_WMMA SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE"; do
  tag=$(echo "$set" | awk '{print $1}')
  HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 R3_MOE_WORKER_MIB=22528 R3_MOE_WORKER_SMALL_ROWS=8 R3_MOE_WORKER_MAX_ROWS=4096 \
    R3_MOE_NATIVE=1 R3_MOE_WORKER_MOE_DOWN_TILE=1 RADIANCE_HOME=$B/radiance_home timeout 900 \
    rocprofv3 --pmc $set --output-format csv -d "$OUT/$tag" -o run -- \
    $B/radiance_home/helpers/r3_moe_worker --replay "$M" "$F" > "$OUT-$tag.log" 2>&1
  echo "== $set: rc=$?"
  python3 - "$OUT/$tag" <<'PY'
import csv, glob, sys, collections
agg = collections.defaultdict(float)
for f in glob.glob(sys.argv[1] + '/**/*counter_collection.csv', recursive=True):
    for r in csv.DictReader(open(f)):
        if 'moe_gemm_q_kernel' in r['Kernel_Name']:
            agg[r['Counter_Name']] += float(r['Counter_Value'])
for k, v in sorted(agg.items()):
    print('  %-22s %.4g' % (k, v))
PY
done
echo COUNTERS_DONE
