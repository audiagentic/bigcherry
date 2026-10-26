#!/bin/bash
# ROCm 10.1 vs 7.2.4 on the Flash-Next production layout: build this checkout with each ROCm (the radiance lane takes ROCm
# from PATH), then profile one decode per build (rad20-profile.sh: rocprofv3 on helper 0, tok/s, clocks) with each build's
# own runtime libraries. Usage: rocm10-compare.sh [rocm10 install=${BC_WORK_ROOT}/toolchains/rocm-10.1.0/install]
set -u
source tools/env/bigcherry-env.sh >/dev/null
R10=${1:-$BC_WORK_ROOT/toolchains/rocm-10.1.0/install}
OUT=$BC_WORK_ROOT/runs/rocm10-compare-$(date +%m%d%H%M); mkdir -p $OUT
git log -1 --oneline | tee $OUT/builds.txt   # this checkout (main + the RAD_HIPCC build fix)
build() {   # $1 label, $2 ROCm root
    ROCM_PATH=$2 HIP_PATH=$2 PATH=$2/bin:$2/llvm/bin:$PATH LD_LIBRARY_PATH=$2/lib \
        PYTHONPATH=tools timeout 7200 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance \
        --experiment moe-remote-handoff > $OUT/build-$1.log 2>&1
    grep -oE 'ok build_plan_id=[0-9a-f]+' $OUT/build-$1.log | tail -1 | cut -d= -f2
}
P72=$(build r72 /opt/rocm-7.2.4)
P10=$(build r101 $R10)
echo "rocm 7.2.4 plan $P72 / rocm 10.1 plan $P10" | tee -a $OUT/builds.txt
[ -n "$P72" ] && [ -n "$P10" ] && [ "$P72" != "$P10" ] || { echo "builds failed or share a plan (see $OUT)"; grep -hE ' error:|FAILED' $OUT/build-*.log | sort -u | head; exit 1; }
for p in $P72 $P10; do
    b=$(ls -d $BC_WORK_ROOT/builds/*/$p)
    echo "$p: $(LD_LIBRARY_PATH= ldd $b/bin/radiance | grep -m1 amdhip) | helper $(ldd $b/radiance_home/helpers/r3_moe_worker | grep -m1 amdhip)" | tee -a $OUT/builds.txt
done
bash tools/lab/radiance/rad20-profile.sh x x rocm10 "r72=$P72,LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib r101=$P10,LD_LIBRARY_PATH=$R10/lib" 2>&1 | tee $OUT/profile.txt | grep -E "tok/s|gemv|OUT"
echo "OUT $OUT"; echo ROCM10_DONE
