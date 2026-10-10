#!/bin/bash
# Do the patches that lower MTP draft acceptance (queue-decode-acceptance.sh, 2026-10-09: 1334 and 1347 decode faster
# with the patch off, acceptance 57% -> 62-63%) move the target's next-token distribution away from the reference?
# One flash-fidelity.sh run per switch on the given build at 8K, without MTP:
#   D = production (patch on), S = the switch off, D2 = production again (floor), C = CPU f32 reference.
# Reading: if S is closer to C than D is, by more than the D-D2 floor, the patch costs accuracy and that is the
# likely cause of the lower acceptance; if D and S are equally far from C, the acceptance change is not an accuracy
# loss of the target.
# Usage: queue-accuracy-off-switches.sh <build tag, e.g. b-main3> [ID:VAR=value ...]
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=/mnt/data/bigcherry-work/runs
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
build=$1; shift
[ $# -eq 0 ] && set -- 1334:BIGCHERRY_FA_SPARSE=0 1347:BIGCHERRY_F32_THIN_MMVF=0 1350:BIGCHERRY_MMQ_FEW_TILE_STREAMK=0
for item in "$@"; do
    id=${item%%:*} sw=${item#*:}
    run=acc-$id-off
    echo "== $id $sw (D = production, S = switch off, C = CPU f32 reference)"
    rm -rf "$R/$run" "$R/$run.log"
    j=$(mktemp)
    echo "VIS=0,1,2,3 SCRIPT $run tools/lab/flash-next/flash-fidelity.sh 8192 @$build $R/$run ref $sw" > "$j"
    bash tools/lab/plan-qualification/queue.sh "$j" | grep -E "rc=|blocked"
    rm -f "$j"
    grep -vE "PCIe" "$R/$run.log" | cut -c1-260
done
echo ACCURACY_OFF_SWITCHES_DONE
