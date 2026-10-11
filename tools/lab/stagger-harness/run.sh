#!/bin/bash
# QFP49 rungs A and B: build meta_pair_harness.cpp against a build that has 1361 and run it on the target cards.
# Run as a queue SCRIPT job; the first argument is the build's llama-server path (the queue resolves @<build run>),
# whose bin directory holds the ggml libraries; the headers come from the source tree named in the build's log.
#   CARDS=0,1,2 ... VIS=0,1,2 SCRIPT sh1 tools/lab/stagger-harness/run.sh @<build run> <out-dir> [T L n k reps]
# env: CARDS (0,1,2), SRC (source tree; default: read from the build run.s log),
#      HARNESS_TIMEOUT (900 s). Any BIGCHERRY_* / GGML_HIP_* the caller exports is passed through (the pair needs
#      BIGCHERRY_AR_TRANSFER_STREAM=1 or whichever route flags the build under test wants). This harness defaults
#      BIGCHERRY_SCHED_ASYNC_INPUTS=1 so the production host-prefix copy path (1326) is exercised.
set -u
server=$1; out=$2; shift 2
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
bin=$(dirname "$server")
if [ -z "${SRC:-}" ]; then
    build=$(dirname "$(dirname "$bin")")
    SRC=$(grep -hoE "/[^ \"']*/work/sources/[0-9a-f]{32}" "$bin/../compile_commands.json" 2>/dev/null | head -1)
fi
[ -d "${SRC:-/nonexistent}/ggml/include" ] || { echo "NO_SOURCE_TREE (set SRC)"; exit 1; }
# the ggml libraries find ROCm through their own rpath; adding another ROCm here mixes two rocBLAS installs
export LD_LIBRARY_PATH="$bin:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES=${CARDS:-0,1,2}
export BIGCHERRY_SCHED_ASYNC_INPUTS=${BIGCHERRY_SCHED_ASYNC_INPUTS:-1}
libs=$(ls "$bin"/libggml*.so 2>/dev/null | sed -E 's#.*/lib(.*)\.so#-l\1#' | tr '\n' ' ')
if ! g++ -O2 -std=c++17 "$here/meta_pair_harness.cpp" -I"$SRC/ggml/include" -L"$bin" $libs -Wl,-rpath,"$bin" \
        -o "$out/meta_pair_harness" > "$out/build.log" 2>&1; then
    echo "BUILD_FAILED"; grep -E "error" "$out/build.log" | head -12 | cut -c1-220; exit 1
fi
echo "cards: HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES   source: $SRC"
env | grep -E "^(BIGCHERRY_|GGML_HIP_)" | sort | tr '\n' ' '; echo
timeout "${HARNESS_TIMEOUT:-900}" "$out/meta_pair_harness" "$@" 2>&1 | tee "$out/harness.log" | grep -vE "^(ggml_|load_backend|register_)" | cut -c1-240
rc=${PIPESTATUS[0]}
echo "harness exit $rc"
exit "$rc"
