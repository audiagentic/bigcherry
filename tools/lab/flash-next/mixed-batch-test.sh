#!/bin/bash
# QFP22: real mixed token/embd batches through upstream's test-llama-archs ("Mixed batch" column: token/embd/token
# chunks vs one mixed batch, per architecture and device config, incl. tensor split), for 1333.
# Builds the test target in each given build tree and runs it for the architectures we serve, so a patched build
# can be compared with an unpatched one. The chunked pass and the mixed pass alternate, which also exercises the
# 2-branch <-> 3-branch graph transitions of the on-demand patch.
# Usage: mixed-batch-test.sh <out-dir> <build run>... [log to wait for as last arg prefixed with wait=]
set -u
OUT=${1:?out dir}
shift
R=/mnt/data/bigcherry-work/runs
ARCHS=${ARCHS:-qwen4exp qwen35 qwen35moe gemma3}
mkdir -p "$OUT"
for arg in "$@"; do
    case "$arg" in wait=*) until grep -q "^ALL_JOBS_DONE" "${arg#wait=}" 2>/dev/null; do sleep 20; done ;; esac
done
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2}
for run in "$@"; do
    case "$run" in wait=*) continue ;; esac
    bin=$(sed -n 's/^BUILD_BINARY=//p' "$R/$run.log" | tail -1)
    [ -n "$bin" ] || { echo "$run: no BUILD_BINARY"; continue; }
    tree=$(dirname "$(dirname "$bin")")
    echo "== $run: building test-llama-archs in $tree"
    cmake --build "$tree" --target test-llama-archs -j 16 > "$OUT/$run.build.log" 2>&1 || { echo "$run: BUILD_FAILED"; tail -5 "$OUT/$run.build.log"; continue; }
    for arch in $ARCHS; do
        echo "-- $run $arch"
        timeout 900 "$tree/bin/test-llama-archs" -a "$arch" > "$OUT/$run.$arch.log" 2>&1
        echo "rc=$?"
        sed 's/\x1b\[[0-9;]*m//g' "$OUT/$run.$arch.log" | grep -E "^\|" | grep -vE "^\|-+\|" | tail -n +2
        sed 's/\x1b\[[0-9;]*m//g' "$OUT/$run.$arch.log" | grep -iE "realloc|GGML_ASSERT|error|failed|tests" | tail -4
    done
done
echo MIXED_TEST_DONE
