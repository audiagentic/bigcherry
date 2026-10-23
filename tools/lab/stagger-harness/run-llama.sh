#!/bin/bash
# QFP49 rung C: build llama_prefill_harness.cpp against a build and run it once per arm, comparing the arms' logits.
# Run as a queue SCRIPT job; the first argument is the build's llama-server path (the queue resolves @<build run>).
#   ARMS='seq: ; pair:BIGCHERRY_PREFILL_STAGGER=256' ... VIS=0,1,2 SCRIPT lh1 tools/lab/stagger-harness/run-llama.sh @<build run> <out-dir> [harness args]
# ARMS is a ';'-separated list of <name>:<VAR=value ...>; every arm also gets the caller's BIGCHERRY_* / GGML_*
# environment and BASE_ENV. Each arm's logits are compared with the first arm's (cmp: bit-identical or not).
# TRACE=1 runs every arm under rocprofv3 and prints kernel-overlap-stats.py for it (slower; timings are then traced).
# env: SRC (source tree of the build; required), BC_MODEL, CARDS (0,1,2), BASE_ENV, OT, HARNESS_TIMEOUT (600 s),
#      GGML_CUDA_ALLREDUCE (cpu-root, the lab profiles' --allreduce), BIGCHERRY_FEATURES (flashnext)
set -u
server=$1; out=$2; shift 2
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
bin=$(dirname "$server")
[ -d "${SRC:-/nonexistent}/include" ] || { echo "NO_SOURCE_TREE (set SRC)"; exit 1; }
model=${BC_MODEL:-${BC_MODEL_ROOT}/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
export LD_LIBRARY_PATH="$bin:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES=${CARDS:-0,1,2} ROCR_VISIBLE_DEVICES=${CARDS:-0,1,2}
export GGML_CUDA_ALLREDUCE=${GGML_CUDA_ALLREDUCE:-cpu-root} BIGCHERRY_FEATURES=${BIGCHERRY_FEATURES:-flashnext}
if ! g++ -O2 -std=c++17 "$here/llama_prefill_harness.cpp" -I"$SRC/include" -I"$SRC/ggml/include" -L"$bin" -lllama -lggml -lggml-base \
        -Wl,-rpath,"$bin" -o "$out/llama_prefill_harness" > "$out/build.log" 2>&1; then
    echo "BUILD_FAILED"; grep -E "error" "$out/build.log" | head -12 | cut -c1-220; exit 1
fi
# OT: space-separated placement overrides, each '<regex>=CPU' (kept out of the queue's job line, which is word-split)
for o in ${OT:-}; do set -- "$@" -ot "$o"; done
first=
IFS=';' read -ra arms <<< "${ARMS:-seq:}"
for arm in "${arms[@]}"; do
    name=$(echo "${arm%%:*}" | tr -d ' '); vars=${arm#*:}
    [ -n "$name" ] || continue
    wrap=()
    [ "${TRACE:-0}" = 1 ] && wrap=(rocprofv3 --kernel-trace --output-format csv -d "$out/rocprof-$name" --)
    echo "######## arm $name: ${vars:-(no extra environment)}"
    # shellcheck disable=SC2086
    env ${BASE_ENV:-} $vars timeout "${HARNESS_TIMEOUT:-600}" "${wrap[@]}" "$out/llama_prefill_harness" -m "$model" -o "$out/logits-$name.bin" "$@" \
        > "$out/$name.log" 2>&1
    rc=$?
    grep -E "^== |^wall |^host ms|^logits |PATCH_HIT patch=136|STAGGER declined|GGML_ASSERT|ROCm error|failed" "$out/$name.log" | cut -c1-260
    echo "arm $name exit $rc"
    if [ -z "$first" ]; then
        first=$name
    elif [ -s "$out/logits-$name.bin" ] && cmp -s "$out/logits-$first.bin" "$out/logits-$name.bin"; then
        echo "arm $name logits: IDENTICAL to $first"
    else
        echo "arm $name logits: DIFFERENT from $first"
    fi
    [ "${TRACE:-0}" = 1 ] && python3 "$here/../flash-next/kernel-overlap-stats.py" "$out/rocprof-$name" 2>&1 | head -40
done
