#!/bin/bash
# Kernel/copy trace of one Flash-Next prefill under rocprofv3, optionally with the R9700 helper
# process running (HELPER=1) and the engine hook off (HOOK=0) or on. Compare two traces with
# fn-profile-compare.py to split a slowdown into kernel time vs gaps between kernels.
# Usage: B=<lane build dir> MODEL=<.rad> HELPER=0|1 HOOK=0|1 tools/lab/radiance/fn-profile.sh <out-dir>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?out dir}
B=${B:?set B to the lane build directory}
mkdir -p "$out"
wrap="$out/radiance-rocprof"
cat >"$wrap" <<EOF
#!/bin/bash
exec rocprofv3 --kernel-trace --memory-copy-trace --output-format csv -d "$out/trace" -o run -- "$B/bin/radiance" "\$@"
EOF
chmod +x "$wrap"
rm -f /dev/shm/r3-moe-helper
N=${N:-1} HELPER=${HELPER:-0} HOOK=${HOOK:-${HELPER:-0}} EXTRA_ARGS="${EXTRA_ARGS:---gpu-headroom-mib 320}" \
    RAD_BIN="$wrap" RADIANCE_HOME="$B/radiance_home" LOAD_TIMEOUT=${LOAD_TIMEOUT:-1500} \
    bash "$here/fn-helper.sh" "$out/run" 2>&1 | grep -E "prefill|decode|EXIT|TIMEOUT" || true
find "$out/trace" -name "*kernel_trace.csv" -o -name "*memory_copy_trace.csv" | sort
