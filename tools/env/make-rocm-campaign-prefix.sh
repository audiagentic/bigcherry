#!/usr/bin/env bash
# Build a campaign-compatible ROCm prefix around a real ROCm/TheRock install.
#
# The patch campaign compiles with $HIP_PATH/bin/clang and
# -DCMAKE_PREFIX_PATH=$HIP_PATH. TheRock trees keep clang under llvm/bin
# (and /opt/rocm under lib/llvm/bin), and amdclang must be invoked under its
# own name, so this creates <wrap>/ with every top-level entry of <real>
# symlinked, bin/ entries symlinked, and bin/clang(++) as exec wrappers.
#
# Usage: tools/env/make-rocm-campaign-prefix.sh <real-rocm-root> <wrap-root>
# e.g.   tools/env/make-rocm-campaign-prefix.sh vendor/rocm/10.1.0rc2 vendor/rocm/10.1.0rc2-campaign
set -euo pipefail
real=$(cd "$1" && pwd)
wrap=$2
clang_dir=""
for candidate in "$real/llvm/bin" "$real/lib/llvm/bin"; do
    if [[ -x "$candidate/amdclang" ]]; then clang_dir=$candidate; break; fi
done
[[ -n "$clang_dir" ]] || { echo "no amdclang under $real/llvm/bin or $real/lib/llvm/bin" >&2; exit 1; }

rm -rf "$wrap"
mkdir -p "$wrap/bin"
wrap=$(cd "$wrap" && pwd)
for entry in "$real"/*; do
    [[ "$(basename "$entry")" == bin ]] && continue
    ln -s "$entry" "$wrap/$(basename "$entry")"
done
for tool in "$real"/bin/*; do
    ln -s "$tool" "$wrap/bin/$(basename "$tool")"
done
printf '#!/bin/bash\nexec %s/amdclang "$@"\n' "$clang_dir" > "$wrap/bin/clang"
printf '#!/bin/bash\nexec %s/amdclang++ "$@"\n' "$clang_dir" > "$wrap/bin/clang++"
chmod +x "$wrap/bin/clang" "$wrap/bin/clang++"

probe=$(mktemp -d)
trap 'rm -rf "$probe"' EXIT
echo 'int main(){return 0;}' > "$probe/t.c"
"$wrap/bin/clang" "$probe/t.c" -o "$probe/t" && "$probe/t"
echo "campaign prefix ready: $wrap ($("$real/bin/hipconfig" --version 2>/dev/null || echo unknown))"
