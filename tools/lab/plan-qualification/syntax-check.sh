#!/bin/bash
# Syntax-check the host-side files a patch touches, after compose-apply.py has composed them, with a local clang++.
# Catches the compile errors of a patch written without a compiler (undeclared names, missing includes) in seconds;
# it does not replace the HIP build: ggml-cuda sources are skipped (they need the HIP headers and device macros).
# Usage: syntax-check.sh <compose dir>          (the <out-dir> given to compose-apply.py; run from the repository root)
# env: CXX (clang++ on PATH, else the ROCm one on Windows), VENDOR (vendor/llama.cpp)
set -u
compose=${1:?compose dir}
vendor=${VENDOR:-vendor/llama.cpp}
cxx=${CXX:-$(command -v clang++ || echo "/c/Program Files/AMD/ROCm/7.1/bin/clang++")}
full=$compose/_full
rm -rf "$full"; mkdir -p "$full/ggml/src" "$full/src"
# one header tree: the vendor's (production patches applied) with the composed files laid over it
cp -r "$vendor/ggml/include" "$full/ggml/"
cp "$vendor"/ggml/src/*.h "$vendor"/ggml/src/*.cpp "$vendor"/ggml/src/*.c "$full/ggml/src/" 2>/dev/null
cp "$vendor"/src/*.h "$full/src/" 2>/dev/null
[ -d "$compose/ggml/include" ] && cp "$compose"/ggml/include/* "$full/ggml/include/"
ls "$compose"/ggml/src/*.cpp > /dev/null 2>&1 && cp "$compose"/ggml/src/*.cpp "$full/ggml/src/"
[ -d "$compose/src" ] && cp "$compose"/src/* "$full/src/"
bad=0
for f in "$compose"/ggml/src/*.cpp "$compose"/src/*.cpp; do
    [ -f "$f" ] || continue
    rel=${f#"$compose"/}
    errors=$("$cxx" -std=c++17 -fsyntax-only -DGGML_BACKEND_SHARED -I"$full/src" -I"$vendor/src" -I"$vendor/include" \
        -I"$full/ggml/include" -I"$full/ggml/src" -I"$vendor/vendor" -I"$vendor/common" "$full/$rel" 2>&1 | grep -E "error|fatal")
    if [ -n "$errors" ]; then
        bad=$((bad + 1)); echo "FAIL $rel"; echo "$errors" | head -12 | cut -c1-240
    else
        echo "ok   $rel"
    fi
done
echo "$bad file(s) with errors"
[ "$bad" -eq 0 ]
