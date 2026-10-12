#!/bin/bash
# Offline structural validation for every libr3 native rewrite against the source commit they are written for.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
pin=89cee7ce414486c717aa2085c99e1f522e4c0ea6
[ -d "$src/.git" ] || { echo "R3_REWRITE_CHECK_NO_SOURCE $src"; exit 1; }
got=$(git -C "$src" rev-parse HEAD)
[ "$got" = "$pin" ] || { echo "R3_REWRITE_CHECK_WRONG_PIN got=$got want=$pin"; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if ! python3 "$repo/engines/radiance/kernels/libr3/r3_rewrite.py" \
        "$src/libr4d" "$tmp/out" "$repo/engines/radiance/kernels/libr3/rewrites.txt" \
        "$repo/engines/radiance/kernels/libr3/native"; then
    echo "R3_REWRITE_CHECK_FAILED pin=${pin:0:7}"
    exit 1
fi
n=$(find "$repo/engines/radiance/kernels/libr3/native" -maxdepth 1 -name '*.rw' -type f | wc -l)
echo "R3_REWRITE_CHECK_OK pin=${pin:0:7} rw=$n"
