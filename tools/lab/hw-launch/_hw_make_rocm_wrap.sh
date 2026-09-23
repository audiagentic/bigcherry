#!/bin/bash
# Build a wrapper ROCm prefix that mirrors /opt/rocm but exposes
# bin/clang + bin/clang++ as wrapper SCRIPTS that exec the real
# amdclang/amdclang++ (argv[0] becomes .../amdclang, passing amdclang's
# name check). The campaign hardcodes $HIP_PATH/bin/clang and
# -DCMAKE_PREFIX_PATH=$HIP_PATH.
set -e
REAL=/opt/rocm
WRAP=/mnt/vault/tmp/bc-rocm
rm -rf "$WRAP"
mkdir -p "$WRAP/bin"
# Symlink every top-level entry of the real ROCm tree EXCEPT bin.
for entry in "$REAL"/*; do
	name=$(basename "$entry")
	if [ "$name" = "bin" ]; then
		continue
	fi
	ln -s "$entry" "$WRAP/$name"
done
# Symlink every entry of the real bin/.
for tool in "$REAL"/bin/*; do
	ln -s "$tool" "$WRAP/bin/$(basename "$tool")"
done
# clang / clang++ as wrapper scripts (exec the real amdclang so argv[0] passes).
printf '#!/bin/bash\nexec /opt/rocm/lib/llvm/bin/amdclang "$@"\n' >"$WRAP/bin/clang"
printf '#!/bin/bash\nexec /opt/rocm/lib/llvm/bin/amdclang++ "$@"\n' >"$WRAP/bin/clang++"
chmod +x "$WRAP/bin/clang" "$WRAP/bin/clang++"
echo "=== wrapper ready ==="
ls -la "$WRAP/bin/clang" "$WRAP/bin/clang++"
echo "=== compile test ==="
cd /tmp
echo 'int main(){return 0;}' >_clangtest.c
"$WRAP/bin/clang" _clangtest.c -o _clangtest 2>&1 | head -5
if test -x /tmp/_clangtest; then
	/tmp/_clangtest
	echo "COMPILE_TEST_OK (exit=$?)"
else
	echo "COMPILE_TEST_FAILED"
fi
