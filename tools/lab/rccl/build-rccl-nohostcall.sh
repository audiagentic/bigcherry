#!/bin/bash
# PGC14: build RCCL (ROCm/rccl checkout on Brutus) without hostcall-dependent kernels so it runs on GPUs
# without PCIe atomics (RX 6900 XT on a chipset port). Stock ROCm 7.2.4 RCCL's ncclDevKernel_Generic_{1,2,4}
# (and Debug variants) declare hidden_hostcall_buffer for gfx1030; ROCr refuses to dispatch them there.
# Then verify no kernel in the new library requests hostcall, and run rccl-tests on R9700 + 6900 XT.
# Usage: build-rccl-nohostcall.sh <out-dir>
set -u
out=$1; mkdir -p "$out"
src=$HOME/rccl-heterogeneous-src/rccl
bld=$HOME/rccl-heterogeneous-src/build-nohostcall
inst=$HOME/rccl-heterogeneous-src/install-nohostcall
LLVM=/opt/rocm/llvm/bin
export PATH=/opt/rocm/bin:$PATH
cmake -S "$src" -B "$bld" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$inst" \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DGPU_TARGETS="gfx1100;gfx1201;gfx1030" \
  -DCOLLTRACE=OFF -DENABLE_MSCCL_KERNEL=OFF -DENABLE_MSCCLPP=OFF -DENABLE_NPKIT=OFF -DBUILD_TESTS=OFF \
  -DCMAKE_CXX_FLAGS="-DNDEBUG -mprintf-kind=buffered" > "$out/configure.log" 2>&1 || { tail -30 "$out/configure.log"; exit 1; }
cmake --build "$bld" -j 16 > "$out/build.log" 2>&1 || { tail -40 "$out/build.log"; exit 1; }
cmake --install "$bld" > "$out/install.log" 2>&1 || { tail -20 "$out/install.log"; exit 1; }
lib=$(ls "$inst"/lib/librccl.so.1.* | head -1)
cp "$lib" "$out/lib.so"
"$LLVM/llvm-objcopy" --dump-section=.hip_fatbin="$out/rccl.fat" "$out/lib.so" "$out/lib2.so"
for t in gfx1030 gfx1100 gfx1201; do
  "$LLVM/clang-offload-bundler" --type=o --input="$out/rccl.fat" --targets=hipv4-amdgcn-amd-amdhsa--$t \
    --output="$out/$t.co" --unbundle && "$LLVM/llvm-readelf" --notes "$out/$t.co" > "$out/$t.meta"
  echo "$t hostcall kernels: $(grep -c hidden_hostcall_buffer "$out/$t.meta")"
done
rm -f "$out/lib.so" "$out/lib2.so" "$out/rccl.fat" "$out"/*.co
B=$HOME/rccl-heterogeneous-src/rccl-tests/build/all_reduce_perf
for vis in 2,3 0,1; do
  echo "== rccl-tests devices $vis"
  LD_LIBRARY_PATH="$inst/lib" HIP_VISIBLE_DEVICES=$vis timeout 300 "$B" -g 2 -b 1K -e 64M -f 4 -n 10 -w 3 2>&1 \
    | grep -E "Librccl|FATAL|^\s+[0-9]+ " | head -16
done
echo RCCL_BUILD_DONE
