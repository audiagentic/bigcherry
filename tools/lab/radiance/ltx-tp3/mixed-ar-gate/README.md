# LTX C07 native three-rank collective probe

This **opt-in GPU lab** exercises Radiance 2.2's **existing** `all_reduce` kernel entries and the **actual** BigCherry `bc_shared_slot.cpp` rendezvous service. It adds **no production all-reduce implementation**, no engine admission, no patch recipe, no model loader and no new transport.

The checked BF16 payloads have C06's 4096-wide video and 2048-wide audio head geometry. On every rank, input columns are either disjoint local-head Q/K slices (the proposed C05 zero-packed exact exchange) or overlapping output projection partial sums. The harness verifies the ascending-rank f32 sum rounded once to BF16, **bit for bit**, for both 8/8/16 and 12/12/8 splits. It tests in-place/out-of-place and a one-row transfer. Seven patterns x three ranks = **21 independent output comparisons** per completed run.

## Configure and build (CPU, no lease)

Use a **frozen 2.2 source**, its matching native core archive, and the installed ROCm HIP package. The CMake test rejects native cores whose build receipts do not identify the same 2.2 source.

```sh
cmake -S tools/lab/radiance/ltx-tp3/mixed-ar-gate -B "${BC_WORK_ROOT}/ltx-c07-build" \
  -DLTX_SOURCE_ROOT="${RADIANCE_22_SOURCE}" \
  -DLTX_NATIVE_CORE_LIB="${RADIANCE_22_NATIVE_CORE}" \
  -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build "${BC_WORK_ROOT}/ltx-c07-build" --parallel 2
```

## GPU run, only under a valid GPU lease

The runner enforces an 80-second timeout, preserves a log and JSON receipt for each (row, device map, visible card set) and propagates signal exits. The device identifiers are **ordinal indices relative to the visible set**. The SDK must contain the exact ABI44 kernel libraries `libr3.so` and `libr4d.so`. Neither `libr3d.so` nor `libref.so` substitutes for them here.

```sh
agent-lease acquire gpu --ttl 25m --est 10m --wait

python3 tools/lab/radiance/ltx-tp3/mixed-ar-gate/run_mixed_ar.py \
  --binary "${BC_WORK_ROOT}/ltx-c07-build/ltx25_mixed_ar_gate" \
  --sdk "${RADIANCE_22_SDK}" --output "${BC_WORK_ROOT}/ltx-c07-proof" \
  --devices 0,0,0 --visible 2

python3 tools/lab/radiance/ltx-tp3/mixed-ar-gate/run_mixed_ar.py \
  --binary "${BC_WORK_ROOT}/ltx-c07-build/ltx25_mixed_ar_gate" \
  --sdk "${RADIANCE_22_SDK}" --output "${BC_WORK_ROOT}/ltx-c07-proof" \
  --devices 0,0,1 --visible 0,1

python3 tools/lab/radiance/ltx-tp3/mixed-ar-gate/run_mixed_ar.py \
  --binary "${BC_WORK_ROOT}/ltx-c07-build/ltx25_mixed_ar_gate" \
  --sdk "${RADIANCE_22_SDK}" --output "${BC_WORK_ROOT}/ltx-c07-proof" \
  --devices 0,1,2 --visible 0,1,2

agent-lease release gpu
```

`--devices 0,0,0 --visible 2` creates three ranks on **one R9700**, with independent nonblocking HIP streams. `--devices 0,0,1 --visible 0,1` creates three ranks across **two XTX** devices. These are controls, not the target heterogeneous layout. The final command covers **two XTX and one R9700** and is **currently expected to fail** with SIGSEGV on the first native collective launch. The harness does not suppress this failure, retry it with a fabricated CPU collective or claim it as a passing test.

See `docs/evidence/2026-11-22-ltx25-c07-native-mixed-ar/README.md` for outcomes and the diagnostic next gate. Never start the GPU runner in CI or without a lease; do not change production native rendezvous to satisfy this standalone harness.
