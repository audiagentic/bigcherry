# LTX C07/C08 native three-rank collective probe

This **opt-in GPU lab** exercises Radiance 2.2's **existing** `all_reduce` kernel entries and the **actual** BigCherry `bc_shared_slot.cpp` rendezvous service. It adds **no production all-reduce implementation**, no engine admission, no patch recipe, no model loader and no new transport.

The checked BF16 payloads have C06's 4096-wide video and 2048-wide audio head geometry. On every rank, input columns are either disjoint local-head Q/K slices (the proposed C05 zero-packed exact exchange) or overlapping output projection partial sums. The harness verifies the ascending-rank f32 sum rounded once to BF16, **bit for bit**, for both 8/8/16 and 12/12/8 splits. It tests in-place/out-of-place and a one-row transfer. C07 initially checked seven patterns x three ranks = **21 outputs**; the current expanded C08 harness checks eleven patterns x three ranks = **33 independent exact BF16 output comparisons** per completed run, including 16- and 64-row batches.

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

`--devices 0,0,0 --visible 2` creates three ranks on **one R9700**, with independent nonblocking HIP streams. `--devices 0,0,1 --visible 0,1` creates three ranks across **two XTX** devices. These are controls, not the target heterogeneous layout. The final command covers **two XTX and one R9700** and **fails in direct-HIP C07 mode**, where the normal engine AQL backend is absent. **C08 AQL-engine mode passes** using the same unmodified kernels. The harness does not suppress this failure, retry it with a fabricated CPU collective or claim it as a passing test.

See `docs/evidence/2026-11-22-ltx25-c07-native-mixed-ar/README.md` for outcomes and the diagnostic next gate. Never start the GPU runner in CI or without a lease; do not change production native rendezvous to satisfy this standalone harness.


## C08: real Radiance AQL dispatch (GPU lease required)

The optional C08 executable links the exact **HIP-enabled** Radiance 2.2 SDK
core whole, including AQL dispatch and per-device HIP fat-binary registration.
The configure check requires the matching 2.2 source and a HIP-on native build;
a CPU-only core or a different source/version is rejected. The runner still
calls the existing native kernel rows and shared-slot rendezvous; the test
uses the engine's AQL streams and collective launch fencing. It does **not**
run a production serving session, change kernel selection or enable LTX TP3.

Build once with the native CPU core and the matching HIP-on SDK core:

    cmake -S tools/lab/radiance/ltx-tp3/mixed-ar-gate \
      -B "$BC_WORK_ROOT/ltx-c08-build" \
      -DLTX_SOURCE_ROOT="$RADIANCE_22_SOURCE" \
      -DLTX_NATIVE_CORE_LIB="$RADIANCE_22_NATIVE_CORE" \
      -DLTX_ENGINE_NATIVE_CORE_LIB="$RADIANCE_22_HIP_CORE" \
      -DCMAKE_PREFIX_PATH=/opt/rocm
    cmake --build "$BC_WORK_ROOT/ltx-c08-build" --parallel 2

Under an acquired GPU lease, run both supported exact host-mailbox variants
on the three visible physical cards:

    python3 tools/lab/radiance/ltx-tp3/mixed-ar-gate/run_mixed_ar.py \
      --binary "$BC_WORK_ROOT/ltx-c08-build/ltx25_mixed_ar_engine_gate" \
      --sdk "$RADIANCE_22_SDK" --output "$BC_WORK_ROOT/ltx-c08-proof" \
      --devices 0,1,2 --visible 0,1,2 \
      --row ar_oneshot_3rank_exact_host

    python3 tools/lab/radiance/ltx-tp3/mixed-ar-gate/run_mixed_ar.py \
      --binary "$BC_WORK_ROOT/ltx-c08-build/ltx25_mixed_ar_engine_gate" \
      --sdk "$RADIANCE_22_SDK" --output "$BC_WORK_ROOT/ltx-c08-proof" \
      --devices 0,1,2 --visible 0,1,2 \
      --row ar_twoshot_3rank_exact_host

The 2026-11-22 runs passed both rows at **33/33 exact BF16 comparisons**
each, across gfx1100 + gfx1100 + gfx1201. Device peer access was unavailable,
so they used the existing host mailbox (not direct P2P). See
docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/README.md.
This closes the native mixed-card collective gate, **not** complete LTX
self-attention, TP1 parity or production admission. Always release the lease.
