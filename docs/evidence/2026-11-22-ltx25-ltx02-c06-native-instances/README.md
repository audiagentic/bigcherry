# LTX02 C06: native three-rank self-attention instance gate

**Date:** 2026-11-22. **Result: PASS (native host initialization only).**
The LTX 2.5 IQ8R TP3 self-attention source-backed declaration test now
retains the *same* three native `RadBuilder` objects and completes
`Builder::finish()` concurrently, allowing the original Radiance exact
collective kernels to initialize at their normal three-rank rendezvous.

This is **not** an issued attention graph, checkpoint-weight residency,
GPU numerical parity, serving activation, or generated video. The
upstream `world_size > 1` production refusal is unchanged.

## Reused path and coverage

- Reuses `tools/lab/radiance/ltx-tp3/native-c01-gate/ltx_c02_lin_declare_test.cpp`
  and the existing C06 `Attn::declare`, `C01Slice`, `RadFile`,
  `RadBuilder`, and native `Registry`. There is no alternate
  attention builder, kernel registry, model reader, or collective.
- Adds `--c06-instances` to the existing test executable and a
  timeout-bounded `ltx_c06_native_instance_rendezvous` CTest. The
  established `--c06` graph-only matrix remains unchanged.
- Uses checkpoint-derived video and audio self-attention geometry
  from DiT block 0, both legal 8/8/16 and 12/12/8 splits, and all
  three rank contexts per case.
- Four distinct three-rank graphs finish with no errors or missing
  host kernel bands. The native registry reports **36 initialized
  instances** in total, nine per three-rank graph: Q, K and
  output exact all-reduce on each rank. The test refuses a missing
  instance, failed rank finish, incomplete kernel coverage or
  mismatched head ownership.
- Instance initialization failures are caught per rank rather than
  terminating the process; rendezvous is performed concurrently,
  as required by the existing native collective implementation.

## Source identity

Frozen upstream Radiance 2.2 / ABI44 `arch/ltx2/ltx2.cpp` SHA-256:
`543490eb70829693f945233d1b66e2d6d87b5e6a2817a72233fa0649218e6b53`.

Source-managed diagnostic `rad0060_ltx2_tp3_linear_declare/patch.py`
SHA-256: `964d6dea9001b052092aca3124f188dc4766f89cfbed8c9ae4ac6e5aadf87ba2`.
The tests compile the previously staged patched architecture from
that frozen source, with the exact version/source-pinned native SDK.
No production recipe or patch source is changed.

## Verification

- **7/7 native model-backed CTests passed**: real checkpoint weight
  receipt, C02 linear declarations, C04 deferred reduction, C05 graph
  declarations, C06 graph declarations, the new C06 instantiated
  three-rank rendezvous, and actual issued C05 host Q/K RMS.
- **12/12 focused Python tests passed**, with the actual read-only
  IQ8R container and frozen upstream source provided (no skips).
- `git diff --check` passed. No GPU lease was used in this
  host-only slice.

Run the new CTest using the existing C01 native build setup,
with the real `LTX_IQ8R_MODEL`, `LTX_NATIVE_CORE_LIB`,
`LTX_SOURCE_ROOT`, `LTX_STAGE_ROOT`, and `LTX_REFERENCE_LIB`
configuration values:

```sh
ctest --test-dir "${BC_WORK_ROOT}/ltx02-c06-native-instances-build" \
  -R '^ltx_c06_native_instance_rendezvous$' --output-on-failure
```

## Remaining LTX02 gates

The next step is **actual C06 model-backed graph issue** with Radiance's
existing weight loader and native `RadCtx` and a captured TP1
checkpoint-activation oracle. That includes fused IQ8R QKV, full-head
Q/K RMS, local RoPE, dense attention, head gate, one exact output
reduction and once-only bias on gfx1100/gfx1100/gfx1201. Check
startup refusal, missing kernels, missing rank and bounded cancellation
under a real GPU lease before claiming LTX02 completion.

Neither the 36 initialized host instances nor the already proven C05
AQL seeded-input parity substitutes for that gate. Shared group
lifetime/fault recovery remains RUM06-owned, and generic AQL
qualification remains RUM13-owned; do not add LTX-specific runtime
wrappers for either.
