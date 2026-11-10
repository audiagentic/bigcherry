---
id: VTP09
order: 8
plan: run-dense-27b-virtual-tp
state: active
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# gfx1100 IQ6R M=64 dispatch crossover: exact tiled overlay

## Measured problem and constrained solution

Pinned Radiance in `libr4d/r4d_rows.cpp` registers
`gemm_iqr_nt_m64` for M<=64, K divisible by 256, N divisible by 16
(priority 60); `gemm_iqr_tiled` is eligible for any M, K divisible
by 256 and N divisible by 128 (priority 40). At **M=64**, selection
therefore chooses the narrow kernel for otherwise legal tiled shapes,
including Qwen3.8-27B FFN projection rows.

An isolated original/tiled/tiled/original A/B on both gfx1100 XTXs,
M=64, native IQ6R B=6, 9 iterations, 256-MiB cold flushing,
N=10240 K=5120 gate/up and N=5120 K=5120 down, gave:

| GPU/projection | Narrow mean us | Tiled mean us | Speedup |
| --- | ---: | ---: | ---: |
| XTX0 gate/up | 1927.4 | 436.7 | 4.41x |
| XTX0 down | 1168.6 | 287.1 | 4.07x |
| XTX1 gate/up | 2035.6 | 418.0 | 4.87x |
| XTX1 down | 1250.2 | 290.4 | 4.31x |

Receipts in scratch:
`${BC_WORK_ROOT}/xtx27-prefill/iqr-layout-fix/m64-force-tiled/measurements/`.
Tiled M=64 numerical suites (129 checks per XTX) passed against
Radiance's corresponding tiled INT8 reference. These measurements
are kernel-only and *not* a direct cold-IQ6R versus warm-W8A8
speed comparison or full-engine throughput claim.

## Implementation

`engines/radiance/components/libr3/r3_registry.cpp` injects
`gemm_iqr_tiled_m64_gfx11` as an additional *gfx11-only*
`gemm_nt_q` row. It references **exactly the existing tiled kernel's
launch, scratch, layout and relayout**; it copies every tiled shape
constraint and ADDS `M == 64`; its priority is one above the narrow
row. It does not change pinned upstream, original kernel names,
gfx1201 `libr4d`, or legality at small N that fits N16 but not N128.

The override **fails closed** if required upstream rows are missing,
stubbed, have incompatible ABI or changed constraints. Query
`r3_iqr_m64_tiled_override()` returns 1 only if installed. The
`rad_kernel_concurrent/collective` callbacks forward the new row's
metadata using the original upstream tiled row index.

`tools/lab/radiance/xtx27_iqr_m64_registry_check.py` opens a
**built** libr3.so through the actual C ABI and rejects incorrect
overlays. Its report checks 10 boundary/legality cases: M=1, 8,
32, 63, 64, 65 and 128; gate/up/down N; N16/N48 fallback; exact
tiled function pointers and the original narrowing constraints.
`tools/tests/lab/test_xtx27_iqr_m64_registry_check.py`
contains seven negative/positive card-free tests.

## Hardware qualification

The complete gfx1100 plugin was rebuilt in isolated scratch at
`${BC_WORK_ROOT}/xtx27-prefill/iqr-layout-fix/gfx11-full/`.
The ABI-level checker passed on the rebuilt binary:
`${BC_WORK_ROOT}/xtx27-prefill/iqr-layout-fix/m64-registry-check-v1.json`.

A **separate scratch selftest** invoked the *actual new registry row*
by name on each XTX (not merely the old tiled row), using the
unchanged rebuilt libr3.so. Both runs passed **129/129 M=64 numerical
tests**, zero failures, with receipts under
`${BC_WORK_ROOT}/xtx27-prefill/iqr-layout-fix/m64-overlay-smoke/runs/`.
The rebuilt plugin SHA-256 is
`3be695d0523ec9ecfddac48c18fd646fb18340ec04a59234a39ed6591d7db905`.

The production plugin was NOT installed and the current engine's
automatic selector was not yet exercised against a full 27B
converted IQ6R checkpoint. The ABI check verifies registered
priority and shape selection among these rows, and the standalone
numeric test proves the row works when selected by name. Keep
full-engine selection/profiling as a required integration gate.

## Next bounded experiment: crossover below M64

Benchmark original narrow versus the **same** tiled kernel at
M=16, 32, 48, 63 on both XTXs, separately for FFN gate/up and
down, under an exclusive GPU lease. Use paired/interleaved cold
measurements and a strict numerical reference before extending
the overlay to any lower batch size. The current production
change intentionally covers **only M64**; M<64 is unchanged.

Do not merge this branch based only on microkernels if production
dispatch is expected to cover a different weight encoding or
full-model flow; document and test the real operator selection.
