---
id: VTP09
order: 8
plan: run-dense-27b-virtual-tp
state: in_progress
created-at: '2026-11-10T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# gfx1100 IQ6R M=64 dispatch crossover: exact tiled overlay

> Archived 2026-11-12 from branch `exp/xtx27-prefill-opt` (git tag `archive/exp-xtx27-prefill-opt`), which never merged. Its lab programs (tools/lab/radiance/xtx27_*), tests and early patch drafts are in that tag. Superseded by VTP14 (native Radiance TP3, completed 2026-11-12), which kept these programs as reference oracles only.

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

## Completed follow-up: crossover below M64 (2026-11-10)

The paired A/B experiments described here were executed and are
recorded below. The first M64-only commit is preserved; a separate
model-shape-gated M17-63 extension is being committed independently.

Do not merge this branch based only on microkernels if production
dispatch is expected to cover a different weight encoding or
full-model flow; document and test the real operator selection.

### Crossover measurements: 36/36 qualified paired cases

The same complete gfx1100 plugin and two dedicated scratch selftest
executables ran ABBA narrow/tiled/tiled/narrow comparisons with the
original Qwen3.8-27B FFN shapes (gate/up N=10240 K=5120; down N=5120
K=5120). Both XTXs were tested at M=16,17,20,24,28,31,32,48,63.
Every case used native IQ6R-V4 B=6, nine iterations and 256-MiB cold
weight flushing. Exact immutable per-case logs and receipt hashes:
`${BC_WORK_ROOT}/xtx27-prefill/iqr-layout-fix/iqr-crossover/measurements/`.
The two resume-safe summary receipts are `sweep-summary.json`
(16/16 M16,32,48,63) and `sweep-near-summary.json` (20/20
M17,20,24,28,31), zero failed or partial cases.

| M | Gate/up tiled speedup, XTX0/XTX1 | Down tiled speedup, XTX0/XTX1 |
| ---: | ---: | ---: |
| 16 | 0.33x / 0.33x | 0.31x / 0.32x |
| 17 | 1.75x / 1.73x | 1.48x / 1.47x |
| 20 | 1.70x / 1.76x | 1.43x / 1.41x |
| 24 | 1.73x / 1.75x | 1.42x / 1.41x |
| 28 | 1.70x / 1.72x | 1.47x / 1.47x |
| 31 | 1.75x / 1.76x | 1.53x / 1.54x |
| 32 | 1.73x / 1.71x | 1.57x / 1.48x |
| 48 | 3.53x / 3.65x | 3.16x / 3.26x |
| 63 | 4.51x / 4.59x | 4.18x / 4.95x |

**Important negative control:** at M=17, N=512, K=1024, narrow ran
49.5 us and tiled averaged 87.8 us on XTX0; a globally higher
priority M17..63 tiled row would regress small shapes by ~1.77x.
At M=64, the same small N/K shape favoured tiled (413.1 vs
91.8 us), supporting retention of the already-tested M64 rule.

### Safe gfx1100 extension: exact measured 27B geometry only

Added a second disjoint row `gemm_iqr_tiled_m17_63_gfx11` in
`engines/radiance/components/libr3/r3_registry.cpp`, which copies
the existing tiled row's hooks and original shape constraints,
and additionally requires **17 <= M <= 63, K == 5120,
N in {5120,10240}**. It has narrow-priority+1.
For all other N/K (including the demonstrated small-shape
regression), the narrow row remains eligible and preferred.
M<=16 stays narrow; M=64 retains the separately qualified
`gemm_iqr_tiled_m64_gfx11`; M>=65 uses native tiled.

This is intentionally more conservative than a global M17
crossover. It does not affect the gfx1201 plugin or modify pinned
Radiance. Upstream numeric IN constraint matching was checked at
`core/util/rad_util.cpp` and the native band builder.

The full gfx1100 plugin rebuilt in two isolated scratch build
actions. Binary ABI/constraint/pointer/selection validator
`tools/lab/radiance/xtx27_iqr_m64_registry_check.py`
successfully checked 29 selection and negative-fallback bands;
final plugin SHA-256:
`387309a3c54f453d30a0020a05c485e82ab0b3613ba2ee7440dff823852093a8`.
Receipt:
`${BC_WORK_ROOT}/xtx27-prefill/iqr-layout-fix/iqr-crossover/registry-m17-64-shapegated.json`.
The seven card-free contract tests include missing row, bad
constraint, pointer, priority, ABI, small-shape fallback and
no-overwrite cases.

### Numerical qualification of the actual overlay rows

The rebuilt plugin's **new early row was invoked by name** using a
dedicated scratch `r3_selftest`, including five explicitly legal
Qwen27 FFN shapes: M17 N5120/10240 K5120; M32 N5120 K5120;
M48 N10240 K5120; M63 N5120 K5120. **192 / 192 numerical checks
passed per XTX, zero failures**, including the extra crossover
cases. Underlying tiled IQR tests also passed 435 checks per XTX,
but not all synthetic shapes were legal for the shape-gated row.
Receipts:
`${BC_WORK_ROOT}/xtx27-prefill/iqr-layout-fix/iqr-crossover/numeric/model-oracle/runs/`.

The existing M64 overlay was requalified **129 / 129 checks per
XTX** against the final shape-gated plugin, zero failures:
`${BC_WORK_ROOT}/xtx27-prefill/iqr-layout-fix/m64-overlay-smoke/runs/`.
All four final overlay runs used the identical plugin hash above.
The GPU lease was held exclusively for each run; no production
binary, model or serving process was modified.

### Remaining release gates

These tests demonstrate per-row numerical agreement and strong
model-shaped kernel speedups, **not** automatic row selection within
the full production engine. Calibrated 27B IQ6R weights and a
real mixed-architecture virtual TP4 runtime remain unavailable.
Require an actual IQ6R model-format and end-to-end graph-selection
trace, BF16 quality comparisons, and full runtime/VRAM/collective
tests before production rollout.
