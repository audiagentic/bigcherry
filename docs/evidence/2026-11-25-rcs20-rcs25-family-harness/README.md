# One family against the full build: the unit harness on the IQR family (RCS20, RCS25)

The unit harness (`tools/lab/radiance/unit_harness.py`) builds a named family from the eight family targets into a small test plugin and checks it
against a shipped release. Run: `run --family iqr --arch gfx1100 --release bc-radiance-2020.13.0`, production patch composition, Release build type.

## Build time

| Build | Time |
|---|---:|
| IQR family plugin (4 units) plus the selftest binary, selftest cache present | 155 s |
| the same with the selftest cache cold (selftest binary built once per source and architecture) | 225 s |
| full lane build in CI, radiance build job, ccache about 86% hits | 25 to 28 min |

The family build compiles 4 units; the lane build compiles 245 HIP commands. The family build does not use ccache.

## What the family run checked against 2020.13.0

| Check | Result |
|---|---|
| Output hashes of the family's selftest case over its matrix and the V4 trunk shapes and row counts | 555 compared, 0 differ, 0 missing |
| Selftest oracle failures, reference and candidate | 0 and 0 |
| Resource records of 3,521 kernels | 3,516 unchanged, 0 added, 0 removed, 5 changed |
| Replay of 52 rows (two plugins in ABBA order, cold, windows until two medians agree) | several rows flip between regress and same from run to run, many unconverged; no stable difference |

The 5 changed records are the same five `iqr_prefill_kernel` forms in two runs, with and without the Release build type: 236 to 240 vector registers, 2 or 3 spills.
The build type is therefore not their cause, and the production patch composition is already what the harness builds. They are recorded as a known difference between a
cold build of the source and the shipped library (BU03 follow-up); the output hashes are identical.

## Family dependency closure (IQR family, gfx1100)

From the build's own dependency records, project headers only (compiler and system headers excluded):

| Unit | Project headers |
|---|---:|
| r4d_gemm_iqr | 21 |
| r4d_gemm_iqr_p1, p2, p3 | 22 each |
| support units built for the selftest's int8 reference rows (r4d_gemm_fp8a8, r4d_gemm_w8a8_tiled, r4d_keyed) | 19, 13, 5 |

The whole family set uses 29 distinct project headers; the two shared by every unit are `r3_compat.h` and `r3_mma16.h`. A change to either rebuilds every unit of the family;
a change to one IQR unit's own source rebuilds that unit only.

## Notes

- The harness work root is shared by every run (selftest cache `selftest-<arch>-<key>`); remove only a run's own dated directory.
- The family run is the discrete proof for a kernel change; the full plugin build, Gate E and serving runs are for a release.
