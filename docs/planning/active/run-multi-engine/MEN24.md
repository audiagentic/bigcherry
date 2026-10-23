---
id: MEN24
order: 24
plan: run-multi-engine
state: pending
created-at: '2026-10-21T00:30:00+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Faster, less fragile engine bumps

## Description

The radiance 1.9.0 bump (2026-10-20) took about an hour, but measured compile time is about 6 minutes on dev (20 cores): engine about 75 s wall, libr3 about 4.7 min (one unit, r4d_gemm_mxfp4a8_decode, alone 264 s and the critical path), moe-remote 10 s, ngram-helper 2 s. The rest is pipeline: the engine is built twice (compile gate, then the lane build for the CPU smoke), there is no compiler cache, the patch-test step runs the whole offline patch suite (208 files, 1,444 tests, about 140 s, only 2 files about radiance), and selftest/bench binaries are built in the gates. The bump also stopped on 10 libr3 native rewrite blocks that upstream had moved: in-source text rewrites of libr4d break on every release, while our own components (moe-remote, ngram-helper) needed only the ABI rebuild.

## Steps

1. Record per-phase wall time in the pin-bump run record (the 1.9.0 run's timestamped log is the first sample).
2. Build once: the lane-build phase reuses the compile gate's build when source, options and targets are identical, instead of building again.
3. Compiler cache: ccache as CMAKE_C/CXX/HIP_COMPILER_LAUNCHER for lane and bump builds, cache directory on the workspace volume; record hit rate.
4. Engine-scoped bump tests: run_patch_tests runs only the bumped engine's patch and component tests (a radiance bump does not run llama.cpp patch tests), plus the component ctest targets.
5. Gates skip selftest and benchmark targets (r4d_selftest 51 s); they stay in the full lane build.
6. Split or reorder the slowest libr3 unit so it is not the build's tail (build it first, or split mxfp4 decode by shape).
7. Nightly drift check: run the native-rewrite drift and patch-rebase checks against upstream main without a GPU, so anchors that break are seen when upstream changes them, not on bump day.
8. Fewer in-source rewrites: move self-contained enhancements out of libr3/native/*.rw into kernel files of our own with their own rows (candidates: the all-reduce timing hook and 4-bit wire, the pipelined norm), leaving rewrites only for what must change inside upstream code. New enhancements (RAD15) are written as rows from the start.
9. Keep the gate suite green: failing tests on main are fixed or precisely skipped, never waved through as pre-existing.

## Validation

The next radiance bump records per-phase times; target under 15 minutes end to end when no rewrites drift; one engine build per bump; a radiance bump runs no llama.cpp tests; the nightly drift job reports on upstream main.

## Notes

Measured 2026-10-20 from the .ninja_log files of lane build 4babe498 and a local run of tools/tests/patch. Related: MEN10-MEN19 (managed engine workflow), RAD15 (engine rows instead of rewrites).

## Change Log

- 2026-10-21T00:30:00+00:00 (created-by): Created by agent
